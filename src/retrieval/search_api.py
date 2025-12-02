#!/usr/bin/env python3
import os
import re
import json
import time
import torch
import torch.nn.functional as F
import psycopg
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# --- Config ---
PG_HOST = os.environ.get("PGHOST", "db")
PG_DB = os.environ.get("PGDATABASE", "gis")
PG_USER = os.environ.get("PGUSER", "gis")
PG_PASS = os.environ.get("PGPASSWORD", "gis")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))

# Collections
LISTING_COLLECTION = "hemnet_listings_v1"
CHIP_COLLECTION = os.environ.get("QDRANT_COLLECTION", "sweden_demo_v0")

# --- MODEL CONFIG ---
# Qwen 2.5 7B Instruct
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Paths
STACK_ROOT = Path(os.environ.get("GIS_STACK_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", STACK_ROOT / "data"))
META_PATH = Path(os.environ.get("METADATA_PATH", DATA_ROOT / "chips" / "metadata.parquet"))
LLM_CONFIG_PATH = Path(__file__).parent / "llm_config.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Globals ---
ml_models = {}
legacy_meta = None
llm_config = {}

def log(event, **kw):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    msg = " ".join(f"{k}={repr(v)}" for k,v in kw.items())
    print(f"{ts} {event} {msg}".strip(), flush=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global legacy_meta, llm_config
    print(f"--- STARTUP: Loading models on {device} ---")
    
    if LLM_CONFIG_PATH.exists():
        with open(LLM_CONFIG_PATH, "r") as f:
            llm_config = json.load(f)
    else:
        llm_config = {"system_prompt": "Output JSON.", "examples": []}

    # 1. Retrieval Models
    ml_models['text'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    ml_models['clip'] = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    ml_models['clip_proc'] = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    ml_models['qdrant'] = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True, timeout=300)

    # 2. Legacy Metadata
    if META_PATH.exists():
        print(f"--- STARTUP: Loading legacy metadata from {META_PATH} ---")
        legacy_meta = pd.read_parquet(META_PATH)[["png_path", "lon", "lat"]].reset_index().rename(columns={"index": "row"})
    else:
        legacy_meta = pd.DataFrame(columns=["row", "png_path", "lon", "lat"])

    # 3. LLM (Qwen 7B - FP16 Mode - Multi-GPU)
    # We removed BitsAndBytesConfig to run in native Float16.
    # This requires ~14GB VRAM, which fits across your two 11GB cards (22GB Total).
    print(f"--- STARTUP: Loading LLM ({LLM_MODEL_ID} FP16)...")
    try:
        ml_models['llm_tok'] = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
        if ml_models['llm_tok'].pad_token is None:
            ml_models['llm_tok'].pad_token = ml_models['llm_tok'].eos_token
        
        ml_models['llm'] = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID, 
            torch_dtype=torch.float16,  # Native FP16 (No quantization errors)
            device_map="auto",          # Automatically spreads across GPU 0 and GPU 1
            trust_remote_code=True
        )
        print("--- STARTUP: LLM Ready ---")
    except Exception as e:
        print(f"!!! LLM LOAD FAILED: {e}")
    
    yield
    ml_models.clear()
    print("--- SHUTDOWN: Models cleared ---")

app = FastAPI(title="Retrieval Agent API", lifespan=lifespan)

# --- Schemas ---
class ListingFilters(BaseModel):
    municipality: Optional[str] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    min_rooms: Optional[float] = None
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    radius_km: Optional[float] = None

class SearchRequest(BaseModel):
    query: str
    topk: int = 10
    filters: Optional[ListingFilters] = None

class HybridSearchRequest(BaseModel):
    text_query: str
    image_query: str
    topk: int = 10
    filters: Optional[ListingFilters] = None

class AgentQueryRequest(BaseModel):
    prompt: str
    topk: int = 10


class PostGISRegion(BaseModel):
    mode: Literal["circle", "none"] = "none"
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    radius_km: Optional[float] = None


class HardFilters(BaseModel):
    price_floor_str: Optional[str] = None
    price_ceiling_str: Optional[str] = None
    min_rooms_num: Optional[float] = None
    municipality: Optional[str] = None


class QdrantFilterClause(BaseModel):
    field: Literal["asking_price_sek", "number_of_rooms", "municipality"]
    type: Literal["range", "match"]
    gte: Optional[float] = None
    lte: Optional[float] = None
    value: Optional[str] = None


class LLMIntent(BaseModel):
    semantic_search: str
    postgis_region: PostGISRegion = Field(default_factory=PostGISRegion)
    hard_filters: HardFilters = Field(default_factory=HardFilters)
    qdrant_filters: List[QdrantFilterClause] = Field(default_factory=list)

# --- Helper Functions ---

def get_geo_polygon(lat, lon, radius_km):
    try:
        conn_str = f"host={PG_HOST} dbname={PG_DB} user={PG_USER} password={PG_PASS}"
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ST_AsGeoJSON(ST_Simplify(ST_Buffer(ST_MakePoint(%s, %s)::geography, %s)::geometry, 0.0001));", (lon, lat, radius_km * 1000.0))
                row = cur.fetchone()
                return json.loads(row[0]) if row else None
    except Exception as e:
        log("postgis_error", error=str(e))
        return None

def build_qdrant_filter(f: ListingFilters):
    if not f: return None
    conds = []
    
    if f.municipality: 
        conds.append(models.FieldCondition(key="municipality", match=models.MatchText(text=f.municipality)))
        
    if f.min_price or f.max_price: 
        conds.append(models.FieldCondition(key="asking_price_sek", range=models.Range(gte=f.min_price, lte=f.max_price)))
    
    if f.min_rooms: 
        conds.append(models.FieldCondition(key="number_of_rooms", range=models.Range(gte=f.min_rooms)))
    
    if f.center_lat and f.center_lon and f.radius_km:
        poly = get_geo_polygon(f.center_lat, f.center_lon, f.radius_km)
        if poly and poly['type'] == 'Polygon':
            pts = [models.GeoPoint(lon=p[0], lat=p[1]) for p in poly['coordinates'][0]]
            conds.append(models.FieldCondition(key="location", geo_polygon=models.GeoPolygon(exterior=models.GeoLineString(points=pts), interiors=[])))
            log("postgis_filter_active", lat=f.center_lat, lon=f.center_lon, radius=f.radius_km)
    
    return models.Filter(must=conds) if conds else None

def encode_text(q):
    with torch.inference_mode(): return F.normalize(ml_models['text'].encode(q, convert_to_tensor=True), p=2, dim=0).cpu().tolist()

def encode_image(q):
    with torch.inference_mode():
        inputs = ml_models['clip_proc'](text=[q], return_tensors="pt", padding=True).to(device)
        return F.normalize(ml_models['clip'].get_text_features(**inputs), dim=-1).detach().cpu().numpy()[0].tolist()

def parse_price_string(s: str) -> Optional[int]:
    if not s: return None
    if isinstance(s, (int, float)): return int(s)
    clean = s.lower().replace(" ", "").replace(",", ".")
    multiplier = 1
    if "m" in clean or "milj" in clean:
        multiplier = 1_000_000
        clean = re.sub(r"[^\d\.]", "", clean)
    elif "k" in clean:
        multiplier = 1_000
        clean = re.sub(r"[^\d\.]", "", clean)
    try:
        val = float(clean)
        return int(val * multiplier)
    except:
        return None


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Return the first valid JSON object embedded inside text."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def build_qdrant_plan(price_floor: Optional[int], price_ceiling: Optional[int], min_rooms: Optional[float], municipality: Optional[str]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    if price_floor is not None or price_ceiling is not None:
        plan.append({
            "field": "asking_price_sek",
            "type": "range",
            "gte": price_floor,
            "lte": price_ceiling
        })
    if min_rooms is not None:
        plan.append({
            "field": "number_of_rooms",
            "type": "range",
            "gte": float(min_rooms),
            "lte": None
        })
    if municipality:
        plan.append({
            "field": "municipality",
            "type": "match",
            "value": municipality
        })
    return plan

def llm_parse(query: str):
    if 'llm' not in ml_models: return {"text_query": query, "image_query": query, "filters": {}}
    
    # --- HOT RELOAD CONFIG ---
    config = {"system_prompt": "Output JSON.", "examples": []}
    if LLM_CONFIG_PATH.exists():
        try:
            with open(LLM_CONFIG_PATH, "r") as f:
                config = json.load(f)
        except Exception as e:
            log("config_read_error", error=str(e))

    sys_prompt = config.get("system_prompt", "")
    examples = config.get("examples", [])
    
    messages = [{"role": "system", "content": sys_prompt}]
    for ex in examples:
        messages.append({"role": "user", "content": ex['input']})
        messages.append({"role": "assistant", "content": json.dumps(ex['output'])})
    messages.append({"role": "user", "content": query})
    
    text = ml_models['llm_tok'].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ml_models['llm_tok']([text], return_tensors="pt").to(device)
    input_len = inputs['input_ids'].shape[1]
    
    with torch.inference_mode():
        out_tokens = ml_models['llm'].generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=False,
            pad_token_id=ml_models['llm_tok'].pad_token_id,
            eos_token_id=ml_models['llm_tok'].eos_token_id
        )
    
    generated_tokens = out_tokens[0][input_len:]
    raw = ml_models['llm_tok'].decode(generated_tokens, skip_special_tokens=True)
    log("llm_raw_output", raw=raw)

    data = {"text_query": query, "image_query": query, "filters": {}, "llm_plan": None, "qdrant_plan": []}
    
    try:
        raw_clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = extract_first_json_object(raw_clean)
        if not parsed:
            raise ValueError("no_json_object_found")

        intent = LLMIntent(**parsed)
        hard_filters = intent.hard_filters or HardFilters()
        text_intent = intent.semantic_search.strip() or query

        clean_filters: Dict[str, Any] = {}
        price_floor = parse_price_string(hard_filters.price_floor_str)
        price_ceiling = parse_price_string(hard_filters.price_ceiling_str)
        if price_floor is not None:
            clean_filters["min_price"] = price_floor
        if price_ceiling is not None:
            clean_filters["max_price"] = price_ceiling
        if hard_filters.min_rooms_num is not None:
            clean_filters["min_rooms"] = float(hard_filters.min_rooms_num)
        if hard_filters.municipality:
            clean_filters["municipality"] = hard_filters.municipality

        region = intent.postgis_region or PostGISRegion()
        if region.mode == "circle" and region.center_lat is not None and region.center_lon is not None:
            radius = region.radius_km if region.radius_km is not None else 5.0
            clean_filters["center_lat"] = float(region.center_lat)
            clean_filters["center_lon"] = float(region.center_lon)
            clean_filters["radius_km"] = float(radius)

        qdrant_plan = build_qdrant_plan(price_floor, price_ceiling, hard_filters.min_rooms_num, hard_filters.municipality)

        data = {
            "text_query": text_intent,
            "image_query": text_intent,
            "filters": clean_filters,
            "llm_plan": intent.model_dump(),
            "qdrant_plan": qdrant_plan
        }

    except (ValidationError, ValueError, TypeError) as e:
        log("llm_parse_fail", error=str(e), raw=raw)
    
    return data

# --- Endpoints ---

@app.post("/search/hybrid", tags=["Hemnet Listings"])
def search_hybrid(req: HybridSearchRequest):
    q_filter = build_qdrant_filter(req.filters)
    limit = req.topk * 3
    
    g_text = ml_models['qdrant'].query_points_groups(
        collection_name=LISTING_COLLECTION, 
        query=encode_text(req.text_query),
        using="text",
        query_filter=q_filter, 
        group_by="listing_id", 
        limit=limit, 
        group_size=1, 
        with_payload=True
    ).groups

    g_image = ml_models['qdrant'].query_points_groups(
        collection_name=LISTING_COLLECTION, 
        query=encode_image(req.image_query),
        using="image",
        query_filter=q_filter, 
        group_by="listing_id", 
        limit=limit, 
        group_size=1, 
        with_payload=True
    ).groups

    k, scores, payloads = 60, {}, {}
    for r, g in enumerate(g_text):
        scores[g.id] = scores.get(g.id, 0) + (1/(k + r + 1))
        payloads[g.id] = g.hits[0].payload
    for r, g in enumerate(g_image):
        scores[g.id] = scores.get(g.id, 0) + (1/(k + r + 1))
        payloads.setdefault(g.id, g.hits[0].payload)

    results = [{"id": i, "rrf_score": scores[i], "payload": payloads[i]} for i in sorted(scores, key=scores.get, reverse=True)[:req.topk]]
    return {"parsed": req.dict(), "results": results}

@app.post("/agent/query", tags=["Agent"])
def agent_query(req: AgentQueryRequest):
    try:
        params = llm_parse(req.prompt)
        f_data = params.get("filters", {})
        f_obj = ListingFilters(**{k: v for k, v in f_data.items() if v is not None})
        
        hybrid_req = HybridSearchRequest(
            text_query=params.get("text_query", req.prompt),
            image_query=params.get("image_query", req.prompt),
            topk=req.topk,
            filters=f_obj
        )
        return search_hybrid(hybrid_req)
    except Exception as e:
        log("agent_fail", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/text", tags=["Satellite Chips (Legacy)"])
def search_chip_text(q: str, topk: int = 10):
    if 'clip_proc' not in ml_models: raise HTTPException(503, "Models not loaded")
    
    vector = encode_image(q)
    try:
        resp = ml_models['qdrant'].query_points(
            collection_name=CHIP_COLLECTION,
            query=vector, 
            limit=topk, 
            with_payload=True
        )
        results = resp.points
    except Exception as e:
        return {"query": q, "results": [], "error": str(e)}

    output = []
    if legacy_meta is not None and not legacy_meta.empty:
        for hit in results:
            try:
                rid = int(hit.id)
                if 0 <= rid < len(legacy_meta):
                    row = legacy_meta.iloc[rid]
                    output.append({
                        "id": rid,
                        "score": float(hit.score),
                        "png": row["png_path"],
                        "lon": float(row["lon"]),
                        "lat": float(row["lat"]),
                    })
            except (ValueError, IndexError):
                continue
    return {"query": q, "results": output}

@app.post("/search/listing_image")
def legacy_stub(req: SearchRequest): return {}
