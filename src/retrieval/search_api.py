#!/usr/bin/env python3
import os
import re
import json
import time
import torch
import torch.nn.functional as F
import psycopg
import pandas as pd
import ast
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# --- Config ---
PG_HOST = os.environ.get("PGHOST", "db")
PG_DB = os.environ.get("PGDATABASE", "gis")
PG_USER = os.environ.get("PGUSER", "gis")
PG_PASS = os.environ.get("PGPASSWORD", "gis")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# Collections
LISTING_COLLECTION = "hemnet_listings_v1"
CHIP_COLLECTION = os.environ.get("QDRANT_COLLECTION", "sweden_demo_v0")

# Models
PHI_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# Paths
STACK_ROOT = Path(os.environ.get("GIS_STACK_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", STACK_ROOT / "data"))
META_PATH = Path(os.environ.get("METADATA_PATH", DATA_ROOT / "chips" / "metadata.parquet"))

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Globals ---
ml_models = {}
legacy_meta = None

def log(event, **kw):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    msg = " ".join(f"{k}={repr(v)}" for k,v in kw.items())
    print(f"{ts} {event} {msg}".strip(), flush=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global legacy_meta
    print(f"--- STARTUP: Loading models on {device} ---")
    
    # 1. Retrieval Models
    ml_models['text'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    ml_models['clip'] = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    ml_models['clip_proc'] = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # timeout=60s to prevent 'timed out' errors on heavy queries
    ml_models['qdrant'] = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=False, timeout=60)

    # 2. Legacy Metadata
    if META_PATH.exists():
        print(f"--- STARTUP: Loading legacy metadata from {META_PATH} ---")
        legacy_meta = pd.read_parquet(META_PATH)[["png_path", "lon", "lat"]].reset_index().rename(columns={"index": "row"})
    else:
        print("--- STARTUP: No legacy metadata found (Legacy search will be empty) ---")
        legacy_meta = pd.DataFrame(columns=["row", "png_path", "lon", "lat"])

    # 3. LLM (Phi-3)
    print("--- STARTUP: Loading LLM (Phi-3 4-bit)...")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        ml_models['llm_tok'] = AutoTokenizer.from_pretrained(PHI_MODEL_ID, trust_remote_code=True)
        ml_models['llm'] = AutoModelForCausalLM.from_pretrained(
            PHI_MODEL_ID, 
            quantization_config=bnb_config, 
            trust_remote_code=False, 
            attn_implementation="eager",
            device_map="auto"
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

# --- Helper Functions ---
def get_geo_polygon(lat, lon, radius_km):
    try:
        conn_str = f"host={PG_HOST} dbname={PG_DB} user={PG_USER} password={PG_PASS}"
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                # Create a buffer (circle) and return as GeoJSON
                cur.execute("SELECT ST_AsGeoJSON(ST_Simplify(ST_Buffer(ST_MakePoint(%s, %s)::geography, %s)::geometry, 0.0001));", (lon, lat, radius_km * 1000.0))
                row = cur.fetchone()
                return json.loads(row[0]) if row else None
    except Exception as e:
        log("postgis_error", error=str(e))
        return None

def build_qdrant_filter(f: ListingFilters):
    if not f: return None
    conds = []
    # Fuzzy match for municipality
    if f.municipality: conds.append(models.FieldCondition(key="municipality", match=models.MatchText(text=f.municipality)))
    if f.min_price or f.max_price: conds.append(models.FieldCondition(key="asking_price_sek", range=models.Range(gte=f.min_price, lte=f.max_price)))
    if f.min_rooms: conds.append(models.FieldCondition(key="number_of_rooms", range=models.Range(gte=f.min_rooms)))
    
    # PostGIS Geo-Filter
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
    """Robustly parse prices like '4M', '4.5M', '4 000 000'."""
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

def llm_parse(query: str):
    """Parses user intent, coordinates, and prices using Python for math."""
    if 'llm' not in ml_models: return {"text_query": query, "image_query": query, "filters": {}}
    
    # UPDATED PROMPT: Explicitly forbids guessing coordinates for names.
    sys_prompt = (
        'You are a parser. Rules:\n'
        '1. Coordinates: Extract "lat", "lon" ONLY if explicit numbers appear (e.g. "59.3, 18.0").\n'
        '   - DO NOT convert city names (e.g. "centrum", "Täby") to coordinates. Leave lat/lon null.\n'
        '2. Prices: Extract RAW strings (e.g. "4M"). Under/Below -> max_price_raw. Over/Above -> min_price_raw.\n'
        '3. Ignore location names in filters (put in text_query).\n'
        'Schema: {"text_query": str, "filters": {"min_price_raw": str, "max_price_raw": str, "lat": float, "lon": float, "radius": float}}\n'
        '\nExamples:\n'
        'Input: "Houses near 59.3, 18.0 under 5M"\n'
        'Output: {"text_query": "Houses", "filters": {"max_price_raw": "5M", "lat": 59.3, "lon": 18.0}}\n'
        'Input: "Apartment in centrum under 2M"\n'
        'Output: {"text_query": "Apartment in centrum", "filters": {"max_price_raw": "2M", "lat": null, "lon": null}}\n'
    )
    prompt = f"<|system|>\n{sys_prompt}<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
    
    inputs = ml_models['llm_tok'](prompt, return_tensors="pt").to(device)
    input_len = inputs['input_ids'].shape[1]
    
    with torch.inference_mode():
        out_tokens = ml_models['llm'].generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=ml_models['llm_tok'].eos_token_id)
    
    generated_tokens = out_tokens[0][input_len:]
    raw = ml_models['llm_tok'].decode(generated_tokens, skip_special_tokens=True)
    log("llm_raw_output", raw=raw)

    data = {"text_query": query, "image_query": query, "filters": {}}
    
    try:
        raw_clean = raw.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', raw_clean.replace("\n", " "), re.DOTALL)
        
        if match:
            candidate = match.group(0)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(candidate)
            
            # --- PYTHON POST-PROCESSING ---
            f = parsed.get("filters", {})
            clean_filters = {}

            # 1. Handle Coordinates
            if f.get("lat") and f.get("lon"):
                clean_filters["center_lat"] = float(f["lat"])
                clean_filters["center_lon"] = float(f["lon"])
                clean_filters["radius_km"] = float(f.get("radius", 5))
            
            # 2. Handle Prices (Python Math)
            if "min_price_raw" in f:
                clean_filters["min_price"] = parse_price_string(str(f["min_price_raw"]))
            if "max_price_raw" in f:
                clean_filters["max_price"] = parse_price_string(str(f["max_price_raw"]))
                
            # 3. Handle Text Query fallback
            text_q = parsed.get("text_query", query)
            
            # 4. Construct Final Object
            data = {
                "text_query": text_q,
                "image_query": text_q,
                "filters": clean_filters
            }
            
    except Exception as e:
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
        
        # Pydantic safe parsing
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