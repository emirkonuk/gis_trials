#!/usr/bin/env python3
import gc
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import psycopg
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from osm_parser import heuristic_osm_filters
from osm_service import OSMQuerySpec, OSMAmenitySpec, OSMLocationSpec, OSMService
from pydantic import BaseModel, Field, ValidationError
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor
import transformers as _transformers
from transformers.generation.logits_process import LogitsProcessor as _CompatLogitsProcessor

if not hasattr(_transformers, "LogitsWarper"):
    class _CompatLogitsWarper(_CompatLogitsProcessor):
        def __call__(self, input_ids, scores):
            return scores
    _transformers.LogitsWarper = _CompatLogitsWarper
from jsonformer import Jsonformer

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
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_GPU_IDS = os.environ.get("LLM_GPU_IDS")
LLM_GPU_MAX_MEM = os.environ.get("LLM_GPU_MAX_MEM", "8GiB")
LLM_CPU_MAX_MEM = os.environ.get("LLM_CPU_MAX_MEM", "48GiB")
JSONFORMER_MAX_ARRAY = int(os.environ.get("LLM_JSON_MAX_ARRAY", "6"))
LLM_PREFER_4BIT = os.environ.get("LLM_PREFER_4BIT", "0") not in {"0", "false", "False"}
RETRIEVAL_EMBED_DEVICE = os.environ.get("RETRIEVAL_EMBED_DEVICE")

# Paths
STACK_ROOT = Path(os.environ.get("GIS_STACK_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", STACK_ROOT / "data"))
META_PATH = Path(os.environ.get("METADATA_PATH", DATA_ROOT / "chips" / "metadata.parquet"))
LLM_CONFIG_PATH = Path(__file__).parent / "llm_config.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_device = RETRIEVAL_EMBED_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")


def _detect_gpu_ids() -> List[int]:
    raw = LLM_GPU_IDS
    if not raw:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            raw = visible
    if not raw:
        return [i for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    ids: List[int] = []
    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        if token == "-1":
            continue
        try:
            ids.append(int(token))
        except ValueError:
            continue
    total = torch.cuda.device_count()
    if total:
        ids = [gid for gid in ids if 0 <= gid < total]
    if not ids and total:
        ids = [i for i in range(total)]
    return ids


GPU_ID_LIST = _detect_gpu_ids()

# --- Globals ---
ml_models = {}
legacy_meta = None
llm_config = {}
osm_client = OSMService()
llm_loading = False

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
    ml_models['text'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=embed_device)
    ml_models['clip'] = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(embed_device).eval()
    ml_models['clip_proc'] = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    ml_models['clip_device'] = embed_device
    
    ml_models['qdrant'] = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True, timeout=300)

    # 2. Legacy Metadata
    if META_PATH.exists():
        print(f"--- STARTUP: Loading legacy metadata from {META_PATH} ---")
        legacy_meta = pd.read_parquet(META_PATH)[["png_path", "lon", "lat"]].reset_index().rename(columns={"index": "row"})
    else:
        legacy_meta = pd.DataFrame(columns=["row", "png_path", "lon", "lat"])

    # 3. LLM (optional)
    print(f"--- STARTUP: Loading LLM ({LLM_MODEL_ID}) ...")
    global llm_loading
    llm_loading = True
    try:
        ml_models['llm_tok'] = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
        if ml_models['llm_tok'].pad_token is None:
            ml_models['llm_tok'].pad_token = ml_models['llm_tok'].eos_token

        if torch.cuda.is_available():
            gpu_loaded = False
            quant_order = [True, False] if LLM_PREFER_4BIT else [False, True]
            for quantize in quant_order:
                try:
                    gpu_model = _load_llm_model(
                        "auto",
                        torch.float16,
                        max_memory=_gpu_max_memory_map(),
                        quantize_4bit=quantize
                    )
                    _reset_llm_reference(gpu_model, "cuda")
                    label = "GPU 4-bit" if quantize else "GPU"
                    print(f"--- STARTUP: LLM Ready ({label}) ---")
                    gpu_loaded = True
                    break
                except Exception as err:
                    mode_label = "4-bit " if quantize else ""
                    if _is_cuda_oom(err):
                        print(f"--- STARTUP: {mode_label}GPU load OOM ({err}) ---")
                    else:
                        print(f"--- STARTUP: {mode_label}GPU load failed ({err}) ---")
            if not gpu_loaded:
                print("--- STARTUP: GPU load unavailable; falling back to CPU ---")
        if 'llm' not in ml_models:
            cpu_model = _load_llm_model("cpu", torch.float32, max_memory=_cpu_max_memory_map())
            _reset_llm_reference(cpu_model, "cpu")
            print("--- STARTUP: LLM Ready (CPU) ---")
    except Exception as e:
        print(f"!!! LLM LOAD FAILED: {e}")
        raise
    finally:
        llm_loading = False
    
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
    geometry_geojson: Optional[Dict[str, Any]] = None

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
    center_lat: Optional[str] = None
    center_lon: Optional[str] = None
    radius_km: Optional[str] = None


class HardFilters(BaseModel):
    price_floor_str: Optional[str] = None
    price_ceiling_str: Optional[str] = None
    min_rooms_num: Optional[str] = None
    municipality: Optional[str] = None


class QdrantFilterClause(BaseModel):
    field: Literal["asking_price_sek", "number_of_rooms", "municipality"]
    type: Literal["range", "match"]
    gte: Optional[str] = None
    lte: Optional[str] = None
    value: Optional[str] = None


class LLMIntent(BaseModel):
    semantic_search: str
    postgis_region: PostGISRegion = Field(default_factory=PostGISRegion)
    hard_filters: HardFilters = Field(default_factory=HardFilters)
    qdrant_filters: List[QdrantFilterClause] = Field(default_factory=list)
    osm_filters: Optional[OSMQuerySpec] = None


def _is_cuda_oom(err: Exception) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    msg = str(err).lower()
    return "cuda" in msg and ("out of memory" in msg or "memory" in msg)


OSM_RELATION_VALUES = ["near", "within", "inside", "around", "intersects", "adjacent", "unknown"]


def _llm_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "semantic_search": {"type": "string"},
            "postgis_region": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["circle", "none"]},
                    "center_lat": {"type": "string"},
                    "center_lon": {"type": "string"},
                    "radius_km": {"type": "string"}
                },
                "required": ["mode", "center_lat", "center_lon", "radius_km"],
                "additionalProperties": False
            },
            "hard_filters": {
                "type": "object",
                "properties": {
                    "price_floor_str": {"type": "string"},
                    "price_ceiling_str": {"type": "string"},
                    "min_rooms_num": {"type": "string"},
                    "municipality": {"type": "string"}
                },
                "required": ["price_floor_str", "price_ceiling_str", "min_rooms_num", "municipality"],
                "additionalProperties": False
            },
            "qdrant_filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "enum": ["asking_price_sek", "number_of_rooms", "municipality"]},
                        "type": {"type": "string", "enum": ["range", "match"]},
                        "gte": {"type": "string"},
                        "lte": {"type": "string"},
                        "value": {"type": "string"}
                    },
                    "required": ["field", "type"],
                    "additionalProperties": False
                }
            },
            "osm_filters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["neighborhood", "district", "municipality", "address", "bbox", "unknown"]},
                            "value": {"type": "string"}
                        },
                        "required": ["type", "value"],
                        "additionalProperties": False
                    },
                    "address": {"type": "string"},
                    "amenities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "relation": {"type": "string", "enum": OSM_RELATION_VALUES}
                            },
                            "required": ["type", "relation"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["location", "address", "amenities"],
                "additionalProperties": False
            }
        },
        "required": ["semantic_search", "postgis_region", "hard_filters", "qdrant_filters", "osm_filters"],
        "additionalProperties": False
    }


def _normalize_nullable_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"null", "none", "n/a"}:
            return None
        return stripped
    return str(value)


def _float_string_or_none(value: Any, field: str) -> Optional[str]:
    clean = _normalize_nullable_string(value)
    if clean is None:
        return None
    try:
        return str(float(clean))
    except (TypeError, ValueError):
        log("sanitize_warning", field=field, value=clean)
        return None


def _price_string_or_none(value: Any, field: str) -> Optional[str]:
    clean = _normalize_nullable_string(value)
    if clean is None:
        return None
    parsed = parse_price_string(clean)
    if parsed is None:
        log("sanitize_warning", field=field, value=clean)
        return None
    return str(parsed)


def _sanitize_llm_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    postgis = parsed.get("postgis_region") or {}
    postgis["center_lat"] = _float_string_or_none(postgis.get("center_lat"), "center_lat")
    postgis["center_lon"] = _float_string_or_none(postgis.get("center_lon"), "center_lon")
    postgis["radius_km"] = _float_string_or_none(postgis.get("radius_km"), "radius_km")
    if not postgis.get("mode"):
        postgis["mode"] = "none"
    parsed["postgis_region"] = postgis

    hard_filters = parsed.get("hard_filters") or {}
    orig_floor_raw = hard_filters.get("price_floor_str")
    orig_ceiling_raw = hard_filters.get("price_ceiling_str")
    hard_filters["price_floor_str"] = _price_string_or_none(hard_filters.get("price_floor_str"), "price_floor_str")
    hard_filters["price_ceiling_str"] = _price_string_or_none(hard_filters.get("price_ceiling_str"), "price_ceiling_str")
    hard_filters["min_rooms_num"] = _price_string_or_none(hard_filters.get("min_rooms_num"), "min_rooms_num")
    hard_filters["municipality"] = _normalize_nullable_string(hard_filters.get("municipality"))

    def _has_keyword(raw, keywords):
        if not isinstance(raw, str):
            return False
        lower = raw.lower()
        return any(word in lower for word in keywords)

    if _has_keyword(orig_floor_raw, ["below", "under", "less"]):
        hard_filters["price_ceiling_str"] = hard_filters["price_floor_str"]
        hard_filters["price_floor_str"] = None
    if _has_keyword(orig_ceiling_raw, ["above", "over", "more"]):
        hard_filters["price_floor_str"] = hard_filters["price_ceiling_str"]
        hard_filters["price_ceiling_str"] = None
    parsed["hard_filters"] = hard_filters

    q_filters = []
    for clause in parsed.get("qdrant_filters", []) or []:
        if not isinstance(clause, dict):
            continue
        clean = {
            "field": clause.get("field"),
            "type": clause.get("type"),
            "gte": _price_string_or_none(clause.get("gte"), "qdrant_gte"),
            "lte": _price_string_or_none(clause.get("lte"), "qdrant_lte"),
            "value": _normalize_nullable_string(clause.get("value"))
        }
        if clean["field"] and clean["type"]:
            q_filters.append(clean)
    parsed["qdrant_filters"] = q_filters

    osm = parsed.get("osm_filters") or {}
    location = osm.get("location") or {}
    loc_type = location.get("type") or "unknown"
    if loc_type not in ["neighborhood", "district", "municipality", "address", "bbox", "unknown"]:
        loc_type = "unknown"
    loc_value = _normalize_nullable_string(location.get("value"))
    if loc_value:
        location_clean = {"type": loc_type, "value": loc_value}
    else:
        location_clean = None

    address_val = _normalize_nullable_string(osm.get("address"))
    amenities_raw = osm.get("amenities") or []
    amenities: List[Dict[str, str]] = []
    for item in amenities_raw:
        if not isinstance(item, dict):
            continue
        a_type = _normalize_nullable_string(item.get("type"))
        relation = item.get("relation")
        if relation not in OSM_RELATION_VALUES:
            relation = "near"
        if a_type:
            amenities.append({"type": a_type, "relation": relation})

    if location_clean is None and loc_type == "unknown" and not address_val and not amenities:
        parsed["osm_filters"] = None
    else:
        parsed["osm_filters"] = {
            "location": location_clean,
            "address": address_val,
            "amenities": amenities
        }
    return parsed


def _truncate_geometry_placeholder(filters: Optional[Dict[str, Any]]) -> None:
    if not isinstance(filters, dict):
        return
    geom = filters.get("geometry_geojson")
    if not isinstance(geom, dict):
        return
    coords = geom.get("coordinates")
    count = 0
    try:
        if isinstance(coords, list) and coords:
            first = coords[0]
            if isinstance(first, list):
                count = len(first)
    except Exception:
        count = 0
    filters["geometry_geojson"] = f"<Polygon with {count} coordinates>"


def _serialize_qdrant_filter(filter_obj: Optional[models.Filter]) -> Optional[Dict[str, Any]]:
    if not filter_obj:
        return None
    try:
        as_dict = filter_obj.dict()
    except Exception:
        return {"raw": str(filter_obj)}

    def _strip_geo(payload: Any):
        if isinstance(payload, dict):
            clean = {}
            for key, val in payload.items():
                if key == "geo_polygon" and isinstance(val, dict):
                    exterior = val.get("exterior")
                    points = exterior.get("points") if isinstance(exterior, dict) else None
                    count = len(points) if isinstance(points, list) else 0
                    clean[key] = f"<GeoPolygon with {count} points>"
                else:
                    clean[key] = _strip_geo(val)
            return clean
        if isinstance(payload, list):
            return [_strip_geo(item) for item in payload]
        return payload

    return _strip_geo(as_dict)


def _gpu_max_memory_map() -> Dict[Any, Any]:
    mem: Dict[Any, Any] = {}
    for gid in GPU_ID_LIST:
        mem[gid] = LLM_GPU_MAX_MEM
    mem["cpu"] = LLM_CPU_MAX_MEM
    return mem


def _cpu_max_memory_map() -> Dict[str, str]:
    return {"cpu": LLM_CPU_MAX_MEM}


def _load_llm_model(device_map: str, dtype, max_memory: Optional[Dict[Any, Any]] = None, quantize_4bit: bool = False):
    kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": (device_map == "cpu")
    }
    if max_memory:
        kwargs["max_memory"] = max_memory
    if quantize_4bit:
        kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": dtype,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        })
    else:
        kwargs["torch_dtype"] = dtype
    return AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        **kwargs
    )


def _reset_llm_reference(model, device_label: str):
    ml_models['llm'] = model
    ml_models['llm_device'] = device_label
    _configure_llm_runtime(model)


def _configure_llm_runtime(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    try:
        cfg.use_cache = False
    except Exception:
        pass


def _move_llm_to_cpu():
    global llm_loading
    if ml_models.get('llm_device') == "cpu":
        return
    log("llm_reload_cpu", message="Reloading model on CPU after CUDA OOM")
    try:
        if 'llm' in ml_models:
            del ml_models['llm']
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    llm_loading = True
    try:
        cpu_model = _load_llm_model("cpu", torch.float32, max_memory=_cpu_max_memory_map())
        _reset_llm_reference(cpu_model, "cpu")
    finally:
        llm_loading = False
    log("llm_reload_cpu_done")


def _wait_for_llm(timeout: float = 1200.0):
    start = time.time()
    while llm_loading:
        if time.time() - start > timeout:
            raise RuntimeError("llm_load_timeout")
        time.sleep(0.25)


def _run_jsonformer(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    def _execute() -> Dict[str, Any]:
        generator = Jsonformer(
            ml_models['llm'],
            ml_models['llm_tok'],
            schema,
            prompt=prompt,
            temperature=0.01,
            max_array_length=JSONFORMER_MAX_ARRAY,
        )
        return generator()

    try:
        return _execute()
    except Exception as err:
        if _is_cuda_oom(err):
            _move_llm_to_cpu()
            return _execute()
        raise

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

def _geojson_to_qdrant_polygon(poly: Dict[str, Any]) -> Optional[models.GeoPolygon]:
    if not poly:
        return None
    coords: Optional[List[List[float]]] = None
    if poly.get("type") == "Polygon":
        coords = poly.get("coordinates", [None])[0]
    elif poly.get("type") == "MultiPolygon":
        polys = poly.get("coordinates")
        if polys and polys[0]:
            coords = polys[0][0]
    if not coords or len(coords) < 4:
        return None
    points = [models.GeoPoint(lon=float(p[0]), lat=float(p[1])) for p in coords]
    if points[0].lon != points[-1].lon or points[0].lat != points[-1].lat:
        points.append(points[0])
    return models.GeoPolygon(exterior=models.GeoLineString(points=points), interiors=[])


def build_qdrant_filter(f: ListingFilters):
    if not f:
        return None
    conds = []
    
    # TODO: Re-enable this once Qdrant is backfilled with municipality data. Currently, this field
    # is empty in the DB, so filtering by it yields 0 results.
    # if f.municipality:
    #     conds.append(models.FieldCondition(key="municipality", match=models.MatchText(text=f.municipality)))
        
    if f.min_price is not None or f.max_price is not None:
        range_kwargs: Dict[str, Any] = {}
        if f.min_price is not None:
            range_kwargs["gte"] = f.min_price
        if f.max_price is not None:
            range_kwargs["lte"] = f.max_price
        if range_kwargs:
            conds.append(models.FieldCondition(key="asking_price_sek", range=models.Range(**range_kwargs)))
    
    if f.min_rooms is not None:
        conds.append(models.FieldCondition(key="number_of_rooms", range=models.Range(gte=f.min_rooms)))
    
    polygon = None
    if f.geometry_geojson:
        polygon = _geojson_to_qdrant_polygon(f.geometry_geojson)
    elif f.center_lat is not None and f.center_lon is not None and f.radius_km is not None:
        poly = get_geo_polygon(f.center_lat, f.center_lon, f.radius_km)
        polygon = _geojson_to_qdrant_polygon(poly)
        if polygon:
            log("postgis_filter_active", lat=f.center_lat, lon=f.center_lon, radius=f.radius_km)

    if polygon:
        conds.append(models.FieldCondition(key="location", geo_polygon=polygon))
    
    return models.Filter(must=conds) if conds else None

def encode_text(q):
    with torch.inference_mode(): return F.normalize(ml_models['text'].encode(q, convert_to_tensor=True), p=2, dim=0).cpu().tolist()

def encode_image(q):
    with torch.inference_mode():
        clip_device = ml_models.get('clip_device', device)
        inputs = ml_models['clip_proc'](text=[q], return_tensors="pt", padding=True).to(clip_device)
        return F.normalize(ml_models['clip'].get_text_features(**inputs), dim=-1).detach().cpu().numpy()[0].tolist()

def parse_price_string(s: str) -> Optional[int]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return int(s)

    text = str(s).lower()
    prefix_tokens = [
        "under", "below", "less than", "less", "more than", "above", "over",
        "at most", "at least", "greater than", "not more than", "up to", "around"
    ]
    for token in prefix_tokens:
        text = text.replace(token, " ")
    text = text.replace("+", " ")
    text = text.replace("≈", " ")
    text = text.replace("~", " ")
    text = text.replace(",", ".")
    suffix_tokens = ["sek", "kr", "kronor", "msek", "sek.", "swedishkronor", "price", "prices", "millioner", "million"]
    for token in suffix_tokens:
        text = text.replace(token, " ")
    text = re.sub(r"(rooms?|rum|r)\b", " ", text)
    text = re.sub(r"\s+", " ", text)

    match = re.search(r"(-?\d+(?:[\s,_]?\d{3})*(?:\.\d+)?)(?:\s*(m|milj|miljon|miljoner|k))?", text)
    if not match:
        digits_only = re.sub(r"[^\d\.]", "", text)
        if not digits_only:
            return None
        number_text = digits_only
        suffix = None
    else:
        number_text = match.group(1)
        suffix = match.group(2)

    number_text = number_text.replace(" ", "").replace("_", "").replace(",", "")
    try:
        value = float(number_text)
    except ValueError:
        return None

    multiplier = 1
    if suffix:
        if suffix in {"m", "milj", "miljon", "miljoner"}:
            multiplier = 1_000_000
        elif suffix == "k":
            multiplier = 1_000
    return int(value * multiplier)


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

def _intent_to_payload(intent: LLMIntent, fallback_query: str) -> Dict[str, Any]:
    hard_filters = intent.hard_filters or HardFilters()
    text_intent = intent.semantic_search.strip() or fallback_query

    clean_filters: Dict[str, Any] = {}
    price_floor = parse_price_string(hard_filters.price_floor_str)
    price_ceiling = parse_price_string(hard_filters.price_ceiling_str)
    if price_floor is not None:
        clean_filters["min_price"] = price_floor
    if price_ceiling is not None:
        clean_filters["max_price"] = price_ceiling
    rooms_value = None
    if hard_filters.min_rooms_num is not None:
        try:
            rooms_value = float(hard_filters.min_rooms_num)
            clean_filters["min_rooms"] = rooms_value
        except (TypeError, ValueError):
            rooms_value = None
    if hard_filters.municipality:
        clean_filters["municipality"] = hard_filters.municipality

    region = intent.postgis_region or PostGISRegion()
    if region.mode == "circle" and region.center_lat is not None and region.center_lon is not None:
        radius = region.radius_km if region.radius_km is not None else 5.0
        try:
            clean_filters["center_lat"] = float(region.center_lat)
            clean_filters["center_lon"] = float(region.center_lon)
            clean_filters["radius_km"] = float(radius)
        except (TypeError, ValueError):
            pass

    osm_context = None
    if intent.osm_filters:
        try:
            osm_context = osm_client.resolve_query(intent.osm_filters)
            if osm_context.get("filter_geometry"):
                clean_filters["geometry_geojson"] = osm_context["filter_geometry"]
        except Exception as osm_err:
            log("osm_resolution_error", error=str(osm_err))

    qdrant_plan = build_qdrant_plan(price_floor, price_ceiling, rooms_value, hard_filters.municipality)
    return {
        "text_query": text_intent,
        "image_query": text_intent,
        "filters": clean_filters,
        "llm_plan": intent.model_dump(),
        "qdrant_plan": qdrant_plan,
        "osm_context": osm_context
    }


def llm_parse(query: str):
    osm_hint = heuristic_osm_filters(query)
    heuristic_context = None
    if osm_hint:
        try:
            heuristic_context = json.dumps(osm_hint.model_dump())
        except Exception:
            heuristic_context = None
    _wait_for_llm()
    if 'llm' not in ml_models:
        raise RuntimeError("llm_not_loaded")
    
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
    if heuristic_context:
        messages.append({
            "role": "system",
            "content": f"Heuristic OSM cues extracted automatically (review, correct, or expand as needed before emitting the final JSON): {heuristic_context}"
        })
    for ex in examples:
        messages.append({"role": "user", "content": ex['input']})
        messages.append({"role": "assistant", "content": json.dumps(ex['output'])})
    messages.append({"role": "user", "content": query})
    
    text = ml_models['llm_tok'].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    schema = _llm_json_schema()
    parsed = None
    raw = ""
    try:
        with torch.inference_mode():
            parsed = _run_jsonformer(text, schema)
        parsed = _sanitize_llm_output(parsed)
        raw = json.dumps(parsed)
        log("llm_structured_output", raw=raw)

        intent = LLMIntent(**parsed)
        return _intent_to_payload(intent, query)

    except (ValidationError, ValueError, TypeError) as e:
        log("llm_parse_fail", error=str(e), raw=raw)
        raise
    finally:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- Endpoints ---

@app.post("/search/hybrid", tags=["Hemnet Listings"])
def search_hybrid(req: HybridSearchRequest):
    q_filter = build_qdrant_filter(req.filters)
    log("qdrant_filter", filter=_serialize_qdrant_filter(q_filter))
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
    parsed_payload = req.dict()
    _truncate_geometry_placeholder(parsed_payload.get("filters"))
    return {"parsed": parsed_payload, "results": results}

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
