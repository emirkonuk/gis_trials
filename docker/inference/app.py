# /app/app.py  — DEBUG OVERRIDE VERSION (no env used)
import os, json, uuid, sqlite3
from io import BytesIO

import torch
from PIL import Image, ImageDraw
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoModelForVision2Seq

# ==== HARD FIXED DEBUG CONFIG (NO ENV) ====
MODEL_ID   = "Qwen/Qwen2-VL-7B-Instruct"
GPU_INDEX  = 0
GPU_MEM    = "8GiB"
CPU_MEM    = "48GiB"
SYSTEM_PROMPT = "You see a map tile. Base answers on pixels. Be literal and concise."
RESULTS_DIR = "/project/results"

DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
_processor = None
_model = None
_loaded = False
_last_error = None

def _ensure_model():
    global _processor, _model, _loaded, _last_error
    if _loaded: return
    try:
        _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        max_mem = {GPU_INDEX: GPU_MEM, "cpu": CPU_MEM}
        _model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map="auto",
            max_memory=max_mem,
            trust_remote_code=True,
        ).eval()
        _loaded = True
        _last_error = None
    except Exception as e:
        _last_error = f"{type(e).__name__}: {e}"
        raise

def mbtiles_fetch_image(mbpath: str, z=None, x=None, y=None):
    con = sqlite3.connect(mbpath); cur = con.cursor()
    if None in (z, x, y):
        cur.execute("SELECT zoom_level,tile_column,tile_row FROM tiles LIMIT 1")
        z_db, x_db, tms_y = cur.fetchone()
        y_xyz = (1 << z_db) - 1 - tms_y
        cur.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                    (z_db, x_db, tms_y))
        blob = cur.fetchone()[0]; con.close()
        return Image.open(BytesIO(blob)).convert("RGB"), (z_db, x_db, y_xyz)
    z,x,y = int(z),int(x),int(y)
    tms_y = (1 << z) - 1 - y
    cur.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (z, x, tms_y))
    row = cur.fetchone(); con.close()
    if row is None: raise ValueError(f"tile not found z={z} x={x} y={y}")
    return Image.open(BytesIO(row[0])).convert("RGB"), (z, x, y)

def resize_max_side(img: Image.Image, max_side=512):
    w,h = img.size; s = max(w,h)/max_side
    return img.resize((int(w/s), int(h/s))) if s>1 else img

def make_overlay_png(text: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    img = Image.new("RGBA", (1024,1024), (0,0,0,0))
    d = ImageDraw.Draw(img)
    d.rectangle([10,10,1014,160], fill=(0,0,0,90))
    d.text((24,24), text[:600], fill=(255,255,255,255))
    p = os.path.join(out_dir, "overlay.png"); img.save(p, "PNG"); return p

@app.get("/infer/healthz")
def healthz():
    return {"ready": _loaded, "device": DEVICE, "model": MODEL_ID, "error": _last_error}

@app.get("/infer/api")
def ping():
    return {"status":"ok","device":DEVICE,"model":MODEL_ID}

@app.post("/infer/mbtile")
async def infer_mbtile(request: Request):
    body = await request.body()
    p = json.loads(body.decode("utf-8")) if body else {}
    mb = p.get("mbtiles")
    if not mb or not os.path.isfile(mb):
        return JSONResponse({"error": f"mbtiles not found: {mb}"}, status_code=400)

    z,x,y = p.get("z"), p.get("x"), p.get("y")
    try:
        image, (zz,xx,yy) = mbtiles_fetch_image(mb, z,x,y)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    image = resize_max_side(image, 512)               # same as run_tile.py
    dbg_path = "/app/tile_debug.png"
    try: image.save(dbg_path, "PNG")
    except Exception: dbg_path = None

    prompt = p.get("text", "Describe key man-made and natural features.")
    _ensure_model()

    messages = [
      {"role":"system","content":[{"type":"text","text":SYSTEM_PROMPT}]},
      {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}
    ]
    tpl = _processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = _processor(text=[tpl], images=[image], return_tensors="pt")
    inputs = {k:(v.to(DEVICE) if hasattr(v,"to") else v) for k,v in inputs.items()}  # avoid cpu/cuda mismatch

    with torch.inference_mode():
        out_ids = _model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
            num_beams=1
        )
    text_out = _processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    # keep only assistant part, first sentence
    def assistant_only(s):
        parts = s.split("\nassistant\n", 1)
        t = parts[1] if len(parts) > 1 else s
        # trim to first sentence or line
        for sep in [". ", "\n"]:
            if sep in t:
                t = t.split(sep,1)[0]
                break
        return t.strip()
    text_out = assistant_only(text_out)

    uid = str(uuid.uuid4()); out_dir = os.path.join(RESULTS_DIR, uid)
    make_overlay_png(text_out, out_dir)

    return JSONResponse({
        "text": text_out,
        "image_url": f"/results/{uid}/overlay.png",
        "tile": {"mbtiles": mb, "z": zz, "x": xx, "y": yy},
        "debug_image": dbg_path
    })
































# # /app/app.py  (mounted from map_serving/docker/inference/app.py)
# import os, json, uuid, re, base64, sqlite3
# from io import BytesIO

# import torch
# from PIL import Image, ImageDraw
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from transformers import AutoProcessor, AutoModelForVision2Seq

# # ------------------------- CONFIG --------------------------------------------
# # Match the working script defaults exactly. You can override via env.
# MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")

# # Choose which single GPU to use for VRAM budget; rest spills to CPU.
# GPU_INDEX = int(os.getenv("GPU_INDEX", "0"))         # e.g., 0 or 1
# GPU_MEM   = os.getenv("GPU_MEM", "8GiB")             # VRAM cap per your script
# CPU_MEM   = os.getenv("CPU_MEM", "48GiB")            # CPU offload cap

# RESULTS_DIR = "/project/results"
# SYSTEM_PROMPT = "You see a map tile. Base answers on pixels. Be literal and concise."
# DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ------------------------- APP STATE -----------------------------------------
# app = FastAPI()
# _processor = None
# _model = None
# _loaded = False
# _last_error = None

# def _ensure_model():
#     """Load processor+model once, with the SAME settings as your run_tile.py."""
#     global _processor, _model, _loaded, _last_error
#     if _loaded:
#         return
#     try:
#         _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
#         max_mem = {GPU_INDEX: GPU_MEM, "cpu": CPU_MEM}  # EXACT offload policy
#         _model = AutoModelForVision2Seq.from_pretrained(
#             MODEL_ID,
#             torch_dtype=DTYPE,
#             device_map="auto",
#             max_memory=max_mem,
#             trust_remote_code=True,
#         ).eval()
#         _loaded = True
#         _last_error = None
#     except Exception as e:
#         _last_error = f"{type(e).__name__}: {e}"
#         raise

# # ------------------------- UTILS ---------------------------------------------
# def make_overlay_png(text: str, out_dir: str):
#     os.makedirs(out_dir, exist_ok=True)
#     w, h = 1024, 1024
#     img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
#     draw = ImageDraw.Draw(img)
#     draw.rectangle([10, 10, w - 10, 160], fill=(0, 0, 0, 90))
#     draw.text((24, 24), text[:600], fill=(255, 255, 255, 255))
#     p = os.path.join(out_dir, "overlay.png")
#     img.save(p, "PNG")
#     return p

# def mbtiles_fetch_image(mbpath: str, z=None, x=None, y=None):
#     """
#     Return (PIL.Image, (z,x,y_XYZ)).
#     If z/x/y are None, use the first tile in the DB.
#     Converts incoming XYZ to TMS row for MBTiles table lookup.
#     """
#     con = sqlite3.connect(mbpath)
#     cur = con.cursor()
#     if None in (z, x, y):
#         cur.execute("SELECT zoom_level, tile_column, tile_row FROM tiles LIMIT 1")
#         z_db, x_db, tms_y = cur.fetchone()
#         y_xyz = (1 << z_db) - 1 - tms_y
#         cur.execute(
#             "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
#             (z_db, x_db, tms_y),
#         )
#         blob = cur.fetchone()[0]
#         con.close()
#         return Image.open(BytesIO(blob)).convert("RGB"), (z_db, x_db, y_xyz)
#     # supplied XYZ → TMS
#     z, x, y = int(z), int(x), int(y)
#     tms_y = (1 << z) - 1 - y
#     cur.execute(
#         "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
#         (z, x, tms_y),
#     )
#     row = cur.fetchone()
#     con.close()
#     if row is None:
#         raise ValueError(f"tile not found z={z} x={x} y={y}")
#     return Image.open(BytesIO(row[0])).convert("RGB"), (z, x, y)

# def resize_max_side(img: Image.Image, max_side=512) -> Image.Image:
#     w, h = img.size
#     s = max(w, h) / max_side
#     if s > 1:
#         return img.resize((int(w / s), int(h / s)))
#     return img

# # ------------------------- HEALTH/DEBUG --------------------------------------
# @app.get("/infer/healthz")
# def healthz():
#     return {"ready": _loaded, "device": DEVICE, "model": MODEL_ID, "error": _last_error}

# @app.post("/infer/debug")
# async def infer_debug(request: Request):
#     body = await request.body()
#     try:
#         p = json.loads(body.decode("utf-8")) if body else {}
#     except Exception:
#         p = {}
#     _ensure_model()
#     # make a trivial 256×256 to verify tensor keys
#     img = Image.new("RGB", (256, 256), (180, 200, 220))
#     msgs = [
#         {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
#         {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": p.get("text", "describe")}]},
#     ]
#     tpl = _processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
#     inputs = _processor(text=[tpl], images=[img], return_tensors="pt")
#     shapes = {k: tuple(v.shape) for k, v in inputs.items() if hasattr(v, "shape")}
#     return {"keys": list(inputs.keys()), "shapes": shapes}

# # ------------------------- MBTILES INFERENCE ---------------------------------
# @app.post("/infer/mbtile")
# async def infer_mbtile(request: Request):
#     """
#     Body JSON:
#       {
#         "mbtiles": "/project/data/rasters/ortho_2017_demo.mbtiles",
#         "text": "Describe key man-made and natural features.",
#         "z": 13, "x": 4498, "y": 2414   # optional; if omitted, first tile in DB
#       }
#     """
#     body = await request.body()
#     p = json.loads(body.decode("utf-8")) if body else {}

#     mb = p.get("mbtiles")
#     if not mb or not os.path.isfile(mb):
#         return JSONResponse({"error": f"mbtiles not found: {mb}"}, status_code=400)

#     z = p.get("z"); x = p.get("x"); y = p.get("y")
#     try:
#         image, (zz, xx, yy) = mbtiles_fetch_image(mb, z, x, y)
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=400)

#     # EXACTLY like your run_tile.py: resize to ≤512 BEFORE saving/feeding
#     image = resize_max_side(image, 512)

#     # Save the exact image we feed to the model (debug parity with run_tile.py)
#     dbg_path = "/app/tile_debug.png"
#     try:
#         image.save(dbg_path, "PNG")
#     except Exception:
#         dbg_path = None

#     prompt = p.get("text", "Describe key man-made and natural features.")

#     _ensure_model()  # uses MODEL_ID + max_memory that match your script

#     # System + User messages identical to run_tile.py
#     messages = [
#         {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
#         {"role": "user", "content": [{"type": "image", "image": image},
#                                      {"type": "text",  "text": prompt}]},
#     ]
#     text_tpl = _processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

#     # Build inputs and move to one device to avoid "cpu vs cuda" index_select errors.
#     inputs = _processor(text=[text_tpl], images=[image], return_tensors="pt")
#     inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}

#     with torch.inference_mode():
#         out_ids = _model.generate(
#             **inputs,
#             max_new_tokens=64,
#             do_sample=False,
#             temperature=0.0,
#             num_beams=1
#         )

#     text_out = _processor.batch_decode(out_ids, skip_special_tokens=True)[0]

#     uid = str(uuid.uuid4())
#     out_dir = os.path.join(RESULTS_DIR, uid)
#     make_overlay_png(text_out, out_dir)

#     return JSONResponse({
#         "text": text_out,
#         "image_url": f"/results/{uid}/overlay.png",
#         "tile": {"mbtiles": mb, "z": zz, "x": xx, "y": yy},
#         "debug_image": dbg_path
#     })

# # ------------------------- SIMPLE PING ---------------------------------------
# @app.get("/infer/api")
# def ping():
#     return {"status": "ok", "device": DEVICE, "model": MODEL_ID}








# import os, json, uuid, re, base64, torch
# from io import BytesIO
# from PIL import Image, ImageDraw
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from transformers import AutoProcessor, AutoModelForVision2Seq

# RESULTS_DIR = "/project/results"
# # MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
# # MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
# # MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
# # MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
# MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")


# DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
# DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# app = FastAPI()
# _processor = None
# _model = None
# _loaded = False
# _last_error = None

# def _ensure_model():
#     global _processor, _model, _loaded, _last_error
#     if _loaded:
#         return
#     try:
#         # from transformers import AutoProcessor, AutoModelForVision2Seq

#         _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
#         max_mem = {1: "8GiB", "cpu": "48GiB"}  # same as your run_tile.py
#         _model = AutoModelForVision2Seq.from_pretrained(
#             MODEL_ID,
#             torch_dtype=DTYPE,
#             device_map="auto",
#             max_memory=max_mem,
#             trust_remote_code=True
#         ).eval()


#         _loaded = True
#     except Exception as e:
#         _last_error = str(e)
#         raise

# def image_from_dataurl(s: str):
#     m = re.match(r'^data:image/(png|jpeg);base64,(.+)$', s)
#     if not m:
#         return None
#     return Image.open(BytesIO(base64.b64decode(m.group(2)))).convert("RGB")

# def blank_image(size=256):
#     return Image.new("RGB", (size, size), (255, 255, 255))

# def make_overlay_png(text, out_dir):
#     os.makedirs(out_dir, exist_ok=True)
#     w, h = 1024, 1024
#     img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
#     draw = ImageDraw.Draw(img)
#     draw.rectangle([10, 10, w-10, 160], fill=(0, 0, 0, 90))
#     draw.text((24, 24), text[:300], fill=(255,255,255,255))
#     p = os.path.join(out_dir, "overlay.png")
#     img.save(p, "PNG")
#     return p

# @app.get("/infer/healthz")
# def healthz():
#     return {"ready": _loaded, "device": DEVICE, "model": MODEL_ID, "error": _last_error}

# @app.get("/infer/api")
# def hello():
#     return {"status": "ok", "device": DEVICE, "model": MODEL_ID}

# @app.post("/infer/api")
# async def infer(request: Request):
#     body = await request.body()
#     try:
#         payload = json.loads(body.decode("utf-8")) if body else {}
#     except Exception:
#         payload = {}

#     prompt = payload.get("text", "Describe the image.")
#     _ = payload.get("bbox", None)

#     _ensure_model()

#     img = None
#     if isinstance(payload.get("image_data"), str):
#         try:
#             img = image_from_dataurl(payload["image_data"])
#         except Exception:
#             img = None
#     image = img if img is not None else blank_image(256)

#     SYSTEM_PROMPT = (
#     "You are a vision-language model. The user request includes an image. "
#     "Base your answer on the image content. Do not say you cannot see images."
#     )

#     messages = [
#         {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
#         {"role": "user", "content": [
#             {"type": "image", "image": image},
#             {"type": "text",  "text": prompt}
#         ]}
#     ]
#     text_tpl = _processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#     inputs = _processor(text=[text_tpl], images=[image], return_tensors="pt").to(DEVICE)
#     with torch.inference_mode():
#         output_ids = _model.generate(
#             **inputs,
#             max_new_tokens=48,
#             do_sample=False,      # greedy
#             temperature=0.0,
#             top_p=1.0,
#             num_beams=1
#         )

#     text_out = _processor.batch_decode(output_ids, skip_special_tokens=True)[0]



#     uid = str(uuid.uuid4())
#     out_dir = os.path.join(RESULTS_DIR, uid)
#     make_overlay_png(text_out, out_dir)

#     return JSONResponse({
#         "text": text_out,
#         "echo": {"text": prompt},
#         "image_url": f"/results/{uid}/overlay.png"
#     })

# # Debug endpoint
# @app.post("/infer/debug")
# async def infer_debug(request: Request):
#     body = await request.body()
#     try:
#         payload = json.loads(body.decode("utf-8")) if body else {}
#     except Exception:
#         payload = {}

#     img = None
#     if isinstance(payload.get("image_data"), str):
#         try:
#             img = image_from_dataurl(payload["image_data"])
#         except Exception:
#             img = None
#     image = img if img is not None else blank_image(256)

#     _ensure_model()
#     inputs = _processor(images=[image], text=[payload.get("text","describe")], return_tensors="pt")
#     shapes = {k: tuple(v.shape) for k,v in inputs.items() if hasattr(v, "shape")}
#     return {"has_image": img is not None, "keys": list(inputs.keys()), "shapes": shapes}


# @app.post("/infer/prove")
# async def infer_prove(request: Request):
#     # returns hashes and tensor stats to prove the image flowed into the model inputs
#     body = await request.body()
#     try:
#         payload = json.loads(body.decode("utf-8")) if body else {}
#     except Exception:
#         payload = {}

#     # decode image
#     img = None
#     if isinstance(payload.get("image_data"), str):
#         try:
#             img = image_from_dataurl(payload["image_data"])
#         except Exception:
#             img = None
#     image = img if img is not None else blank_image(256)

#     _ensure_model()

#     # build inputs using Qwen2-VL chat template so vision tokens align
#     messages = [{"role":"user","content":[
#         {"type":"image","image":image},
#         {"type":"text","text": payload.get("text","describe")}
#     ]}]
#     text_tpl = _processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#     inputs = _processor(text=[text_tpl], images=[image], return_tensors="pt")

#     # numeric proof the vision tensor exists and is non-zero
#     pv = inputs.get("pixel_values")
#     stats = {
#         "has_image": img is not None,
#         "keys": list(inputs.keys()),
#         "pixel_values_shape": tuple(pv.shape) if hasattr(pv, "shape") else None,
#         "pixel_values_sum": float(pv.float().sum()) if pv is not None else None,
#         "pixel_values_mean": float(pv.float().mean()) if pv is not None else None,
#         "image_grid_thw": tuple(inputs.get("image_grid_thw", [None])[0].tolist()) if "image_grid_thw" in inputs else None,
#     }
#     return stats

# @app.post("/infer/forward_probe")
# async def forward_probe(request: Request):
#     import torch
#     body = await request.body()
#     try:
#         payload = json.loads(body.decode("utf-8")) if body else {}
#     except Exception:
#         payload = {}

#     # decode image or fall back to a valid size
#     img = None
#     if isinstance(payload.get("image_data"), str):
#         try:
#             img = image_from_dataurl(payload["image_data"])
#         except Exception:
#             img = None
#     image = img if img is not None else blank_image(256)

#     prompt = payload.get("text", "describe briefly")

#     _ensure_model()

#     # Qwen2-VL needs chat template tokens
#     messages = [{"role":"user","content":[
#         {"type":"image","image":image},
#         {"type":"text","text":prompt}
#     ]}]
#     text_tpl = _processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#     inputs = _processor(text=[text_tpl], images=[image], return_tensors="pt")
#     inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}

#     try:
#         with torch.inference_mode():
#             out = _model(**inputs, output_hidden_states=True, return_dict=True)
#         stats = {
#             "keys_in": list(inputs.keys()),
#             "pixel_values_shape": tuple(inputs["pixel_values"].shape) if "pixel_values" in inputs else None,
#             "image_grid_thw": tuple(inputs["image_grid_thw"][0].tolist()) if "image_grid_thw" in inputs else None,
#             "logits_shape": tuple(out.logits.shape) if hasattr(out, "logits") else None,
#             "has_hidden_states": bool(getattr(out, "hidden_states", None) is not None),
#         }
#         return stats
#     except Exception as e:
#         return {"error": type(e).__name__, "message": str(e)}

# @app.post("/infer/conditioning_probe")
# async def conditioning_probe(request: Request):
#     import torch
#     body = await request.body()
#     payload = json.loads(body.decode("utf-8")) if body else {}
#     prompt = payload.get("text", "describe briefly")

#     # decode image if provided, else 256x256 blank
#     img = None
#     if isinstance(payload.get("image_data"), str):
#         try:
#             img = image_from_dataurl(payload["image_data"])
#         except Exception:
#             img = None
#     image = img if img is not None else blank_image(256)

#     _ensure_model()

#     # build inputs WITH image
#     msgs = [{"role":"user","content":[{"type":"image","image":image},
#                                       {"type":"text","text":prompt}]}]
#     tpl_with = _processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
#     inp_with = _processor(text=[tpl_with], images=[image], return_tensors="pt")
#     inp_with = {k:(v.to(DEVICE) if hasattr(v,"to") else v) for k,v in inp_with.items()}

#     # build inputs WITHOUT image (same text path)
#     msgs_no = [{"role":"user","content":[{"type":"text","text":prompt}]}]
#     tpl_no = _processor.apply_chat_template(msgs_no, add_generation_prompt=True, tokenize=False)
#     inp_no = _processor(text=[tpl_no], return_tensors="pt")
#     inp_no = {k:(v.to(DEVICE) if hasattr(v,"to") else v) for k,v in inp_no.items()}

#     with torch.inference_mode():
#         out_with = _model(**inp_with, return_dict=True)
#         out_no   = _model(**inp_no,   return_dict=True)

#     # compare first-step logits
#     lw = out_with.logits[0, -1].float()
#     ln = out_no.logits[0, -1].float()
#     diff = (lw - ln).abs()
#     topk = torch.topk(diff, k=5)
#     return {
#         "delta_l1": float(diff.mean().item()),
#         "top5_idx": topk.indices.tolist(),
#         "top5_delta": topk.values.tolist()
#     }


# @app.post("/infer/color_probe")
# async def color_probe(request: Request):
#     import torch, math
#     body = await request.body()
#     payload = json.loads(body.decode("utf-8")) if body else {}
#     img = None
#     if isinstance(payload.get("image_data"), str):
#         try:
#             img = image_from_dataurl(payload["image_data"])
#         except Exception:
#             img = None
#     image = img if img is not None else blank_image(256)

#     _ensure_model()
#     colors = ["red","green","blue","orange","yellow","purple","pink","brown","black","white","gray"]
#     tok = _processor.tokenizer
#     # single-token ids using leading space for correct tokenization
#     ids = {}
#     for c in colors:
#         enc = tok.encode(" " + c, add_special_tokens=False)
#         if len(enc)==1: ids[c] = enc[0]
#     # build inputs with vision tokens
#     messages = [{"role":"user","content":[
#         {"type":"image","image":image},
#         {"type":"text","text":"The dominant color is"}
#     ]}]
#     tpl = _processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#     inputs = _processor(text=[tpl], images=[image], return_tensors="pt")
#     inputs = {k:(v.to(DEVICE) if hasattr(v,"to") else v) for k,v in inputs.items()}
#     with torch.inference_mode():
#         out = _model(**inputs, return_dict=True)
#     logits = out.logits[0, -1]  # next-token logits
#     probs = torch.softmax(logits, dim=-1)
#     scores = {c: float(probs[tid].item()) for c, tid in ids.items()}
#     top = max(scores, key=scores.get) if scores else None
#     return {"token_ids": ids, "scores": scores, "top": top}



# @app.post("/infer/score_texts")
# async def score_texts(request: Request):
#     import torch
#     body = await request.body()
#     p = json.loads(body.decode("utf-8")) if body else {}
#     cands = p.get("candidates", [])
#     prompt = p.get("prompt", "The dominant color is")

#     # image
#     img = None
#     if isinstance(p.get("image_data"), str):
#         try: img = image_from_dataurl(p["image_data"])
#         except Exception: img = None
#     image = img if img is not None else blank_image(256)

#     _ensure_model()

#     # prefix with vision tokens
#     msgs_prefix = [{"role":"user","content":[
#         {"type":"image","image":image},
#         {"type":"text","text":prompt}
#     ]}]
#     prefix = _processor.apply_chat_template(msgs_prefix, add_generation_prompt=True, tokenize=False)
#     # tokenize prefix alone to get its token length
#     prefix_ids = _processor.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
#     prefix_len = prefix_ids.size(0)

#     scores = []
#     with torch.inference_mode():
#         for s in cands:
#             # full text = prefix + candidate
#             full = prefix + " " + s
#             inp = _processor(text=[full], images=[image], return_tensors="pt")
#             inp = {k:(v.to(DEVICE) if hasattr(v,"to") else v) for k,v in inp.items()}

#             # labels same shape as input_ids, mask prefix
#             labels = inp["input_ids"].clone()
#             labels[:, :prefix_len] = -100  # ignore prefix tokens
#             inp["labels"] = labels

#             out = _model(**inp, return_dict=True)
#             scores.append({"text": s, "neg_avg_nll": -float(out.loss.item())})

#     best = max(scores, key=lambda x: x["neg_avg_nll"]) if scores else None
#     return {"prompt": prompt, "scores": scores, "best": best}




# # --- MBTiles → PIL.Image → VLM ----------------------------------------------
# import sqlite3
# from io import BytesIO

# def mbtile_image(mbpath: str, z: int=None, x: int=None, y: int=None):
#     """
#     If z/x/y are None, pick the first tile found.
#     If z/x/y provided as XYZ, convert y to TMS row.
#     Returns (PIL.Image, (z,x,y_XYZ))
#     """
#     con = sqlite3.connect(mbpath)
#     cur = con.cursor()
#     if None in (z, x, y):
#         cur.execute("SELECT zoom_level, tile_column, tile_row FROM tiles LIMIT 1")
#         z_db, x_db, tms_y = cur.fetchone()
#         # report XYZ y back to caller
#         y_xyz = (1 << z_db) - 1 - tms_y
#         cur.execute(
#             "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
#             (z_db, x_db, tms_y),
#         )
#         blob = cur.fetchone()[0]
#         con.close()
#         return Image.open(BytesIO(blob)).convert("RGB"), (z_db, x_db, y_xyz)
#     # XYZ → TMS
#     tms_y = (1 << z) - 1 - y
#     cur.execute(
#         "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
#         (z, x, tms_y),
#     )
#     row = cur.fetchone()
#     con.close()
#     if row is None:
#         raise ValueError(f"tile not found z={z} x={x} y={y}")
#     return Image.open(BytesIO(row[0])).convert("RGB"), (z, x, y)


# @app.post("/infer/mbtile")
# async def infer_mbtile(request: Request):
#     import sqlite3
#     from io import BytesIO
#     body = await request.body()
#     p = json.loads(body.decode("utf-8")) if body else {}

#     mb = p.get("mbtiles")
#     if not mb or not os.path.isfile(mb):
#         return JSONResponse({"error": f"mbtiles not found: {mb}"}, status_code=400)

#     z = p.get("z"); x = p.get("x"); y = p.get("y")

#     # fetch tile (XYZ in request; MBTiles stores TMS y)
#     con = sqlite3.connect(mb); cur = con.cursor()
#     if None in (z, x, y):
#         cur.execute("SELECT zoom_level, tile_column, tile_row FROM tiles LIMIT 1")
#         z_db, x_db, tms_y = cur.fetchone()
#         y_xyz = (1 << z_db) - 1 - tms_y
#         cur.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
#                     (z_db, x_db, tms_y))
#         blob = cur.fetchone()[0]
#         zz, xx, yy = z_db, x_db, y_xyz
#     else:
#         tms_y = (1 << int(z)) - 1 - int(y)
#         cur.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
#                     (int(z), int(x), tms_y))
#         row = cur.fetchone()
#         if row is None:
#             con.close()
#             return JSONResponse({"error": f"tile not found z={z} x={x} y={y}"}, status_code=404)
#         blob = row[0]
#         zz, xx, yy = int(z), int(x), int(y)
#     con.close()

#     # PIL image
#     image = Image.open(BytesIO(blob)).convert("RGB")

#     # resize to ≤512 on max side
#     w, h = image.size
#     max_side = 512
#     s = max(w, h) / max_side
#     if s > 1:
#         image = image.resize((int(w / s), int(h / s)))

#     # now save the exact image we feed to the model
#     dbg_path = "/app/tile_debug.png"
#     try:
#         image.save(dbg_path, "PNG")
#     except Exception:
#         dbg_path = None

#     prompt = p.get("text", "Describe key man-made and natural features.")

#     _ensure_model()  # uses your existing MODEL_ID and max_memory settings

#     # system + user messages (same as run_tile.py)
#     messages = [
#       {"role":"system","content":[{"type":"text","text":"You see a map tile. Base answers on pixels. Be literal and concise."}]},
#       {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}
#     ]

#     text_tpl = _processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#     inputs = _processor(text=[text_tpl], images=[image], return_tensors="pt").to(DEVICE)

#     # text_tpl = _processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#     # inputs = _processor(text=[text_tpl], images=[image], return_tensors="pt").to(DEVICE)
#     # inputs = _processor(text=[text_tpl], images=[image], return_tensors="pt")


#     # with torch.inference_mode():
#     #     out_ids = _model.generate(
#     #         **inputs,
#     #         max_new_tokens=64,
#     #         do_sample=False,
#     #         temperature=0.0,
#     #         top_p=1.0,
#     #         num_beams=1
#     #     )
#     with torch.inference_mode():
#         out_ids = _model.generate(
#             **inputs,
#             max_new_tokens=64,
#             do_sample=False,
#             temperature=0.0,
#             num_beams=1
#         )


#     text_out = _processor.batch_decode(out_ids, skip_special_tokens=True)[0]

#     uid = str(uuid.uuid4())
#     out_dir = os.path.join(RESULTS_DIR, uid)
#     make_overlay_png(text_out, out_dir)

#     return JSONResponse({
#         "text": text_out,
#         "image_url": f"/results/{uid}/overlay.png",
#         "tile": {"mbtiles": mb, "z": zz, "x": xx, "y": yy},
#         "debug_image": dbg_path
#     })
