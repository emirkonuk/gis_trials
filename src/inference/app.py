import json
import os
import sqlite3
import uuid
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from transformers import AutoModelForVision2Seq, AutoProcessor

BASE_DIR = Path(__file__).resolve().parent
STACK_ROOT = Path(os.environ.get("GIS_STACK_ROOT", BASE_DIR.parents[1]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", STACK_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", STACK_ROOT / "results"))
DEFAULT_MBTILES = os.getenv(
    "DEFAULT_MBTILES", "/workspace/data/rasters/ortho_2017/ortho_2017.mbtiles"
)

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
GPU_INDEX = int(os.getenv("GPU_INDEX", os.getenv("INFER_GPU", "0")))
GPU_MEM = os.getenv("GPU_MEM", "8GiB")
CPU_MEM = os.getenv("CPU_MEM", "48GiB")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You see a map tile. Base answers on pixels. Be literal and concise.",
)

DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
_processor = None
_model = None
_loaded = False
_last_error = None


def _ensure_model() -> None:
    global _processor, _model, _loaded, _last_error
    if _loaded:
        return
    try:
        _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        max_mem = {GPU_INDEX: GPU_MEM, "cpu": CPU_MEM}
        _model = (
            AutoModelForVision2Seq.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
                device_map="auto",
                max_memory=max_mem,
                trust_remote_code=True,
            ).eval()
        )
        _loaded = True
        _last_error = None
    except Exception as exc:  # pragma: no cover - GPU load failure reporting
        _last_error = f"{type(exc).__name__}: {exc}"
        raise


def mbtiles_fetch_image(mbpath: str, z=None, x=None, y=None):
    con = sqlite3.connect(mbpath)
    cur = con.cursor()
    if None in (z, x, y):
        cur.execute("SELECT zoom_level,tile_column,tile_row FROM tiles LIMIT 1")
        z_db, x_db, tms_y = cur.fetchone()
        y_xyz = (1 << z_db) - 1 - tms_y
        cur.execute(
            "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
            (z_db, x_db, tms_y),
        )
        blob = cur.fetchone()[0]
        con.close()
        return Image.open(BytesIO(blob)).convert("RGB"), (z_db, x_db, y_xyz)

    z = int(z)
    x = int(x)
    y = int(y)
    tms_y = (1 << z) - 1 - y
    cur.execute(
        "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
        (z, x, tms_y),
    )
    row = cur.fetchone()
    con.close()
    if row is None:
        raise ValueError(f"tile not found z={z} x={x} y={y}")
    return Image.open(BytesIO(row[0])).convert("RGB"), (z, x, y)


def resize_max_side(img: Image.Image, max_side=512):
    width, height = img.size
    scale = max(width, height) / max_side
    if scale <= 1:
        return img
    return img.resize((int(width / scale), int(height / scale)))


def make_overlay_png(text: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 1014, 160], fill=(0, 0, 0, 90))
    draw.text((24, 24), text[:600], fill=(255, 255, 255, 255))
    path = out_dir / "overlay.png"
    img.save(path, "PNG")
    return path


@app.get("/infer/healthz")
def healthz():
    return {"ready": _loaded, "device": DEVICE, "model": MODEL_ID, "error": _last_error}


@app.get("/infer/api")
def ping():
    return {"status": "ok", "device": DEVICE, "model": MODEL_ID}


@app.post("/infer/mbtile")
async def infer_mbtile(request: Request):
    body = await request.body()
    payload = json.loads(body.decode("utf-8")) if body else {}
    mb_path = payload.get("mbtiles")

    def resolve_mbtiles(path_hint: str | None) -> str | None:
        candidates: list[str] = []
        if path_hint:
            candidates.append(path_hint)
            if path_hint.startswith("/project/"):
                candidates.append("/workspace" + path_hint[len("/project") :])
            elif path_hint.startswith("/workspace/"):
                candidates.append("/project" + path_hint[len("/workspace") :])
            if not path_hint.startswith("/"):
                candidates.append(str((DATA_ROOT / path_hint).resolve()))
        candidates.append(DEFAULT_MBTILES)
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            if os.path.isfile(candidate):
                return candidate
        return None

    mb_path_resolved = resolve_mbtiles(mb_path)
    if not mb_path_resolved:
        return JSONResponse(
            {"error": f"mbtiles not found: {mb_path or DEFAULT_MBTILES}"},
            status_code=400,
        )
    mb_path = mb_path_resolved

    z, x, y = payload.get("z"), payload.get("x"), payload.get("y")
    try:
        image, (zz, xx, yy) = mbtiles_fetch_image(mb_path, z, x, y)
    except Exception as exc:  # pragma: no cover - runtime error path
        return JSONResponse({"error": str(exc)}, status_code=400)

    image = resize_max_side(image, 512)
    debug_path = BASE_DIR / "tile_debug.png"
    try:
        image.save(debug_path, "PNG")
    except Exception:
        debug_path = None

    prompt = payload.get("text", "Describe key man-made and natural features.")
    _ensure_model()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        },
    ]
    template = _processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = _processor(text=[template], images=[image], return_tensors="pt")
    inputs = {
        key: (value.to(DEVICE) if hasattr(value, "to") else value) for key, value in inputs.items()
    }

    with torch.inference_mode():
        out_ids = _model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
        )
    text_out = _processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    def assistant_only(raw: str) -> str:
        parts = raw.split("\nassistant\n", 1)
        trimmed = parts[1] if len(parts) > 1 else raw
        for sep in (". ", "\n"):
            if sep in trimmed:
                trimmed = trimmed.split(sep, 1)[0]
                break
        return trimmed.strip()

    text_out = assistant_only(text_out)
    uid = uuid.uuid4().hex
    output_dir = RESULTS_DIR / uid
    make_overlay_png(text_out, output_dir)

    return JSONResponse(
        {
            "text": text_out,
            "image_url": f"/results/{uid}/overlay.png",
            "tile": {"mbtiles": mb_path, "z": zz, "x": xx, "y": yy},
            "debug_image": str(debug_path) if debug_path else None,
        }
    )
