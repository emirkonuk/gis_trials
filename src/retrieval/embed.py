#!/usr/bin/env python3
# Batched embedding for CLIP-compatible models
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def log(stage, goal, next_step):
    print(json.dumps({"stage": stage, "goal": goal, "next_step": next_step}, ensure_ascii=False))


CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
STACK_ROOT = Path(os.environ.get("GIS_STACK_ROOT", CONFIG_PATH.parents[1]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", STACK_ROOT / "data"))
cfg = yaml.safe_load(CONFIG_PATH.read_text())

device = "cuda" if torch.cuda.is_available() and cfg["model"]["device"] in ("auto", "cuda") else "cpu"
model_name = cfg["model"]["name"]
fallback = cfg["model"]["fallback"]
runtime_cfg = cfg.get("runtime", {})
bs = int(runtime_cfg.get("batch_size", 32))

chips_csv = Path(runtime_cfg.get("chips_csv", DATA_ROOT / "chips" / "chips_index.csv"))
if not chips_csv.is_absolute():
    chips_csv = (DATA_ROOT / chips_csv).resolve()
out_dir = Path(runtime_cfg.get("chips_dir", DATA_ROOT / "chips"))
if not out_dir.is_absolute():
    out_dir = (DATA_ROOT / out_dir).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

meta_out = out_dir / "metadata.parquet"
emb_out = out_dir / "embeddings.npy"


def load_model(name):
    try:
        proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
        return proc, mdl
    except Exception:
        proc = AutoProcessor.from_pretrained(fallback, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(fallback, trust_remote_code=True).to(device).eval()
        return proc, mdl


proc, mdl = load_model(model_name)
df = pd.read_csv(chips_csv)
paths = df["png_path"].tolist()
total = len(paths)
vecs = np.memmap(emb_out, dtype="float32", mode="w+", shape=(total, 768))  # 768 for CLIP ViT-L/14

with torch.inference_mode():
    for start in tqdm(range(0, total, bs), total=math.ceil(total / bs)):
        batch_paths = paths[start : start + bs]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = proc(images=images, return_tensors="pt", padding=True).to(device)
        feats = mdl.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1).detach().cpu().numpy().astype("float32")
        vecs[start : start + feats.shape[0], :] = feats

del vecs  # flush memmap buffers
df.to_parquet(meta_out, index=False)

log(
    "embed",
    f"encoded {total} images in batches of {bs} to {emb_out}",
    "run index_qdrant.py next",
)
