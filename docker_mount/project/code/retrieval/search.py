#!/usr/bin/env python3
import argparse, json, yaml, numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient

def encode_text(text, name, fallback, device):
    proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
    with torch.inference_mode():
        out = mdl.get_text_features(**proc(text=[text], return_tensors="pt").to(device))
        v = torch.nn.functional.normalize(out, dim=-1).squeeze().cpu().numpy().astype("float32")
    return v

def encode_img(p, name, fallback, device):
    proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
    with torch.inference_mode():
        img = Image.open(p).convert("RGB")
        out = mdl.get_image_features(**proc(images=img, return_tensors="pt").to(device))
        v = torch.nn.functional.normalize(out, dim=-1).squeeze().cpu().numpy().astype("float32")
    return v

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str)
    ap.add_argument("--image", type=str)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    device = "cuda" if torch.cuda.is_available() and cfg["model"]["device"] in ("auto","cuda") else "cpu"
    name = cfg["model"]["name"]; fallback = cfg["model"]["fallback"]
    v = encode_text(args.text, name, fallback, device) if args.text else encode_img(args.image, name, fallback, device)

    c = QdrantClient(host=cfg["index"]["host"], port=cfg["index"]["port"])
    res = c.search(collection_name=cfg["index"]["collection"], query_vector=v.tolist(), limit=args.topk)
    for r in res:
        print(json.dumps({"score": float(r.score), **r.payload}, ensure_ascii=False))
