#!/usr/bin/env python3
# Batched embedding for CLIP-compatible models
import os, json, yaml, math
import numpy as np, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

def log(stage, goal, next_step):
    print(json.dumps({"stage": stage, "goal": goal, "next_step": next_step}, ensure_ascii=False))

cfg = yaml.safe_load(open("config.yaml"))
device = "cuda" if torch.cuda.is_available() and cfg["model"]["device"] in ("auto","cuda") else "cpu"
model_name = cfg["model"]["name"]; fallback = cfg["model"]["fallback"]
bs = int(cfg.get("runtime", {}).get("batch_size", 32))
chips_csv = "/project/data/chips/chips_index.csv"
out_dir = "/project/data/chips"
meta_out = f"{out_dir}/metadata.parquet"
emb_out  = f"{out_dir}/embeddings.npy"

def load_model(name):
    try:
        proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        mdl  = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
        return proc, mdl
    except Exception:
        proc = AutoProcessor.from_pretrained(fallback, trust_remote_code=True)
        mdl  = AutoModel.from_pretrained(fallback, trust_remote_code=True).to(device).eval()
        return proc, mdl

proc, mdl = load_model(model_name)
df = pd.read_csv(chips_csv)
paths = df["png_path"].tolist()
N = len(paths)
vecs = np.memmap(emb_out, dtype="float32", mode="w+", shape=(N, 768))  # 768 for CLIP ViT-L/14

with torch.inference_mode():
    for i in tqdm(range(0, N, bs), total=math.ceil(N/bs)):
        batch_paths = paths[i:i+bs]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = proc(images=imgs, return_tensors="pt", padding=True).to(device)
        feats = mdl.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1).detach().cpu().numpy().astype("float32")
        vecs[i:i+feats.shape[0], :] = feats

# ensure data flushed from memmap and write metadata
del vecs
df.to_parquet(meta_out, index=False)

log("embed", f"encoded {N} images in batches of {bs} to {emb_out}", "run index_qdrant.py next")










# #!/usr/bin/env python3
# import os, json, yaml, numpy as np, pandas as pd, torch
# from PIL import Image
# from tqdm import tqdm
# from transformers import AutoProcessor, AutoModel

# def log(stage, goal, next_step):
#     print(json.dumps({"stage": stage, "goal": goal, "next_step": next_step}, ensure_ascii=False))

# cfg = yaml.safe_load(open("config.yaml"))
# device = "cuda" if torch.cuda.is_available() and cfg["model"]["device"] in ("auto","cuda") else "cpu"
# model_name = cfg["model"]["name"]
# fallback = cfg["model"]["fallback"]

# def load_model(name):
#     try:
#         proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
#         mdl = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
#         return proc, mdl
#     except Exception:
#         proc = AutoProcessor.from_pretrained(fallback, trust_remote_code=True)
#         mdl = AutoModel.from_pretrained(fallback, trust_remote_code=True).to(device).eval()
#         return proc, mdl

# proc, mdl = load_model(model_name)
# df = pd.read_csv("/project/data/chips/chips_index.csv")
# vecs = []
# with torch.inference_mode():
#     for p in tqdm(df["png_path"].tolist()):
#         img = Image.open(p).convert("RGB")
#         inputs = proc(images=img, return_tensors="pt")
#         inputs = {k:v.to(device) for k,v in inputs.items()}
#         out = mdl.get_image_features(**inputs)
#         v = torch.nn.functional.normalize(out, dim=-1).squeeze().detach().cpu().numpy().astype("float32")
#         vecs.append(v)

# arr = np.stack(vecs)
# np.save("/project/data/chips/embeddings.npy", arr)
# df.to_parquet("/project/data/chips/metadata.parquet", index=False)

# log("embed", "encode chips to vectors", "create or update Qdrant collection and upsert vectors")
