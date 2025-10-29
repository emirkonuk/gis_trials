import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

img_path = "/app/tile_debug.png"
img = Image.open(img_path).convert("RGB")
print("tile size:", img.size)

proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
max_mem = {0: "8GiB", "cpu": "48GiB"}  # adjust GPU indices if needed
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True, max_memory=max_mem
).eval()

messages = [
  {"role":"system","content":[{"type":"text","text":"You see a map tile. Base answers on pixels. Be literal and concise."}]},
  {"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"Describe key man-made and natural features."}]}
]
tpl = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = proc(text=[tpl], images=[img], return_tensors="pt")
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False, temperature=0.0)
print(proc.batch_decode(out, skip_special_tokens=True)[0])
