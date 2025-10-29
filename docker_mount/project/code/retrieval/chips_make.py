#!/usr/bin/env python3
import os, csv, json, sqlite3, re, io
import requests  # HTTP (Hypertext Transfer Protocol) client
import mercantile  # web-tile math from lon/lat bbox
import yaml  # YAML (YAML Ain’t Markup Language) config

def log(stage, goal, next_step):
    print(json.dumps({"stage": stage, "goal": goal, "next_step": next_step}, ensure_ascii=False))

# ---- config ----
cfg = yaml.safe_load(open("config.yaml"))
z        = int(cfg["chips"]["z"])                      # zoom level
bbox     = list(map(float, cfg["chips"]["bbox"]))      # [min_lon,min_lat,max_lon,max_lat]
max_tiles= int(cfg["chips"]["max_tiles"])              # hard cap
out_dir  = cfg["chips"]["out_dir"]                     # output folder
tile_src = cfg["chips"].get("tile_source","auto")      # "auto" | "http" | "mbtiles"
r_base   = cfg["chips"].get("/raster_base","")         # optional HTTP base
mb_path  = cfg["chips"]["mbtiles"]                     # MBTiles (Mapbox Tiles SQLite) path or "auto"

# ensure output folder
os.makedirs(out_dir, exist_ok=True)
index_csv = os.path.join(out_dir, "chips_index.csv")

# ---- helpers ----
def discover_http_base():
    for base in [
        "http://127.0.0.1:8080/raster/ortho_2017_demo",
        "http://127.0.0.1:8080/raster/ortho",
        "http://127.0.0.1:8080/rasters/ortho",
    ]:
        try:
            r = requests.get(f"{base}/0/0/0.png", timeout=2)
            if r.status_code == 200:
                return base
        except Exception:
            pass
    try:
        txt = open("../../../../docker/nginx.conf","r",encoding="utf-8").read()
        if re.search(r"location\s+/raster/?", txt):
            return "http://127.0.0.1:8080/raster/ortho_2017_demo"
    except Exception:
        pass
    return ""

def list_mbtiles():
    hits=[]
    for dp,_,fs in os.walk("/project/data"):
        for f in fs:
            if f.endswith(".mbtiles"):
                hits.append(os.path.join(dp,f))
    return hits

def fetch_http_png(base, z, x, y):
    url = f"{base}/{z}/{x}/{y}.png"
    r = requests.get(url, timeout=5)
    if r.status_code == 200:
        return r.content
    return None

def fetch_mbtiles_png(cur, z, x, y_xyz):
    y_tms = (1<<z) - 1 - y_xyz
    # try (col,row) then (row,col) with TMS, then with XYZ
    for (qq, args) in [
        ("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?", (z,x,y_tms)),
        ("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_row=? AND tile_column=?", (z,y_tms,x)),
        ("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?", (z,x,y_xyz)),
        ("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_row=? AND tile_column=?", (z,y_xyz,x)),
    ]:
        row = cur.execute(qq, args).fetchone()
        if row:
            return row[0]
    return None

# ---- choose source ----
http_base = ""
con = cur = None
if tile_src in ("auto","http"):
    http_base = r_base or discover_http_base()
if not http_base and tile_src in ("auto","mbtiles"):
    if mb_path == "auto":
        found = list_mbtiles()
        if not found:
            raise SystemExit("No .mbtiles under /project/data")
        mb_path = found[0]
    con = sqlite3.connect(mb_path)
    cur = con.cursor()

# ---- iterate tiles from bbox ----
tiles = list(mercantile.tiles(bbox[0], bbox[1], bbox[2], bbox[3], [z]))

written = 0
skipped = 0
with open(index_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id","z","x","y","lon","lat","png_path"])
    for t in tiles:
        if written >= max_tiles:
            break
        x, y = t.x, t.y
        blob = None
        if http_base:
            blob = fetch_http_png(http_base, z, x, y)
        else:
            blob = fetch_mbtiles_png(cur, z, x, y)
        if not blob:
            skipped += 1
            continue
        fn = os.path.join(out_dir, f"{z}_{x}_{y}.png")
        with open(fn, "wb") as out:
            out.write(blob)   # PNG (Portable Network Graphics) bytes
        lon, lat = mercantile.ul(x, y, z)
        w.writerow([written, z, x, y, lon, lat, fn])
        written += 1

if con:
    con.close()

log("chips_make",
    f"wrote={written}, skipped_missing={skipped}, out_dir={out_dir}",
    "run embed.py next")




# #!/usr/bin/env python3
# import os, csv, json, math, sqlite3, re
# import requests
# import mercantile
# import rasterio
# from rasterio.io import MemoryFile
# from PIL import Image
# import yaml

# # Progress log helper
# def log(stage, goal, next_step):
#     print(json.dumps({"stage": stage, "goal": goal, "next_step": next_step}, ensure_ascii=False))

# cfg = yaml.safe_load(open("config.yaml"))
# out_dir = cfg["chips"]["out_dir"]
# z = int(cfg["chips"]["z"])
# bbox = cfg["chips"]["bbox"]
# max_tiles = int(cfg["chips"]["max_tiles"])
# tile_source = cfg["chips"].get("tile_source","auto")
# raster_base = cfg["chips"].get("/raster_base","")
# os.makedirs(out_dir, exist_ok=True)

# def discover_http_base():
#     # Try common NGINX routes you likely have
#     candidates = [
#         "http://127.0.0.1:8080/raster/ortho_2017_demo",
#         "http://127.0.0.1:8080/raster/ortho",
#         "http://127.0.0.1:8080/rasters/ortho",
#     ]
#     for base in candidates:
#         try:
#             r = requests.get(base.replace("{z}","0"), timeout=2)
#         except Exception:
#             pass
#     # Fallback to reading nginx.conf
#     try:
#         txt = open("../../../../docker/nginx.conf","r").read()
#         m = re.search(r"location\s+/raster/?", txt)
#         if m:
#             return "http://127.0.0.1:8080/raster/ortho_2017_demo"
#     except Exception:
#         pass
#     return ""

# def list_mbtiles():
#     root = "/project/data"
#     hits=[]
#     for dp,_,files in os.walk(root):
#         for f in files:
#             if f.endswith(".mbtiles"):
#                 hits.append(os.path.join(dp,f))
#     return hits

# def tile_png_http(base, z,x,y):
#     url = f"{base}/{z}/{x}/{y}.png"
#     r = requests.get(url, timeout=5)
#     r.raise_for_status()
#     return r.content

# def tile_png_mbtiles(dbpath, z,x,y):
#     # Standard MBTiles schema (z,x,y TMS y-flipped). Try common variants.
#     con = sqlite3.connect(dbpath)
#     cur = con.cursor()
#     # Try tms flip
#     tms_y = (1<<z) - 1 - y
#     for sql in [
#         "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
#         "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_row=? AND tile_column=?",
#     ]:
#         row = cur.execute(sql,(z,x,tms_y)).fetchone()
#         if row: 
#             con.close()
#             return row[0]
#     con.close()
#     raise RuntimeError("tile not found")

# # choose source
# base = ""
# mb = None
# if tile_source in ("auto","http"):
#     base = raster_base or discover_http_base()
# if (not base) and tile_source in ("auto","mbtiles"):
#     if cfg["chips"]["mbtiles"]=="auto":
#         mts = list_mbtiles()
#         if not mts:
#             raise SystemExit("No .mbtiles found under /project/data")
#         mb = mts[0]
#     else:
#         mb = cfg["chips"]["mbtiles"]

# # iterate tiles
# tiles = list(mercantile.tiles(bbox[0], bbox[1], bbox[2], bbox[3], [z]))
# tiles = list(tiles)[:max_tiles]

# index_csv = os.path.join(out_dir, "chips_index.csv")
# with open(index_csv, "w", newline="") as f:
#     w = csv.writer(f)
#     w.writerow(["id","z","x","y","lon","lat","png_path"])
#     for i,t in enumerate(tiles):
#         x,y = t.x, t.y
#         if base:
#             blob = tile_png_http(base, z,x,y)
#         else:
#             blob = tile_png_mbtiles(mb, z,x,y)
#         with MemoryFile(blob) as mem:
#             with mem.open() as ds:
#                 arr = ds.read()  # keep unchanged
#                 img = Image.fromarray(arr.transpose(1,2,0))
#         fn = os.path.join(out_dir, f"{z}_{x}_{y}.png")
#         img.save(fn)
#         lon, lat = mercantile.lnglat(mercantile.ul(x,y,z))
#         w.writerow([i,z,x,y,lon,lat,fn])

# log("chips_make", "produce PNG chips and an index CSV", "move to embed vectors")
