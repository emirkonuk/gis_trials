#!/usr/bin/env python3
import re, json, pandas as pd, mercantile, os, sys, tempfile, shutil

META = "/project/data/chips/metadata.parquet"
TMP  = "/project/data/chips/metadata.parquet.tmp"

rx = re.compile(r"/(\d+)_(\d+)_(\d+)\.png$")

def tile_center(z:int,x:int,y:int):
    b = mercantile.bounds(x, y, z)
    lon = (b.west + b.east)/2.0
    lat = (b.south + b.north)/2.0
    return float(lon), float(lat)

def main():
    df = pd.read_parquet(META)
    # ensure z,x,y exist; if not, derive from filename
    if not all(c in df.columns for c in ("z","x","y")):
        zs, xs, ys = [], [], []
        for p in df["png_path"]:
            m = rx.search(p)
            if not m: zs.append(None); xs.append(None); ys.append(None); continue
            z,x,y = map(int, m.groups()); zs.append(z); xs.append(x); ys.append(y)
        df["z"], df["x"], df["y"] = zs, xs, ys

    lons, lats = [], []
    for z,x,y in zip(df["z"], df["x"], df["y"]):
        if pd.isna(z) or pd.isna(x) or pd.isna(y):
            lons.append(None); lats.append(None); continue
        lon, lat = tile_center(int(z), int(x), int(y))
        lons.append(lon); lats.append(lat)
    df["lon"], df["lat"] = lons, lats

    # atomic replace
    with tempfile.NamedTemporaryFile(delete=False) as tmpf:
        tmp_path = tmpf.name
    df.to_parquet(tmp_path, index=False)
    shutil.move(tmp_path, TMP)
    os.replace(TMP, META)

    ok = int(df["lon"].notna().sum() == len(df))
    out = {
        "rows": int(len(df)),
        "lon_min": float(df["lon"].min()), "lon_max": float(df["lon"].max()),
        "lat_min": float(df["lat"].min()), "lat_max": float(df["lat"].max()),
        "nulls": int(df["lon"].isna().sum() + df["lat"].isna().sum()),
        "ok": bool(ok)
    }
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    sys.exit(main())

