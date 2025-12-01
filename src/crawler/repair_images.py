#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
repair_images.py

CONTEXT & SITUATION:
--------------------
Some listing pages on Hemnet contain a video or carousel element that blocks the 
crawler's "passive" lazy-load scroll. As a result, the crawler captures the DOM 
(which lists all 50+ images in its JSON data) but only captures the network 
traffic for the first few images (e.g., 4-5 images).

This script acts as a "Post-Hoc Repair" tool. It does NOT use a browser. 
Instead, it:
1.  Reads the existing `dom.html` from disk.
2.  Parses the `__NEXT_DATA__` JSON to find the authoritative list of all image URLs.
3.  Checks the local `assets/images/` folder to see which ones are missing.
4.  Downloads missing images directly via HTTP requests.

CRITICAL DOWNSTREAM NOTE (Qdrant/Postgres):
-------------------------------------------
Simply downloading these files to disk is NOT enough for them to appear in search.
The `embed_daemon.py` reads file paths from the Postgres database (`listings_images` table).

After running this script, you must:
1.  Re-run `db_upsert.py` for the affected listings (or all listings). 
    This updates Postgres to "see" the new files on disk.
2.  Once Postgres is updated, the `embed_daemon` (if running the new logic) 
    or a `backfill` script will pick them up and generate the new vectors 
    for Qdrant.
"""

import os
import sys
import json
import time
import random
import hashlib
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# --- Configuration ---
# Adjust this path if running from a different location
PROJECT_ROOT = Path("/project").resolve() 
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "listings_raw" / "hemnet" / "snapshots"

# Headers to behave politely
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Politeness
SLEEP_MIN = 0.5
SLEEP_MAX = 1.5

def _get_safe_ext(content_type: str) -> str:
    """Matches the extension logic in hemnet_crawl.py"""
    ct = (content_type or "").lower()
    if "jpeg" in ct or "jpg" in ct: return ".jpg"
    if "png" in ct: return ".png"
    if "webp" in ct: return ".webp"
    return ".bin"

def _calculate_hash(url: str) -> str:
    """Matches the hashing logic in hemnet_crawl.py"""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

def parse_images_from_dom(dom_path: Path):
    """
    Extracts high-res image URLs from the __NEXT_DATA__ blob in dom.html.
    """
    try:
        soup = BeautifulSoup(dom_path.read_text(encoding="utf-8"), 'html.parser')
        script_tag = soup.find('script', {'id': '__NEXT_DATA__', 'type': 'application/json'})
        
        if not script_tag:
            return []

        data = json.loads(script_tag.string)
        apollo_state = data.get("props", {}).get("pageProps", {}).get("__APOLLO_STATE__", {})
        
        image_urls = set()
        
        # Scan the Apollo State for Image objects
        for key, value in apollo_state.items():
            # Hemnet images typically live in objects with keys like 'Image:12345'
            # and have a 'url' field.
            if key.startswith("Image:") and isinstance(value, dict):
                url = value.get("url")
                if url:
                    image_urls.add(url)
                    
        return list(image_urls)
    
    except Exception as e:
        print(f"Error parsing DOM {dom_path}: {e}")
        return []

def repair_listing(snapshot_path: Path):
    """
    Checks one listing folder for missing images and downloads them.
    """
    dom_path = snapshot_path / "dom.html"
    assets_dir = snapshot_path / "assets" / "images"
    
    if not dom_path.exists():
        return

    # 1. Create assets dir if it somehow doesn't exist
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 2. Find expected images
    urls = parse_images_from_dom(dom_path)
    if not urls:
        return

    # 3. Check which are missing
    missing = []
    for url in urls:
        h = _calculate_hash(url)
        # Check for common extensions
        found = False
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            if (assets_dir / f"{h}{ext}").exists():
                found = True
                break
        
        if not found:
            missing.append(url)

    if not missing:
        return

    print(f"Listing {snapshot_path.name}: Found {len(missing)} missing images.")

    # 4. Download missing
    for url in missing:
        h = _calculate_hash(url)
        try:
            # Politeness sleep
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
            
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                ext = _get_safe_ext(resp.headers.get("Content-Type"))
                out_path = assets_dir / f"{h}{ext}"
                out_path.write_bytes(resp.content)
                print(f"  + Downloaded: {out_path.name}")
            else:
                print(f"  ! Failed {url}: Status {resp.status_code}")
        except Exception as e:
            print(f"  ! Error downloading {url}: {e}")

def main():
    print(f"Starting repair scan in {SNAPSHOTS_DIR}...")
    if not SNAPSHOTS_DIR.exists():
        print("Snapshot directory not found.")
        sys.exit(1)

    # Loop through all snapshot directories
    # Sorting ensures consistent order; random shuffle might be better for large runs
    dirs = sorted([d for d in SNAPSHOTS_DIR.iterdir() if d.is_dir()])
    
    for d in dirs:
        repair_listing(d)

    print("\n--- Scan Complete ---")
    print("REMINDER: Run db_upsert.py on affected listings to register these new files in Postgres.")

if __name__ == "__main__":
    main()