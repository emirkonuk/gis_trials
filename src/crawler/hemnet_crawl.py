#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hemnet crawler: single-threaded, polite, headless Chromium via Playwright.
- Starts from a seed search URL (e.g., https://www.hemnet.se/bostader?location_ids[]=17744)
- Enqueues listing URLs found on search pages
- Visits listings one by one with throttling and jitter
- Captures:
    raw.html  = server HTML (document response)
    dom.html  = post-render DOM after JS + lazy scroll
    assets/   = images + JSON/XHR + selected same-origin assets (css/js/font)
    meta.json = manifest with counts, basic fields, stats, and errors if any

Politeness:
- Single process, single page
- Sleeps 30–60 s by default between fetches
- Per-run caps

Run examples:
  python3 src/crawler/hemnet_crawl.py --seed "https://www.hemnet.se/bostader?location_ids[]=17744" --max-listings 2 --per-run-cap 2
"""

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.parse import urlparse, urljoin

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# -------- Paths and constants --------

PROJECT_ROOT = Path("/project").resolve()
DATA_ROOT = PROJECT_ROOT / "data" / "listings_raw" / "hemnet"
SNAP_DIR = DATA_ROOT / "snapshots"
STATE_DB = DATA_ROOT / "state.sqlite"

SNAP_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)

TIMEOUT_MS = 30000  # per navigation/wait

# -------- Utilities --------

def log(event: str, **kwargs):
    msg = " ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    print(f"{ts} {event} {msg}".strip())
    sys.stdout.flush()

def ensure_state(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS q (
            url TEXT PRIMARY KEY,
            seen INTEGER NOT NULL DEFAULT 0
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS fetched (
            url   TEXT PRIMARY KEY,
            ts    INTEGER NOT NULL,
            path  TEXT NOT NULL,
            kind  TEXT NOT NULL
        )
    """)
    con.commit()

def add_urls_to_queue(con: sqlite3.Connection, urls: List[str]) -> int:
    n = 0
    for u in urls:
        try:
            con.execute("INSERT OR IGNORE INTO q(url,seen) VALUES(?,0)", (u,))
            n += 1
        except Exception:
            pass
    con.commit()
    return n

def pop_next_listing(con: sqlite3.Connection) -> Optional[str]:
    # pick a listing URL never seen
    cur = con.execute("SELECT url FROM q WHERE seen=0 LIMIT 1")
    row = cur.fetchone()
    if not row:
        return None
    url = row[0]
    con.execute("UPDATE q SET seen=1 WHERE url=?", (url,))
    con.commit()
    return url

def seen_or_fetched(con: sqlite3.Connection, url: str) -> bool:
    cur = con.execute("SELECT 1 FROM fetched WHERE url=? LIMIT 1", (url,))
    return cur.fetchone() is not None

def listing_id(url: str) -> str:
    m = re.search(r"/bostad/[^/]+-(\d+)(?:$|[/?#])", url)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d+)(?:$|[/?#])", url)
    return m2.group(1) if m2 else re.sub(r"\W+", "_", url)

def _safe_ext(content_type: str, default: str = ".bin") -> str:
    ct = (content_type or "").lower()
    if "jpeg" in ct or "jpg" in ct: return ".jpg"
    if "png" in ct: return ".png"
    if "webp" in ct: return ".webp"
    if "gif" in ct: return ".gif"
    if "json" in ct: return ".json"
    if "html" in ct: return ".html"
    if "javascript" in ct or "ecmascript" in ct: return ".js"
    if "css" in ct: return ".css"
    if "font" in ct: return ".woff2"
    return default

def _lazy_scroll(page, max_steps=12):
    # best-effort to trigger lazy content. No hard failures.
    try:
        h = page.evaluate("() => document.body.scrollHeight") or 0
        step = max(400, int(h / max_steps)) if h else 800
        y = 0
        for _ in range(max_steps):
            y += step
            page.evaluate(f"() => window.scrollTo(0,{y})")
            page.wait_for_timeout(300)
        page.evaluate("() => window.scrollTo(0, 0)")
        page.wait_for_timeout(200)
    except Exception:
        pass

# -------- Capture one listing --------

def save_listing_bundle(page, con, url: str, sleep_min: int, sleep_jitter: int):
    """
    Raw bundle capture without extra GETs:
      snapshots/<id>/raw.html        (server HTML response body)
      snapshots/<id>/dom.html        (post-render DOM)
      snapshots/<id>/assets/images/* (image responses as seen)
      snapshots/<id>/assets/json/*   (XHR JSON bodies, if any)
      snapshots/<id>/assets/other/*  (selected same-origin assets)
      snapshots/<id>/meta.json       (manifest)
    """
    lid  = listing_id(url)
    root = SNAP_DIR / lid
    img_dir   = root / "assets" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    # json_dir  = root / "assets" / "json"
    # other_dir = root / "assets" / "other"
    # for d in (img_dir, json_dir, other_dir):
        # d.mkdir(parents=True, exist_ok=True)

    captured = []
    raw_written = False
    err = None
    same_dom = False

    def on_response(resp):
        nonlocal raw_written
        try:
            u   = resp.url
            rp  = urlparse(u)
            host = rp.netloc
            if not (host.endswith("hemnet.se") or (host.startswith("bilder.") and host.endswith("hemnet.se"))):
                return

            rt = resp.request.resource_type            # 'document','image','xhr', ...
            ct = (resp.headers.get("content-type","") or "").lower()
            body = resp.body() or b""
            size = len(body)
            h = hashlib.sha1(u.encode("utf-8")).hexdigest()[:16]
            ext = _safe_ext(ct)

            out_path = None
            if rt == "document" and "text/html" in ct:
                (root / "raw.html").write_bytes(body)
                raw_written = True
            elif "image" in ct:
                out_path = img_dir / f"{h}{ext}"
                out_path.write_bytes(body)
            # else: ignore JSON/XHR and CSS/JS/font entirely

            captured.append({
                "url": u, "status": resp.status, "ctype": ct, "rtype": rt,
                "size": size, "path": (str(out_path) if out_path else None)
            })
        except Exception:
            pass


    page.on("response", on_response)

    try:
        # Navigate and render
        page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT_MS)
        _lazy_scroll(page, max_steps=12)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT_MS)

        # post-render DOM always in try
        (root / "dom.html").write_text(page.content(), encoding="utf-8")

        # if server HTML wasn’t captured, fall back to DOM for raw.html
        if not raw_written:
            (root / "raw.html").write_text(
                page.evaluate("() => document.documentElement.outerHTML"),
                encoding="utf-8"
            )
            same_dom = True

    except Exception as e:
        err = repr(e)

    finally:
        # Ensure dom.html exists even on failure
        try:
            if not (root / "dom.html").exists():
                (root / "dom.html").write_text(
                    page.evaluate("() => document.documentElement.outerHTML"),
                    encoding="utf-8"
                )
        except Exception:
            pass

        # Safe prefix test
        def _is_under(p, prefix):
            return isinstance(p, str) and p.startswith(prefix)

        img_prefix  = str(img_dir)
        # json_prefix = str(json_dir)
        images_cnt = sum(1 for c in captured if _is_under(c.get("path"), img_prefix))
        # json_cnt   = sum(1 for c in captured if _is_under(c.get("path"), json_prefix))
        json_cnt   = 0  # we do not capture JSON/XHR bodies now

        # Minimal text for extraction; tolerate render failures
        try:
            txt = page.inner_text("body")
        except Exception:
            txt = ""

        # ---------- Basic fields from rendered text ----------
        def _pick_first(patterns, text):
            for pat in patterns:
                m = re.search(pat, text, flags=re.IGNORECASE)
                if m:
                    return m.group(1).strip()
            return None

        price  = _pick_first([r"([\d\s\u00A0\.]+)\s*kr\b"], txt)                 # "4 200 000 kr"
        rooms  = _pick_first([r"(\d+(?:[.,]\d+)?)\s*rum\b"], txt)                # "1,5 rum"
        area   = _pick_first([r"(\d+(?:[.,]\d+)?)\s*m²\b"], txt)                 # "39 m²"
        fee    = _pick_first([r"([\d\s\u00A0\.]+)\s*kr\s*/\s*mån\b"], txt)       # "3 751 kr/mån"
        year   = _pick_first([r"Byggår\s*(\d{4})", r"Byggnadsår\s*(\d{4})"], txt)
        tenure = _pick_first([r"Upplåtelseform\s*([A-Za-zÅÄÖåäö\-]+)"], txt)
        address = None
        try:
            address = page.eval_on_selector("h1, h2", "el => el?.innerText?.trim()")
        except Exception:
            pass
        if not address:
            address = _pick_first([r"^([^\n,]+,\s*[^\n]+kommun)", r"^([^\n]+),\s*[^\n]+kommun"], txt)

        fields = {
            "price": price,
            "rooms": rooms,
            "area_m2": area,
            "fee_per_month": fee,
            "year_built": year,
            "tenure": tenure,
            "address": address,
        }

        # ---------- Stats: 12m only by default ----------
        # We avoid clicking toggles now; parse what is visible (typically 12m).
        def _pct(label, text):
            m = re.search(label + r".{0,60}?([+\-]?\d{1,3}(?:[.,]\d{1,2})?)\s*%", text, flags=re.IGNORECASE|re.DOTALL)
            return m.group(1).replace(",", ".") if m else None

        def _money_per_sqm(text):
            m = re.search(r"Kvadratmeterpris\s*\(snitt\)\s*([\d\u00A0\s\.]+)\s*kr\s*/\s*m²", text, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"Kvadratmeterpris.*?([\d\u00A0\s\.]+)\s*kr\s*/\s*m²", text, flags=re.IGNORECASE|re.DOTALL)
            return m.group(1).strip().replace("\u00A0", " ") if m else None

        stats = {
            "pct_12m": _pct(r"12\s*m[åa]n", txt),
            "sqm_avg_sek": _money_per_sqm(txt),
        }

        # ---------- Images sample ----------
        images_list = [
            {"url": c["url"], "path": c.get("path"), "size": c.get("size"), "ctype": c.get("ctype")}
            for c in captured if _is_under(c.get("path"), img_prefix)
        ][:30]

        # ---------- Manifest ----------
        meta = {
            "id": lid,
            "url": url,
            "ts": int(time.time()),
            "page": {"raw": "raw.html", "dom": "dom.html"},
            "counts": {"responses": len(captured), "images": images_cnt, "json": json_cnt},
            # "fields": fields,
            # "stats": stats,
            "images_sample": images_list,
            "note": {"raw_equals_dom": same_dom, "partial": err is not None, "error": err},
        }
        try:
            (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        log("saved_listing", id=lid, images=images_cnt, json=json_cnt, path=str(root))

        # Record outcome
        try:
            raw_path = str(root / "raw.html") if (root / "raw.html").exists() else ""
            con.execute("INSERT OR REPLACE INTO fetched(url,ts,path,kind) VALUES(?,?,?,?)",
                        (url, int(time.time()), raw_path, "listing"))
            con.commit()
        except Exception:
            pass
    
    # politeness sleep between listings
    delay = max(0, sleep_min + random.uniform(0, sleep_jitter))
    log("sleep", seconds=round(delay, 2))
    time.sleep(delay)

# -------- Search page handling --------

def extract_listing_urls_from_search_html(html: str, base: str) -> List[str]:
    # Very simple anchor collector: hrefs containing '/bostad/' and a trailing numeric id.
    hrefs = list(dict.fromkeys(re.findall(r'href=["\']([^"\']+)', html)))
    urls = []
    for h in hrefs:
        if "/bostad/" in h and re.search(r"(\d+)(?:$|[/?#])", h):
            u = urljoin(base, h)
            urls.append(u.split("?")[0])
    return list(dict.fromkeys(urls))

def enqueue_search_page(page, con, url: str) -> Tuple[int, Optional[str]]:
    page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    page.wait_for_load_state("networkidle", timeout=TIMEOUT_MS)
    html = page.content()
    urls = extract_listing_urls_from_search_html(html, url)
    added = add_urls_to_queue(con, urls)
    log("search_enqueued", url=url, listings=len(urls))
    # next page
    next_url = None
    m = re.search(r'(?:page=)(\d+)', url)
    if m:
        p = int(m.group(1)) + 1
        next_url = re.sub(r'(?:page=)\d+', f'page={p}', url)
    else:
        delim = "&" if "?" in url else "?"
        next_url = f"{url}{delim}page=2"
    log("search_next_preview", next_url=next_url)
    return added, next_url

# -------- Runner --------

def run_one_cycle(seed: str, max_listings: int, per_run_cap: int, pages_per_run: int,
                  sleep_min: int, sleep_jitter: int, headless: bool):

    con = sqlite3.connect(str(STATE_DB))
    ensure_state(con)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=[
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ])
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"),
            locale="sv-SE",
        )
        page = ctx.new_page()

        # seed once per cycle
        log("seeded", seed=seed)
        next_search = seed
        pages_done = 0
        total_fetched = 0

        while pages_done < pages_per_run and next_search and total_fetched < per_run_cap:
            # 1) enqueue ONE page
            try:
                enqueue_search_page(page, con, next_search)
            except PWTimeout:
                log("timeout", url=next_search)
            except Exception as e:
                log("error", url=next_search, err=repr(e))

            # 2) drain JUST-enqueued listings before moving to next page
            while total_fetched < per_run_cap:
                url = pop_next_listing(con)
                if not url:
                    break  # queue empty for this page
                if seen_or_fetched(con, url):
                    continue
                try:
                    save_listing_bundle(page, con, url, sleep_min, sleep_jitter)
                    total_fetched += 1
                except PWTimeout:
                    log("timeout", url=url)
                    con.execute("INSERT OR REPLACE INTO fetched(url,ts,path,kind) VALUES(?,?,?,?)",
                                (url, int(time.time()), "", "listing"))
                    con.commit()
                except Exception as e:
                    log("error", url=url, err=repr(e))
                    con.execute("INSERT OR REPLACE INTO fetched(url,ts,path,kind) VALUES(?,?,?,?)",
                                (url, int(time.time()), "", "listing"))
                    con.commit()

            # 3) advance to next search page only after draining
            pages_done += 1
            if total_fetched >= per_run_cap:
                break
            if "page=" in next_search:
                next_search = re.sub(r'(?:page=)\d+', lambda m: f"page={int(m.group(0).split('=')[1])+1}", next_search)
            else:
                delim = "&" if "?" in next_search else "?"
                next_search = f"{next_search}{delim}page=2"

            time.sleep(1.0)  # brief politeness pause

        browser.close()
        log("cycle_done", new=total_fetched, total=total_fetched)

        browser.close()
        log("cycle_done", new=total_fetched, total=total_fetched)

# -------- CLI --------

def parse_args():
    ap = argparse.ArgumentParser(description="Polite Hemnet crawler")
    ap.add_argument("--seed", required=True, help="Search URL to start from")
    ap.add_argument("--max-listings", type=int, default=2, help="Global maximum listings to keep overall (deprecated cap; use per-run)")
    ap.add_argument("--per-run-cap", type=int, default=2, help="How many listings to fetch in this run")
    ap.add_argument("--pages-per-run", type=int, default=5, help="How many search pages to enqueue this cycle")
    ap.add_argument("--sleep-min", type=int, default=30, help="Base sleep between listing fetches (seconds)")
    ap.add_argument("--sleep-jitter", type=int, default=30, help="Additional random sleep 0..jitter seconds")
    ap.add_argument("--headless", action="store_true", default=True, help="Run headless")
    return ap.parse_args()

def main():
    args = parse_args()
    run_one_cycle(
        seed=args.seed,
        max_listings=args.max_listings,
        per_run_cap=args.per_run_cap,
        pages_per_run=args.pages_per_run,
        sleep_min=args.sleep_min,
        sleep_jitter=args.sleep_jitter,
        headless=args.headless,
    )

if __name__ == "__main__":
    main()
