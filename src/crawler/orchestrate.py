#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, sqlite3, subprocess, sys, time, random
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/project").resolve()
DATA_ROOT = PROJECT_ROOT / "data" / "listings_raw" / "hemnet"
SNAP_DIR = DATA_ROOT / "snapshots"
STATE_DB = DATA_ROOT / "state.sqlite"

def log(event, **kw):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    msg = " ".join(f"{k}={repr(v)}" for k,v in kw.items())
    print(f"{ts} {event} {msg}".strip(), flush=True)

def ensure_state(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS q (
            url TEXT PRIMARY KEY,
            seen INTEGER NOT NULL DEFAULT 0,
            enq_ts INTEGER NOT NULL DEFAULT (strftime('%s','now'))
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

def mark_existing_as_fetched():
    """Scan snapshots/*, read meta.json, and upsert into fetched(url,ts,path,kind)."""
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(STATE_DB))
    ensure_state(con)
    added = 0
    for snap in sorted(SNAP_DIR.iterdir()):
        if not snap.is_dir():
            continue
        meta = snap / "meta.json"
        raw  = snap / "raw.html"
        if not meta.exists():
            continue
        try:
            j = json.loads(meta.read_text(encoding="utf-8"))
            url = j.get("url")
            if not url:
                continue
            ts  = int(j.get("ts") or time.time())
            path = str(raw) if raw.exists() else ""
            con.execute(
                "INSERT OR IGNORE INTO fetched(url,ts,path,kind) VALUES(?,?,?,?)",
                (url, ts, path, "listing"),
            )
            added += con.total_changes
        except Exception as e:
            log("warn_meta_read", dir=str(snap), err=repr(e))
    con.commit()
    con.close()
    log("skip_seeded", fetched_marked=added)

def launch_and_wait_for_crawler(seed: str, pages_per_run: int, per_run_cap: int, sleep_min: int, sleep_jitter: int):
    """
    Runs the crawler as a subprocess and WAITS for it to complete.
    This is now part of the main sequential loop.
    """
    cmd = [
        "python3", "/project/src/crawler/hemnet_crawl.py",
        "--seed", seed,
        "--pages-per-run", str(pages_per_run),
        "--per-run-cap",  str(per_run_cap),
        "--sleep-min",    str(sleep_min),
        "--sleep-jitter", str(sleep_jitter),
    ]
    log("crawler_exec", cmd=" ".join(cmd))
    
    # Use subprocess.run() to wait for it to finish.
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    
    rc = proc.returncode
    log("crawler_exit", code=rc)
    if rc != 0:
        log("crawler_fail", msg="Crawler exited with non-zero status, pausing for 1 hour.")
        time.sleep(3600) # Wait an hour if the crawl fails
        return False # Indicate failure
    return True # Indicate success


def run_processor_sweep(max_to_process: int = 1_000_000):
    """
    Finds all unparsed snapshots and runs soup.py and db_upsert.py
    on them sequentially.
    """
    soup_script = "/project/src/crawler/soup.py"
    db_script = "/project/src/crawler/db_upsert.py"
    done = 0
    
    snapshots_to_process = []
    for snap in sorted(SNAP_DIR.iterdir()):
        if not snap.is_dir():
            continue
        out = snap / "parsed.json"
        dom = snap / "dom.html"
        if not out.exists() and dom.exists():
            snapshots_to_process.append(snap)
            
    if not snapshots_to_process:
        log("process_sweep_start", pending=0)
        return 0

    log("process_sweep_start", pending=len(snapshots_to_process), max=max_to_process)
    
    for snap in snapshots_to_process:
        if done >= max_to_process:
            log("process_sweep_limit", limit=max_to_process)
            break
            
        out = snap / "parsed.json"
        dom = snap / "dom.html"

        cmd_soup = ["python3", soup_script, str(dom)]
        log("parse_exec", snapshot=str(snap))
        rc_soup = subprocess.call(cmd_soup)
        
        if rc_soup == 0 and out.exists():
            log("parsed_ok", path=str(out))
            
            cmd_db = ["python3", db_script, str(snap)]
            log("db_upsert_exec", snapshot=str(snap))
            rc_db = subprocess.call(cmd_db)
            
            if rc_db == 0:
                log("db_upsert_ok", snapshot=str(snap))
                done += 1
            else:
                log("db_upsert_fail", dir=str(snap), code=rc_db)
        else:
            log("parsed_fail", dir=str(snap), code=rc_soup)
            
    log("process_sweep_done", processed=done)
    return done

def is_quiet_time(start_hour=23, end_hour=7) -> bool:
    """Checks if the current time is within the 'quiet' non-crawling period."""
    current_hour = datetime.now().hour
    if start_hour > end_hour: # Overnight case (e.g., 23:00 to 07:00)
        return current_hour >= start_hour or current_hour < end_hour
    else: # Daytime case (e.g., 09:00 to 17:00)
        return start_hour <= current_hour < end_hour

# --- main: run as a continuous daemon ---
def main():
    ap = argparse.ArgumentParser(description="Orchestrate Hemnet crawl, parse, and load in a continuous loop.")
    ap.add_argument("--seed", required=True)
    # Read from env vars, falling back to new defaults
    ap.add_argument("--pages-per-run", type=int, default=os.environ.get("PAGES_PER_RUN", 10))
    ap.add_argument("--per-run-cap",  type=int, default=os.environ.get("PER_RUN_CAP", 250))
    ap.add_argument("--sleep-min",    type=int, default=os.environ.get("SLEEP_MIN", 30))
    ap.add_argument("--sleep-jitter", type=int, default=os.environ.get("SLEEP_JITTER", 30))
    # Cycle sleep times
    ap.add_argument("--cycle-sleep-min", type=int, default=os.environ.get("CYCLE_SLEEP_MIN", 300)) # 5 min default
    ap.add_argument("--cycle-sleep-max", type=int, default=os.environ.get("CYCLE_SLEEP_MAX", 600)) # 10 min default

    args = ap.parse_args()

    log("daemon_start", **vars(args))

    while True:
        try:
            # 1. Check for Quiet Time
            if is_quiet_time(start_hour=23, end_hour=7):
                log("quiet_time", status="active", sleep_seconds=600)
                time.sleep(600) # Sleep for 10 minutes and check again
                continue

            log("cycle_start", status="running_crawl")
            
            # 2. Mark existing and launch crawler (BLOCKING)
            mark_existing_as_fetched()
            crawl_success = launch_and_wait_for_crawler(
                args.seed, 
                args.pages_per_run, 
                args.per_run_cap, 
                args.sleep_min, 
                args.sleep_jitter
            )
            
            # 3. Run processor sweep (BLOCKING)
            if crawl_success:
                run_processor_sweep()

            # 4. Sleep until next cycle
            sleep_duration = random.randint(args.cycle_sleep_min, args.cycle_sleep_max)
            log("cycle_complete", sleep_seconds=sleep_duration)
            time.sleep(sleep_duration)
        
        except Exception as e:
            log("daemon_loop_error", error=repr(e), action="restarting_loop_in_60s")
            time.sleep(60)


if __name__ == "__main__":
    main()