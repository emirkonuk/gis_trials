#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = ROOT / "tests" / "manual_agent_queries.txt"
API_URL = "http://127.0.0.1:8099/agent/query"
REQUEST_TIMEOUT = float(os.environ.get("AGENT_QUERY_TIMEOUT", "1200"))

if not QUERY_FILE.exists():
    print(f"missing {QUERY_FILE}", file=sys.stderr)
    sys.exit(1)

with open(QUERY_FILE, "r", encoding="utf-8") as f:
    queries = [q.strip() for q in f.readlines() if q.strip()]

session = requests.Session()
for query in queries:
    payload = {"prompt": query, "topk": 5}
    try:
        resp = session.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        status = resp.status_code
        print(f"QUERY: {query}\nSTATUS: {status}")
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        if status == 200 and isinstance(body, dict):
            plan = body.get("parsed", {})
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        else:
            if isinstance(body, dict):
                print(json.dumps(body, ensure_ascii=False, indent=2))
            else:
                print(body)
    except Exception as exc:
        print(f"QUERY FAILED: {query} -> {exc}")
    print("--")
