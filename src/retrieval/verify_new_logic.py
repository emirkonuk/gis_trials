import os
import sys
import time
import datetime
import json
import psycopg
import torch
from decimal import Decimal  # <--- Added for the test
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --- Config ---
PG_HOST = os.environ.get('PGHOST', 'db')
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
TEMP_COLLECTION = "test_sandbox_v2"  # v2 for the new test

TEXT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def log(msg):
    print(f"[SandboxTest] {msg}")

# --- THE LOGIC TO TEST (Copied from final embed_daemon.py) ---
def _make_serializable(data):
    """Recursively converts types that JSON/Qdrant hates."""
    clean = {}
    for k, v in data.items():
        if isinstance(v, (datetime.date, datetime.datetime)):
            clean[k] = v.isoformat()
        elif isinstance(v, Decimal):
            # Test: Convert Decimal to int/float
            clean[k] = int(v) if v % 1 == 0 else float(v)
        elif isinstance(v, dict):
            clean[k] = _make_serializable(v)
        else:
            clean[k] = v
    return clean

def main():
    log("Starting Sandbox Verification (v2 - Decimal Support)...")
    
    # --- 1. Connect ---
    try:
        conn = psycopg.connect(f"host={PG_HOST} dbname=gis user=gis password=gis")
        client = QdrantClient(host=QDRANT_HOST, port=6333)
    except Exception as e:
        log(f"Connection Failed: {e}")
        sys.exit(1)

    # --- 2. Fetch Real Data (Read-Only) ---
    log("Fetching 5 listings from Postgres...")
    listings = []
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM listings_attrs LIMIT 5")
        colnames = [desc.name for desc in cur.description]
        rows = cur.fetchall()
        
        for row in rows:
            row_dict = dict(zip(colnames, row))
            
            # DEBUG: Check if we actually have Decimals to test
            has_decimal = any(isinstance(v, Decimal) for v in row_dict.values())
            
            # Apply the fix
            clean_attrs = _make_serializable(row_dict)
            listings.append(clean_attrs)

            if has_decimal:
                 # Verify it was converted
                 price = clean_attrs.get('asking_price_sek')
                 if price is not None and not isinstance(price, (int, float)):
                     log(f"❌ FAILED: Price {price} is still type {type(price)}")
                     sys.exit(1)

    if not listings:
        log("Error: No listings found.")
        sys.exit(1)

    log(f"Prepared {len(listings)} listings. Serialization check passed.")

    # --- 3. Load Model ---
    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
    
    # --- 4. Create Sandbox Collection ---
    client.recreate_collection(
        collection_name=TEMP_COLLECTION,
        vectors_config={
            "text": models.VectorParams(size=384, distance=models.Distance.COSINE),
            "image": models.VectorParams(size=768, distance=models.Distance.COSINE),
        }
    )

    # --- 5. Upsert (Test if Qdrant accepts the payload) ---
    points = []
    for item in listings:
        lid = item.get('listing_id')
        text = item.get('description_short') or "test"
        vector = text_model.encode(text).tolist()
        
        points.append(models.PointStruct(
            id=abs(hash(f"test_{lid}")) % (10**18),
            vector={"text": vector, "image": [0.0] * 768},
            payload={
                "type": "test_entry",
                **item # Merge attributes
            }
        ))

    log(f"Upserting {len(points)} points to {TEMP_COLLECTION}...")
    try:
        client.upsert(collection_name=TEMP_COLLECTION, points=points, wait=True)
    except Exception as e:
        log(f"❌ UPSERT FAILED: {e}")
        log("This usually means _make_serializable failed to clean a type.")
        sys.exit(1)

    # --- 6. Verification ---
    res = client.scroll(collection_name=TEMP_COLLECTION, limit=1)[0]
    if res:
        payload = res[0].payload
        price = payload.get('asking_price_sek')
        
        log("\n--- SUCCESS! Sample Payload: ---")
        print(json.dumps(payload, indent=2, default=str))
        
        if isinstance(price, (int, float)):
             log(f"\n✅ TEST PASSED: Price is valid number: {price} ({type(price).__name__})")
        else:
             log(f"\n⚠️ WARNING: Price is missing or weird: {price}")
    else:
        log("❌ TEST FAILED: No points found.")

    # --- 7. Cleanup ---
    client.delete_collection(TEMP_COLLECTION)
    log("Sandbox deleted.")

if __name__ == "__main__":
    main()