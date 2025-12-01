import os
import sys
import psycopg
import datetime
from decimal import Decimal
from qdrant_client import QdrantClient, models

def _make_serializable(data):
    clean = {}
    for k, v in data.items():
        if isinstance(v, (datetime.date, datetime.datetime)):
            clean[k] = v.isoformat()
        elif isinstance(v, Decimal):
            clean[k] = int(v) if v % 1 == 0 else float(v)
        elif isinstance(v, dict):
            clean[k] = _make_serializable(v)
        else:
            clean[k] = v
    return clean

def main():
    print("--- Starting Attribute Backfill (Robust) ---")
    
    pg_host = os.environ.get('PGHOST', 'db')
    conn_str = f"host={pg_host} dbname=gis user=gis password=gis"
    
    try:
        conn = psycopg.connect(conn_str)
    except Exception as e:
        print(f"DB Connect Fail: {e}")
        sys.exit(1)

    client = QdrantClient(host="qdrant", port=6333)
    COLLECTION = "hemnet_listings_v1"

    try:
        with conn.cursor() as cur:
            print("Fetching all attributes from Postgres...")
            cur.execute("SELECT * FROM public.listings_attrs")
            colnames = [desc.name for desc in cur.description]
            rows = cur.fetchall()
            
            print(f"Found {len(rows)} listings to update.")
            
            for i, row in enumerate(rows):
                row_dict = dict(zip(colnames, row))
                lid = row_dict.get('listing_id')
                if not lid: continue

                payload_update = _make_serializable(row_dict)
                
                # Find ALL points for this listing_id (with pagination)
                scroll_filter = models.Filter(
                    must=[models.FieldCondition(key="listing_id", match=models.MatchValue(value=lid))]
                )
                
                next_offset = None
                total_updated = 0
                
                while True:
                    points, next_offset = client.scroll(
                        collection_name=COLLECTION,
                        scroll_filter=scroll_filter,
                        limit=1000, # Batch size
                        offset=next_offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    if not points:
                        break
                        
                    point_ids = [p.id for p in points]
                    client.set_payload(
                        collection_name=COLLECTION,
                        payload=payload_update,
                        points=point_ids
                    )
                    
                    total_updated += len(point_ids)
                    
                    if next_offset is None:
                        break

                if (i+1) % 50 == 0:
                    print(f"Updated listing {lid} ({i+1}/{len(rows)}) - {total_updated} vectors")

    except Exception as e:
        print(f"Critical Error: {e}")
    finally:
        conn.close()

    print("--- Backfill Complete ---")

if __name__ == "__main__":
    main()