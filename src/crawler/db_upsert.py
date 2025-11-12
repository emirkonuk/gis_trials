#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
db_upsert.py: Loads data from a single snapshot directory (meta.json and
parsed.json) into the PostgreSQL database.

This script is idempotent. It will:
1.  Create the tables (listings_raw, listings_images, listings_attrs, 
    and embedding_queue) if they don't exist.
2.  Dynamically add new columns to listings_attrs if new fields are found
    in parsed.json.
3.  UPSERT the data for listings_raw and listings_attrs.
4.  DELETE/INSERT images for listings_images to ensure they are in sync.
5.  Add the listing_id to the embedding_queue.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import psycopg
except ImportError:
    print("Error: 'psycopg' library not found. Please install it (e.g., pip install psycopg)", file=sys.stderr)
    sys.exit(1)

def log(event, **kw):
    """Simple structured logger."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    msg = " ".join(f"{k}={repr(v)}" for k,v in kw.items())
    print(f"{ts} {event} {msg}".strip(), flush=True)

def get_db_connection() -> Optional[psycopg.Connection]:
    """Establishes database connection using environment variables."""
    try:
        conn_str = f"host={os.environ.get('PGHOST', 'db')} " \
                   f"dbname={os.environ.get('PGDATABASE', 'gis')} " \
                   f"user={os.environ.get('PGUSER', 'gis')} " \
                   f"password={os.environ.get('PGPASSWORD', 'gis')}"
        conn = psycopg.connect(conn_str)
        return conn
    except (psycopg.OperationalError, KeyError) as e:
        log("db_connect_fail", error=repr(e))
        log("db_connect_fail", msg="Ensure PGHOST, PGDATABASE, PGUSER, and PGPASSWORD env vars are set.")
        return None

def create_tables_if_not_exist(cur: psycopg.Cursor):
    """Creates the three core tables if they don't already exist."""
    
    # 1. Master listings table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS public.listings_raw (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            fetched_at TIMESTAMPTZ NOT NULL,
            snapshot_path TEXT NOT NULL
        );
    """)
    
    # 2. Images table (one-to-many)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS public.listings_images (
            id SERIAL PRIMARY KEY,
            listing_id TEXT NOT NULL REFERENCES public.listings_raw(id) ON DELETE CASCADE,
            image_url TEXT NOT NULL,
            local_path TEXT,
            image_size_bytes BIGINT
        );
        CREATE INDEX IF NOT EXISTS idx_listings_images_listing_id ON public.listings_images(listing_id);
    """)

    # 3. Attributes table (one-to-one, dynamically extended)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS public.listings_attrs (
            listing_id TEXT PRIMARY KEY REFERENCES public.listings_raw(id) ON DELETE CASCADE,
            geom geometry(Point, 4326)
        );
        CREATE INDEX IF NOT EXISTS listings_attrs_geom_idx ON public.listings_attrs USING GIST(geom);
    """)
    log("db_ensure_tables", status="ok")

def create_embedding_queue_if_not_exist(cur: psycopg.Cursor):
    """
    Creates the embedding_queue table if it doesn't exist.
    """
    cur.execute("""
        CREATE TABLE IF NOT EXISTS public.embedding_queue (
            listing_id TEXT PRIMARY KEY REFERENCES public.listings_raw(id) ON DELETE CASCADE,
            processed BOOLEAN DEFAULT false,
            created_at TIMESTAMPTZ DEFAULT now(),
            processed_at TIMESTAMPTZ
        );
        CREATE INDEX IF NOT EXISTS idx_embedding_queue_processed ON public.embedding_queue(processed);
    """)
    log("db_ensure_queue_table", status="ok")

def schedule_for_embedding(cur: psycopg.Cursor, listing_id: str):
    """
    Adds a listing_id to the embedding_queue.
    ON CONFLICT DO NOTHING makes this idempotent.
    """
    cur.execute("""
        INSERT INTO public.embedding_queue (listing_id)
        VALUES (%s)
        ON CONFLICT (listing_id) DO NOTHING;
    """, (listing_id,))
    log("db_queue_embedding", id=listing_id, rows=cur.rowcount)


def _get_pg_type(value: Any) -> str:
    """Maps a Python type to a PostgreSQL data type."""
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "BIGINT"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"

def ensure_columns_exist(cur: psycopg.Cursor, attrs: Dict[str, Any]):
    """
    Checks if all keys in `attrs` exist as columns in `listings_attrs`.
    If not, it adds them.
    """
    RESERVED_COLUMNS = {'listing_id', 'geom'}
    
    # Get existing columns from the database
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'listings_attrs';
    """)
    existing_cols = {row[0] for row in cur.fetchall()}
    
    # Find what's missing
    new_cols = set(attrs.keys()) - existing_cols - RESERVED_COLUMNS
    
    if not new_cols:
        return # All columns already exist
    
    # Add new columns
    for col_name in sorted(list(new_cols)):
        col_type = _get_pg_type(attrs[col_name])
        # Use psycopg.sql for safe identifiers
        sql = psycopg.sql.SQL("ALTER TABLE public.listings_attrs ADD COLUMN {} {};").format(
            psycopg.sql.Identifier(col_name),
            psycopg.sql.SQL(col_type)
        )
        cur.execute(sql)
        log("db_add_column", table="listings_attrs", column=col_name, type=col_type)

def upsert_raw(cur: psycopg.Cursor, listing_id: str, url: str, ts: int, snapshot_path: str):
    """Inserts or updates the master record in listings_raw."""
    cur.execute("""
        INSERT INTO public.listings_raw (id, url, fetched_at, snapshot_path)
        VALUES (%s, %s, to_timestamp(%s), %s)
        ON CONFLICT (id) DO UPDATE SET
            url = EXCLUDED.url,
            fetched_at = EXCLUDED.fetched_at,
            snapshot_path = EXCLUDED.snapshot_path;
    """, (listing_id, url, ts, snapshot_path))
    log("db_upsert_raw", id=listing_id)

def upsert_images(cur: psycopg.Cursor, listing_id: str, images: List[Dict[str, Any]]):
    """Deletes old images and inserts current ones for a listing."""
    
    cur.execute("DELETE FROM public.listings_images WHERE listing_id = %s;", (listing_id,))
    
    insert_data = []
    for img in images:
        if img.get("url"):
            insert_data.append((
                listing_id,
                img.get("url"),
                img.get("path"),
                img.get("size")
            ))
    
    if not insert_data:
        log("db_upsert_images", id=listing_id, count=0)
        return

    cur.executemany("""
        INSERT INTO public.listings_images (listing_id, image_url, local_path, image_size_bytes)
        VALUES (%s, %s, %s, %s);
    """, insert_data)
    log("db_upsert_images", id=listing_id, count=len(insert_data))

def upsert_attrs(cur: psycopg.Cursor, listing_id: str, attrs: Dict[str, Any]):
    """Dynamically inserts or updates all key-value pairs in listings_attrs."""
    
    ensure_columns_exist(cur, attrs)
    
    db_attrs = {k: v for k, v in attrs.items() if v is not None}
    
    cols = ["listing_id"] + list(db_attrs.keys())
    values = [listing_id] + list(db_attrs.values())

    cols_sql = psycopg.sql.SQL(", ").join(map(psycopg.sql.Identifier, cols))
    placeholders_sql = psycopg.sql.SQL(", ").join(map(psycopg.sql.Literal, values))
    
    update_pairs = [
        psycopg.sql.SQL("{} = EXCLUDED.{}").format(psycopg.sql.Identifier(col), psycopg.sql.Identifier(col))
        for col in cols if col != "listing_id"
    ]
    update_sql = psycopg.sql.SQL(", ").join(update_pairs)

    query = psycopg.sql.SQL("""
        INSERT INTO public.listings_attrs ({cols})
        VALUES ({vals})
        ON CONFLICT (listing_id) DO UPDATE SET {updates};
    """).format(
        cols=cols_sql,
        vals=placeholders_sql,
        updates=update_sql
    )
    
    cur.execute(query)

    lat = attrs.get("latitude")
    lon = attrs.get("longitude")
    if lat is not None and lon is not None:
        cur.execute("""
            UPDATE public.listings_attrs
            SET geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326)
            WHERE listing_id = %s;
        """, (lon, lat, listing_id))
    
    log("db_upsert_attrs", id=listing_id, cols=len(cols))


def main(snapshot_dir_str: str):
    """Main execution function."""
    snapshot_dir = Path(snapshot_dir_str).resolve()
    if not snapshot_dir.is_dir():
        log("main_fail", error="Snapshot directory not found", path=snapshot_dir_str)
        sys.exit(1)

    meta_file = snapshot_dir / "meta.json"
    parsed_file = snapshot_dir / "parsed.json"

    if not meta_file.exists():
        log("main_fail", error="meta.json not found", path=str(meta_file))
        sys.exit(1)
        
    if not parsed_file.exists():
        log("main_fail", error="parsed.json not found (run soup.py first?)", path=str(parsed_file))
        sys.exit(1)

    log("main_start", path=str(snapshot_dir))

    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
    parsed_data = json.loads(parsed_file.read_text(encoding="utf-8"))
    
    listing_id = meta_data.get("id")
    if not listing_id:
        log("main_fail", error="No 'id' field in meta.json")
        sys.exit(1)

    conn = get_db_connection()
    if not conn:
        sys.exit(1)

    try:
        with conn.cursor() as cur:
            # Create all tables on the fly
            create_tables_if_not_exist(cur)
            create_embedding_queue_if_not_exist(cur)
            
            upsert_raw(
                cur,
                listing_id,
                meta_data.get("url"),
                meta_data.get("ts"),
                str(snapshot_dir)
            )
            
            schedule_for_embedding(cur, listing_id)
            
            upsert_images(cur, listing_id, meta_data.get("images_sample", []))
            
            upsert_attrs(cur, listing_id, parsed_data)

        conn.commit()
        log("main_success", id=listing_id, path=str(snapshot_dir))
    
    except Exception as e:
        log("main_fail", error=repr(e), id=listing_id)
        conn.rollback()
    
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python db_upsert.py /path/to/snapshot/directory", file=sys.stderr)
        sys.exit(1)
    
    main(sys.argv[1])