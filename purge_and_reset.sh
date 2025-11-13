#!/bin/bash
set -e

# Get the directory of this script to find the project root
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Project Root: $ROOT"

echo "--- 1. Stopping all services ---"
echo "Stopping services using docker compose..."
docker compose -f "$ROOT/infra/compose/crawler.yml" down --remove-orphans || true
docker compose -f "$ROOT/infra/compose/retrieval.yml" down --remove-orphans || true
echo "Services stopped."

echo "--- 2. Archiving polluted snapshots ---"
SNAPSHOT_DIR="$ROOT/data/listings_raw/hemnet/snapshots"
ARCHIVE_DIR="$ROOT/data/listings_raw/hemnet/snapshots_polluted_$(date +%F_%H-%M-%S)"
if [ -d "$SNAPSHOT_DIR" ]; then
    echo "Archiving $SNAPSHOT_DIR to $ARCHIVE_DIR"
    mv "$SNAPSHOT_DIR" "$ARCHIVE_DIR"
    mkdir -p "$SNAPSHOT_DIR"
else
    echo "Snapshot directory not found, skipping."
fi

echo "--- 3. Purging crawler state database ---"
STATE_DB="$ROOT/data/listings_raw/hemnet/state.sqlite"
if [ -f "$STATE_DB" ]; then
    echo "Deleting $STATE_DB"
    rm -f "$STATE_DB"
else
    echo "State DB not found, skipping."
fi

# --- MODIFIED STEP 4 ---
echo "--- 4. Purging and recreating Qdrant collection ---"
docker compose -f "$ROOT/infra/compose/retrieval.yml" up -d qdrant
echo "Waiting for Qdrant service to start..."
sleep 5 # Give the container time to start

echo "Running purge_qdrant.py script inside the container..."
# This command runs the new Python script, which has its own retry logic
docker compose -f "$ROOT/infra/compose/retrieval.yml" \
    run --rm -T retrieval_gpu \
    python3 /workspace/retrieval/purge_qdrant.py
# Note: The path /workspace/retrieval matches the volume mount in retrieval.yml

echo "Qdrant purge complete."

# --- STEP 5 ---
echo "--- 5. Purging Postgres tables ---"
echo ""
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!! ACTION REQUIRED !!"
echo "!! Please open a NEW terminal and run your ./bootstrap.sh"
echo "!! to start all services (including the 'db' container)."
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo ""
read -p "Once all services are running, press [ENTER] to continue..."

DB_CONTAINER_NAME=""
if docker ps --format '{{.Names}}' | grep -q "^gis_db$"; then
    DB_CONTAINER_NAME="gis_db"
elif docker ps --format '{{.Names}}' | grep -q "^db$"; then
    DB_CONTAINER_NAME="db"
else
    echo "Error: Could not find a running container named 'gis_db' or 'db'."
    echo "Please ensure the Postgres container is running and named correctly."
    exit 1
fi

echo "Found Postgres container: $DB_CONTAINER_NAME. Truncating tables..."
docker exec "$DB_CONTAINER_NAME" \
    psql -U gis -d gis -c " \
    TRUNCATE TABLE public.listings_attrs, \
                   public.listings_images, \
                   public.embedding_queue \
    RESTART IDENTITY CASCADE; \
    SELECT 'Postgres tables truncated.'; \
    "

# --- STEP 6 ---
echo "--- 6. Stopping all services ---"
echo "Stopping services. You will restart them with bootstrap.sh after this."
docker compose -f "$ROOT/infra/compose/crawler.yml" down --remove-orphans || true
docker compose -f "$ROOT/infra/compose/retrieval.yml" down --remove-orphans || true
# Stop any other containers bootstrap might have started
docker stop "$DB_CONTAINER_NAME" || true
docker stop gis_qdrant || true # qdrant was started by this script

echo "--- RESET COMPLETE ---"
echo "You can now restart services with bootstrap.sh"