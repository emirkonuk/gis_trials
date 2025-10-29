# Retrieval ETL

## Commands (CLI)
### Build, start
docker compose -f docker-compose.retrieval.yml build retrieval_gpu
docker compose -f docker-compose.retrieval.yml up -d qdrant retrieval_gpu
curl -sS http://127.0.0.1:6333/readyz

### GPU check
docker exec -it gis_retrieval_gpu nvidia-smi

### Generate chips
docker exec -it gis_retrieval_gpu bash -lc "python3 chips_make.py"

### Embed
docker exec -it gis_retrieval_gpu bash -lc "python3 embed.py"

### Index
docker exec -it gis_retrieval_gpu bash -lc "python3 index_qdrant.py"
curl -s "http://127.0.0.1:6333/collections/sweden_demo_v0"

### Search
docker exec -it gis_retrieval_gpu bash -lc "python3 search.py --text 'forest near buildings' --topk 5"

## Notes
- Chips source: HTTP raster at 127.0.0.1:8080 if available, else first *.mbtiles under /project/data is used.
- Outputs: /project/data/chips/{PNG, chips_index.csv, embeddings.npy, metadata.parquet}
- Models cache: /model_cache -> docker_mount/project/models
