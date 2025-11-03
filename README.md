# GIS Stack

This directory reshapes the original `map_serving` tree into a reproducible, flat layout that you can move to any host. Everything is configured to run relative to this folder – no absolute paths and no hidden state.

## Layout

```
gis-stack/
  infra/
    compose/      # docker-compose bundles (core, inference, retrieval)
    dockerfiles/  # per-service Dockerfiles
    configs/      # nginx and related runtime configs
  src/
    inference/    # FastAPI VLM service
    retrieval/    # Qdrant index + search API
    web/          # static assets served by nginx
  scripts/        # data/extraction/raster helpers invoked by bootstrap.sh
  tools/          # optional utilities (one-off rebuild helpers)
  legacy/         # archived scripts kept for reference only
  docs/           # README, reports, troubleshooting notes
  data/           # runtime storage (archives, extracted tiles, rasters, qdrant, results)
    archives/
    extracted/
    rasters/
    results/
    models/
  bootstrap.sh    # single entrypoint
  README.md
```

## One-step bootstrap

1. Copy the official archives (ZIP/TAR) into `./data/archives/`.
2. From this directory run `./bootstrap.sh`.

`bootstrap.sh` will:

- create any missing runtime folders (`data/extracted`, `data/rasters/ortho_{2011,2017}`, `data/vector`, `data/inventory`, `data/chips`, `data/qdrant`, `data/results`, `data/models`),
- remove only this project’s containers while preserving any host containers whose names include `skk-mssql` or `serene_almeida`,
- build all Docker images with `--no-cache` on the first run (use `--force-build` to rebuild later),
- start core services (Postgres/PostGIS, pg_tileserv, mbtileserver, nginx) followed by inference and retrieval stacks,
- migrate legacy rasters and retrieval chips from the previous `map_serving` layout when they exist, or rebuild them in-place when they don’t,
- rebuild mosaics, overviews, and MBTiles for 2011 and 2017 when new data arrives (or when `--force-rasters` is provided); existing `ortho_2017_rgb.tif` and `ortho_{2011,2017}.mbtiles` files are detected and reused,
- seed or refresh the retrieval pipeline (chips, embeddings, Qdrant index) and verify Qdrant health/point counts; use `--force-retrieval` to regenerate everything,
- reload vector layers into PostGIS only when they have not been loaded before (remove `data/.vectors_loaded` or pass `--force-vectors` to re-run),
- run smoke checks (`curl` against mbtileserver, pgtileserv, Nginx search proxy, and a sample raster tile) before printing the key endpoints.

## Adding data later

Drop the new archives into `data/archives/` and rerun `./bootstrap.sh`. The script now detects previously generated deliverables (`ortho_2017_rgb.tif`, MBTiles, `data/.vectors_loaded`, retrieval embeddings) and skips the heavy work unless you explicitly pass `--force-…` flags or remove the corresponding marker files. Extraction still recognises existing folders; run it with `-f` from inside the worker container if you really need to redo the unzip step.

### Running individual pieces

- **Extraction**  
  `docker compose -f infra/compose/core.yml run --rm worker bash -lc 'scripts/extract_archives.sh'`

- **Inventory**  
  `docker compose -f infra/compose/core.yml run --rm worker bash -lc 'scripts/inventory_extracted.sh'`

- **Raster rebuild (2011 + 2017 RGB fix)**  
  `docker compose -f infra/compose/core.yml run --rm worker bash -lc 'scripts/rebuild_rasters.sh'`  
  (Uses `gdal_translate -b 2 -b 3 -b 3` for 2017 and only rebuilds when `ortho_2017_rgb.tif`/MBTiles are missing; set `GIS_BOOTSTRAP_FORCE_RASTERS=1` to force it.)

- **Vector load**  
  `rm -f data/.vectors_loaded && docker compose -f infra/compose/core.yml run --rm worker bash -lc 'scripts/load_vectors_to_postgis.v2.sh'`  
  (Alternatively run `./bootstrap.sh --force-vectors`.)

- **Tile services only (db + pg_tileserv + mbtileserver + nginx)**  
  `docker compose -f infra/compose/core.yml up -d --force-recreate db pgtileserv mbtileserver web`

- **Inference stack only**  
  `docker compose -f infra/compose/core.yml -f infra/compose/inference.yml up -d --force-recreate inference web_infer`

- **Retrieval services only (Qdrant + API)**  
  `docker compose -f infra/compose/retrieval.yml up -d --force-recreate qdrant retrieval_gpu`

- **Retrieval ETL (chips → embeddings → index)**  
  `docker compose -f infra/compose/retrieval.yml run --rm retrieval_gpu bash -lc 'cd /workspace/retrieval && python3 chips_make.py && python3 embed.py && python3 index_qdrant.py'`

- **Stack smoke test (health-gated)**  
  `./scripts/check_stack.sh`

When raster filenames change, restart only the web-facing services with  
`docker compose -f infra/compose/core.yml up -d web web_infer mbtileserver`  
and rerun `./scripts/check_stack.sh` to confirm the stack. `bootstrap.sh` automatically invokes the same smoke tests at the end of every run and aborts if any endpoint fails.

## GPU assignment

`bootstrap.sh` accepts optional GPU selectors:

```
./bootstrap.sh --infer-gpu 0 --search-gpu 1
```

`--infer-gpu` controls the VLM container (`gis_inference`), while `--search-gpu` is passed to the retrieval pipeline and FastAPI search service (`gis_retrieval_gpu`). Both default to `0`. The same values are honoured by the compose files and by the ad-hoc commands that `bootstrap.sh` launches (e.g. the retrieval ETL job).

## Retrieval ETL refresh

The retrieval toolkit lives in `src/retrieval/`. During bootstrap the script will copy the legacy chips/embeddings from `map_serving` when they are present, or rebuild them (`chips_make.py`, `embed.py`, `index_qdrant.py`) if they are missing. You can force a rebuild with `./bootstrap.sh --force-retrieval ...` or trigger individual steps manually:

```bash
docker compose -f infra/compose/retrieval.yml run --rm \
  -e SEARCH_GPU=1 -e CUDA_VISIBLE_DEVICES=1 \
  retrieval_gpu bash -lc 'cd /workspace/retrieval && python3 chips_make.py && python3 embed.py && python3 index_qdrant.py'
```

The FastAPI search service exposed at `/search/text` is defined in `search_api.py` and is proxied through Nginx at `http://127.0.0.1:8082/search/`. `bootstrap.sh` fires a smoke query (`?q=roads&topk=1`) and aborts if the request fails or the Qdrant collection is undersized (< 10 000 points).

## Raster layers

Both years now publish separate artifacts under `data/rasters/ortho_{2011,2017}/`. Nginx exposes them at:

- `http://127.0.0.1:8082/raster/ortho_2017/tiles/{z}/{x}/{y}.jpg` (RGB; bands 2-3-3)
- `http://127.0.0.1:8082/raster/ortho_2011/tiles/{z}/{x}/{y}.png`

The web UIs (`web_preview.html`, `web_infer.html`) default to the 2017 RGB mosaic and include a dropdown that toggles the 2011 layer on demand.

### 2017 raster colour fix

The original `ortho_2017_seamless.vrt` referenced missing per-tile VRTs and the previous “RGB” GeoTIFF actually contained grayscale data. The current pipeline rebuilds the MBTiles straight from the raw 2017 GeoTIFF tiles with band order 2-3-3, applies JPEG encoding at 90 % quality, and generates overviews before the tile service starts.

During bootstrap we rebuild the raster (when new inputs arrive or when `--force-rasters` is used), restart mbtileserver, and run fatal smoke tests:

1. `curl -sS http://gis_mbtileserver:8090/services | grep ortho_2017`
2. `curl -s -o /tmp/mb.jpg -w '%{size_download}\n' http://gis_mbtileserver:8090/services/ortho_2017/ortho_2017/tiles/17/72170/38580.jpg`
3. `curl -s -o /tmp/ng.jpg -w '%{size_download}\n' http://127.0.0.1:8082/raster/ortho_2017/tiles/17/72170/38580.jpg`

Each tile probe must return more than 2 000 bytes. Failures print the latest logs from `gis_mbtileserver` and `gis_web_infer` and abort the bootstrap run. You can rerun the same checks on a live stack with `./scripts/check_stack.sh`.

Debug utilities that were used while developing this fix now live in `code/fixes/ortho_2017_rebuild/`. They are optional and kept separate from the main workflow.

## Host containers kept alive

`bootstrap.sh` never stops or removes containers whose names include `skk-mssql` or `serene_almeida`, honoring the original requirement. Only containers in this stack with prefixes such as `gis_`, `pgtileserv`, or `mbtileserver` are removed during cleanup.

## Notes

- All Docker Compose files run relative to `gis-stack/`; the project root can be moved or rsynced without editing paths.
- GPU-specific overrides live in `infra/compose/inference.gpu.local.yml`. If you do not have a GPU, simply omit that file.
- Legacy scripts from the previous layout reside in `legacy/` with a header noting that they are no longer wired into `bootstrap.sh`.
