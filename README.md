Here is the README for this phase.
Save as `/storage/ekonuk_spare/gis/lantmateriet/map_serving/README.md`.

---

# GIS Map Serving Stack — Browser + QGIS Access

## Purpose

This phase builds a complete **local web map server** for your Lantmäteriet data.
It ingests vector and raster archives, loads them into PostGIS, creates color-correct raster mosaics, and exposes everything via browser (MapLibre) and QGIS.

You can now **see, inspect, and query Swedish geospatial layers** as web tiles exactly like any online map, but all running locally.

---

## Components

| Layer                | Role               | Exposed at                                     | Description                                                                                                                                         |
| -------------------- | ------------------ | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PostGIS (db)**     | Spatial database   | internal only                                  | Stores all vector data (geometry + attributes). Each dataset becomes a table under schema `lm`.                                                     |
| **pg_tileserv**      | Vector tile server | [http://127.0.0.1:7800](http://127.0.0.1:7800) | Reads from PostGIS and serves each table as **Mapbox Vector Tiles (MVT)**. Each table name appears as a “collection.”                               |
| **mbtileserver**     | Raster tile server | [http://127.0.0.1:8090](http://127.0.0.1:8090) | Serves prebuilt `.mbtiles` rasters (JPEG pyramids). You can preview at `/services/ortho_2017_demo/map`.                                             |
| **worker**           | GDAL environment   | internal only                                  | Runs the processing pipeline (`main.sh`): unzips, inventories, loads to PostGIS, builds mosaics, fixes color channels (2-3-3), and creates MBTiles. |
| **web (Nginx)**      | Unified proxy      | [http://127.0.0.1:8080](http://127.0.0.1:8080) | Serves `web_preview.html`. Proxies `/vector/*` → `pg_tileserv` and `/raster/*` → `mbtileserver` with correct CORS headers.                          |
| **web_preview.html** | Map viewer         | via 8080                                       | Uses MapLibre to display both raster and vector tiles in the browser. Layer toggles read from `layer_names.json`.                                   |

---

## Data Flow

```
ZIP archives
   ↓
extract_archives.sh
   ↓
data/extracted/
   ↓
inventory_extracted.sh
   ↓
data/inventory/*.txt
   ↓
load_vectors_to_postgis.v2.sh  →  PostGIS tables (served by pg_tileserv)
   ↓
build_raster_mosaic.sh  →  ortho_2017_seamless.vrt
   ↓
build_raster_overview.sh  →  ortho_2017_seamless.vrt.ovr
   ↓
gdal_translate (-b 2 -b 3 -b 3)  →  ortho_2017_demo.mbtiles (served by mbtileserver)
```

---

## How to Run

```bash
cd /storage/ekonuk_spare/gis/lantmateriet/map_serving
bash ./start_stack.sh
```

This will:

1. Build the worker image from `docker/Dockerfile.worker`.
2. Start all containers.
3. Run `/project/code/main.sh` inside the worker.
4. Restart servers and expose them on localhost.

After completion:

* **Web preview:** [http://127.0.0.1:8080](http://127.0.0.1:8080)
  Raster + vector layers together.

* **Vector catalog:** [http://127.0.0.1:7800/collections](http://127.0.0.1:7800/collections)

* **Raster service JSON:** [http://127.0.0.1:8090/tiles.json](http://127.0.0.1:8090/tiles.json)

---

## What Each Output Is

| File / Folder                              | Meaning                                                        |
| ------------------------------------------ | -------------------------------------------------------------- |
| `data/archives/`                           | Original ZIP downloads. Input only.                            |
| `data/extracted/`                          | Unpacked shapefiles, geopackages, and TIFFs.                   |
| `data/inventory/*.txt`                     | Lists of extracted datasets used for automation.               |
| `data/rasters/ortho_2017_seamless.vrt`     | Virtual mosaic of all 2017 orthophotos.                        |
| `data/rasters/ortho_2017_demo.mbtiles`     | Compressed, color-fixed raster pyramid used by `mbtileserver`. |
| `data/rasters/ortho_2017_seamless.vrt.ovr` | Overviews for faster reads.                                    |
| `layer_names.json`                         | Human-readable names for vector tables.                        |
| `web_preview.html`                         | MapLibre viewer.                                               |
| `.env`                                     | Database credentials shared across containers.                 |

---

## Using the Outputs in QGIS

### A) Load Raster (MBTiles)

1. In QGIS: *Layer → Add Layer → Add Raster Layer…*
2. Source type: *Database → MBTiles*.
3. Path:

   ```
   /storage/ekonuk_spare/gis/lantmateriet/map_serving/docker_mount/project/data/rasters/ortho_2017_demo.mbtiles
   ```
4. CRS: EPSG:3857 (Web Mercator) — same as browser view.

### B) Load Vector Layers from PostGIS

1. *Browser panel → PostGIS → New Connection*

   ```
   Host: localhost
   Port: 55432
   Database: gis
   Username: ekonuk
   Password: CHANGE_ME_STRONG
   ```
2. Click *Connect* → schema `lm`.
3. Add any table (e.g. `lm.ay_riks`).
   QGIS will request tiles via SQL directly from the PostGIS instance.

### C) Alternative: Load Vector Tiles via URL

1. *Layer → Add Layer → Add Vector Tile Layer → New URL…*
2. Enter:

   ```
   http://127.0.0.1:8080/vector/lm.ay_riks/{z}/{x}/{y}.pbf
   ```
3. QGIS will render live MVTs directly from pg_tileserv.

---

## Notes for ML Context

* **Raster (MBTiles)**: each tile is a JPEG 256×256 patch of the orthophoto mosaic in RGB order (bands 2-3-3). Perfect for inference, patch extraction, or downstream dataset generation.
* **Vector (PostGIS)**: geometry + metadata; ideal for semantic supervision (roads, parcels, buildings). You can query by bounding box or join attributes for labeling.
* **Coordinate Reference System:** EPSG:3857, standard Web Mercator, aligns with online basemaps and most deep-learning tilers.
* **Pipeline reproducibility:** all transformations run inside the `worker` container, ensuring identical GDAL and PostGIS versions across environments.

---

**End of README**

