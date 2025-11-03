# ortho_2017 manual rebuild

This helper was used while debugging the 2017 raster colour pipeline. It tiles every `*2017*.tif` under `data/extracted/` with band order 2-3-3 and writes a fresh `data/rasters/ortho_2017/ortho_2017.mbtiles`.

You should not need it during normal operation: `bootstrap.sh` and `scripts/rebuild_rasters.sh` already regenerate colour-corrected MBTiles automatically. Keep this script only if you need to experiment or run ad-hoc rebuilds without touching the main workflow:

```bash
docker compose -f infra/compose/core.yml run --rm worker bash -lc 'tools/ortho_2017_rebuild/rebuild_2017_from_sources.sh'
```

Remove the directory if you do not plan to run manual rebuilds.
