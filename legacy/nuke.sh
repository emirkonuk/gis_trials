# kept for reference from old map_serving layout; not used by bootstrap.sh
#!/usr/bin/env bash
set -euo pipefail

# Stop/remove this stack’s containers by explicit names (compose uses container_name)
docker rm -f gis_web gis_pgtileserv gis_mbtileserver gis_worker gis_db 2>/dev/null || true

# Bring the project fully down and remove volumes/networks from this compose dir
docker compose down -v --remove-orphans || true

# Sanity: show anything still binding 8080/7800/8090
echo "Ports in use (if any):"
( ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null || true ) | egrep ':8080|:7800|:8090' || true
echo "Done."
