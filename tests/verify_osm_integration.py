#!/usr/bin/env python3
"""Smoke test covering OSM parsing and resolution."""
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_ROOT = ROOT / "src" / "retrieval"
if str(RETRIEVAL_ROOT) not in sys.path:
    sys.path.append(str(RETRIEVAL_ROOT))

from osm_parser import heuristic_osm_filters  # type: ignore  # noqa: E402
from osm_service import OSMQuerySpec, OSMAmenitySpec, OSMLocationSpec, OSMService  # type: ignore  # noqa: E402


def ensure_postgres_env():
    os.environ["PGHOST"] = "127.0.0.1"
    os.environ["PGPORT"] = os.environ.get("PGHOST_PORT", "55432")
    os.environ["PGUSER"] = "gis"
    os.environ["PGPASSWORD"] = "gis"
    os.environ["PGDATABASE"] = "gis"


def test_llm_parsing():
    spec = heuristic_osm_filters("Two rooms in Åkeshov near a gym")
    assert spec is not None, "OSM filters missing from plan"
    assert spec.location and "åkeshov" in spec.location.value.lower(), "Åkeshov not detected"
    assert any(a.type == "gym" for a in spec.amenities), "Gym amenity missing from parse"


def test_resolution_and_context(service: OSMService):
    base_spec = OSMQuerySpec(location=OSMLocationSpec(type="neighborhood", value="Åkeshov"))
    context = service.resolve_query(base_spec)
    assert context.get("location"), "Location context missing"
    assert context["location"].get("geometry"), "Location geometry missing"

    amenity_spec = OSMQuerySpec(
        location=OSMLocationSpec(type="neighborhood", value="Åkeshov"),
        amenities=[OSMAmenitySpec(type="gym")]
    )
    amenity_context = service.resolve_query(amenity_spec)
    gyms = amenity_context.get("amenities", [])
    assert gyms, "No gyms returned for Åkeshov"


def test_stockholm_school_context(service: OSMService):
    spec = OSMQuerySpec(
        location=OSMLocationSpec(type="municipality", value="Stockholm"),
        amenities=[OSMAmenitySpec(type="school")]
    )
    context = service.resolve_query(spec)
    assert context.get("location"), "Stockholm location missing"
    assert context.get("filter_geometry") and context["filter_geometry"]["type"] in ("Polygon", "MultiPolygon"), "Stockholm geometry missing"
    schools = context.get("amenities", [])
    assert schools, "No schools returned for Stockholm"


def main():
    ensure_postgres_env()
    test_llm_parsing()
    service = OSMService()
    test_resolution_and_context(service)
    test_stockholm_school_context(service)
    print("SUCCESS: OSM Integration Verified")


if __name__ == "__main__":
    main()
