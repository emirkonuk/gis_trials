#!/usr/bin/env python3
"""Lightweight heuristics for extracting OSM intent from natural language."""
from __future__ import annotations

import re
from typing import Optional

from osm_service import OSMQuerySpec, OSMAmenitySpec, OSMLocationSpec


AMENITY_KEYWORDS = {
    "gym": "gym",
    "gyms": "gym",
    "fitness": "gym",
    "school": "school",
    "schools": "school",
    "park": "park",
    "parks": "park",
    "hospital": "hospital",
    "hospitals": "hospital",
    "metro": "public_transport",
    "station": "public_transport",
}


def heuristic_osm_filters(query: str) -> Optional[OSMQuerySpec]:
    """Return a coarse OSM intent object used as a fallback when the LLM is offline."""
    if not query:
        return None

    found_amenities = []
    lowered = query.lower()
    for token, normalized in AMENITY_KEYWORDS.items():
        if token in lowered:
            found_amenities.append(OSMAmenitySpec(type=normalized, relation="near"))

    loc_match = re.search(
        r"\b(?:in|at|within|around)\s+([A-Za-zÅÄÖåäö0-9\-\s]+?)(?:\s+(?:near|close|around|by)\b|$)",
        query,
        flags=re.IGNORECASE,
    )
    location_value = loc_match.group(1).strip(" ,.") if loc_match else None

    if not location_value and not found_amenities:
        return None

    location_spec = OSMLocationSpec(type="neighborhood", value=location_value) if location_value else None
    return OSMQuerySpec(location=location_spec, amenities=found_amenities)
