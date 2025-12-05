#!/usr/bin/env python3
"""PostGIS-backed helpers for resolving OSM geometries."""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, Field, field_validator


class OSMLocationSpec(BaseModel):
    """Structured description of the requested location."""

    type: Literal["neighborhood", "district", "municipality", "address", "bbox", "unknown"] = "neighborhood"
    value: str

    @field_validator("value")
    @classmethod
    def _clean_value(cls, val: str) -> str:
        return val.strip()


class OSMAmenitySpec(BaseModel):
    """Structured description of a requested amenity constraint."""

    type: str
    relation: Literal["near", "within", "inside", "around", "intersects", "adjacent", "unknown"] = "near"

    @field_validator("type")
    @classmethod
    def _clean_type(cls, val: str) -> str:
        return val.strip().lower()


class OSMQuerySpec(BaseModel):
    """Container for all OSM related filters."""

    location: Optional[OSMLocationSpec] = None
    address: Optional[str] = None
    amenities: List[OSMAmenitySpec] = Field(default_factory=list)

    @field_validator("address")
    @classmethod
    def _clean_address(cls, val: Optional[str]) -> Optional[str]:
        return val.strip() if val else val


def _geometry_from_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _point_bbox(lat: float, lon: float, meters: float = 3000.0) -> Dict[str, Any]:
    """Create a simple square buffer around a point for polygon-only clients."""
    if meters <= 0:
        meters = 200.0
    delta_lat = meters / 111_320.0
    denom = max(math.cos(math.radians(lat)), 0.2)
    delta_lon = meters / (111_320.0 * denom)
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - delta_lon, lat - delta_lat],
            [lon + delta_lon, lat - delta_lat],
            [lon + delta_lon, lat + delta_lat],
            [lon - delta_lon, lat + delta_lat],
            [lon - delta_lon, lat - delta_lat],
        ]]
    }


@dataclass
class OSMService:
    """Thin PostGIS helper that exposes higher level search primitives."""

    schema: str = os.environ.get("OSM_SCHEMA", "osm")
    host: str = os.environ.get("PGHOST", "127.0.0.1")
    port: int = int(os.environ.get("PGPORT", os.environ.get("PGHOST_PORT", "5432")))
    dbname: str = os.environ.get("PGDATABASE", "gis")
    user: str = os.environ.get("PGUSER", "gis")
    password: str = os.environ.get("PGPASSWORD", "gis")

    def __post_init__(self):
        self._conninfo = (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )

    def _query_one(self, sql: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with psycopg.connect(self._conninfo, row_factory=dict_row) as conn:
            with conn.execute(sql, params) as cur:
                return cur.fetchone()

    def _query_all(self, sql: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        with psycopg.connect(self._conninfo, row_factory=dict_row) as conn:
            with conn.execute(sql, params) as cur:
                return cur.fetchall()

    def resolve_named_area(self, value: str, location_type: str = "neighborhood") -> Optional[Dict[str, Any]]:
        if not value:
            return None

        pattern = f"%{value.strip()}%"
        exact = value.strip()
        type_clause = {
            "municipality": "AND (boundary='administrative' AND (admin_level IN ('6','7','8'))) ",
            "district": "AND (boundary='administrative' OR place IN ('district','borough','suburb','quarter')) ",
            "neighborhood": "AND (place IN ('neighbourhood','neighborhood','suburb','quarter','hamlet') OR boundary='administrative') ",
        }.get(location_type, "")

        sql = f"""
            SELECT
                osm_id,
                name,
                place,
                boundary,
                admin_level,
                ST_AsGeoJSON(ST_Transform(way, 4326), 6) AS geom_geojson,
                ST_AsGeoJSON(ST_Envelope(ST_Transform(way, 4326)), 6) AS bbox_geojson,
                ST_Y(ST_Transform(ST_Centroid(way), 4326)) AS centroid_lat,
                ST_X(ST_Transform(ST_Centroid(way), 4326)) AS centroid_lon
            FROM {self.schema}.planet_osm_polygon
            WHERE name ILIKE %(pattern)s
            {type_clause}
            ORDER BY CASE WHEN lower(name) = lower(%(exact)s) THEN 0 ELSE 1 END,
                     COALESCE(admin_level, '999') ASC,
                     ST_Area(way) DESC
            LIMIT 1;
        """
        row = self._query_one(sql, {"pattern": pattern, "exact": exact})
        if row:
            geom = _geometry_from_json(row.get("geom_geojson"))
            bbox = _geometry_from_json(row.get("bbox_geojson"))
            return {
                "name": row.get("name"),
                "type": location_type,
                "place": row.get("place"),
                "boundary": row.get("boundary"),
                "geometry": geom,
                "bbox": bbox,
                "centroid": {"lat": row.get("centroid_lat"), "lon": row.get("centroid_lon")},
            }

        point_sql = f"""
            SELECT
                osm_id,
                name,
                place,
                ST_AsGeoJSON(ST_Transform(way, 4326), 6) AS geom_geojson,
                ST_X(ST_Transform(way, 4326)) AS lon,
                ST_Y(ST_Transform(way, 4326)) AS lat
            FROM {self.schema}.planet_osm_point
            WHERE name ILIKE %(pattern)s
            ORDER BY CASE WHEN lower(name) = lower(%(exact)s) THEN 0 ELSE 1 END
            LIMIT 1;
        """
        row = self._query_one(point_sql, {"pattern": pattern, "exact": exact})
        if not row:
            return None
        geom = _geometry_from_json(row.get("geom_geojson"))
        bbox = None
        if row.get("lat") is not None and row.get("lon") is not None:
            bbox = _point_bbox(row["lat"], row["lon"], 3000.0)
        return {
            "name": row.get("name"),
            "type": location_type,
            "place": row.get("place"),
            "boundary": None,
            "geometry": geom,
            "bbox": bbox,
            "centroid": {"lat": row.get("lat"), "lon": row.get("lon")},
        }

    def resolve_address(self, address: str) -> Optional[Dict[str, Any]]:
        if not address:
            return None

        street, housenumber = self._split_address(address)
        pattern = f"%{street}%"
        sql = f"""
            SELECT
                osm_id,
                name,
                "addr:street" AS street,
                "addr:housenumber" AS housenumber,
                "addr:city" AS city,
                ST_AsGeoJSON(ST_Transform(way, 4326), 6) AS geom_geojson,
                ST_X(ST_Transform(way, 4326)) AS lon,
                ST_Y(ST_Transform(way, 4326)) AS lat
            FROM {self.schema}.planet_osm_point
            WHERE ("addr:street" ILIKE %(pattern)s OR %(pattern)s = '%%' OR name ILIKE %(pattern)s)
            ORDER BY
                CASE WHEN lower("addr:street") = lower(%(street)s) THEN 0 ELSE 1 END,
                CASE WHEN %(house)s IS NOT NULL AND lower("addr:housenumber") = lower(%(house)s) THEN 0 ELSE 1 END,
                osm_id
            LIMIT 1;
        """
        row = self._query_one(sql, {"pattern": pattern, "street": street, "house": housenumber})
        if not row:
            return None

        geom = _geometry_from_json(row.get("geom_geojson"))
        bbox = None
        if row.get("lat") is not None and row.get("lon") is not None:
            bbox = _point_bbox(row["lat"], row["lon"], 400.0)

        return {
            "name": row.get("name") or row.get("street"),
            "street": row.get("street"),
            "housenumber": row.get("housenumber"),
            "city": row.get("city"),
            "geometry": geom,
            "bbox": bbox,
            "centroid": {"lat": row.get("lat"), "lon": row.get("lon")},
        }

    def _amenity_terms(self, keyword: str) -> List[str]:
        base = keyword.strip().lower()
        synonyms = {
            "gym": ["gym", "fitness_centre", "fitness center", "fitness"],
            "fitness": ["gym", "fitness_centre", "fitness center", "fitness"],
            "school": ["school", "college"],
            "park": ["park", "recreation_ground", "playground"],
            "hospital": ["hospital", "clinic"],
        }
        for key, terms in synonyms.items():
            if base == key or base in terms:
                return list(dict.fromkeys(terms))
        return [base]

    def resolve_amenities(self, keyword: str, filter_geometry: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        if not keyword:
            return []
        params: Dict[str, Any] = {"limit": limit}
        term_clauses: List[str] = []
        for idx, term in enumerate(self._amenity_terms(keyword)):
            key = f"pattern_{idx}"
            params[key] = f"%{term}%"
            term_clauses.append(
                f"""(
                    (amenity IS NOT NULL AND amenity ILIKE %({key})s) OR
                    (leisure IS NOT NULL AND leisure ILIKE %({key})s) OR
                    (sport IS NOT NULL AND sport ILIKE %({key})s) OR
                    (name IS NOT NULL AND name ILIKE %({key})s)
                )"""
            )
        where_clause = " OR ".join(term_clauses)
        boundary_clause = ""
        if filter_geometry:
            params["boundary"] = json.dumps(filter_geometry)
            boundary_clause = """
                AND ST_Intersects(
                    way,
                    ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%(boundary)s), 4326), 3857)
                )
            """

        sql = f"""
            SELECT
                osm_id,
                name,
                amenity,
                leisure,
                sport,
                ST_AsGeoJSON(ST_Transform(way, 4326), 6) AS geom_geojson
            FROM {self.schema}.planet_osm_point
            WHERE ({where_clause})
            {boundary_clause}
            ORDER BY
                CASE WHEN amenity ILIKE %(pattern_0)s THEN 0 ELSE 1 END,
                CASE WHEN leisure ILIKE %(pattern_0)s THEN 0 ELSE 1 END,
                name NULLS LAST
            LIMIT %(limit)s;
        """.format(where_clause=where_clause, boundary_clause=boundary_clause)
        rows = self._query_all(sql, params)
        features: List[Dict[str, Any]] = []
        for row in rows:
            geom = _geometry_from_json(row.get("geom_geojson"))
            if not geom:
                continue
            features.append({
                "id": row.get("osm_id"),
                "name": row.get("name") or row.get("amenity") or row.get("leisure") or row.get("sport"),
                "amenity": row.get("amenity"),
                "leisure": row.get("leisure"),
                "sport": row.get("sport"),
                "geometry": geom,
            })
        return features

    def resolve_query(self, query: OSMQuerySpec) -> Dict[str, Any]:
        location_info = None
        address_info = None
        if query.location and query.location.value:
            location_info = self.resolve_named_area(query.location.value, query.location.type)
        if not location_info and query.address:
            address_info = self.resolve_address(query.address)

        filter_geometry = None
        if location_info:
            loc_geom = location_info.get("geometry")
            if loc_geom and loc_geom.get("type") in ("Polygon", "MultiPolygon"):
                filter_geometry = loc_geom
            elif location_info.get("bbox"):
                filter_geometry = location_info["bbox"]
        elif address_info:
            filter_geometry = address_info.get("bbox")

        amenities: List[Dict[str, Any]] = []
        if query.amenities:
            for spec in query.amenities:
                amenities.extend(self.resolve_amenities(spec.type, filter_geometry))

        return {
            "location": location_info,
            "address": address_info,
            "amenities": amenities,
            "filter_geometry": filter_geometry,
        }

    @staticmethod
    def _split_address(address: str) -> (str, Optional[str]):
        text = address.strip()
        match = re.search(r"(.+?)\s+(\d+\w?)$", text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return text, None
