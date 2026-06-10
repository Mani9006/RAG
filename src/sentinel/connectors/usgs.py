"""USGS earthquake connector.

Feed: https://earthquake.usgs.gov/fdsnws/event/1/query (GeoJSON, free, no key).
Relevance: an earthquake matters to us only if it is strong enough AND lands
near our footprint. Severity blends magnitude with proximity to actual
supplier sites / origin ports.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sentinel.connectors.base import Footprint, SignalRecord, http_get_json

API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
MIN_MAGNITUDE = 5.5
RADIUS_KM = 500.0


class UsgsConnector:
    name = "usgs"

    def fetch(self, window_hours: int = 24) -> dict:
        start = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        return http_get_json(API_URL, params={
            "format": "geojson",
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude": MIN_MAGNITUDE,
            "orderby": "time",
        })

    def normalize(self, payload: dict, footprint: Footprint) -> list[SignalRecord]:
        records = []
        for feature in payload.get("features", []):
            props = feature.get("properties", {})
            geometry = feature.get("geometry") or {}
            coords = geometry.get("coordinates") or [None, None]
            lon, lat = coords[0], coords[1]
            mag = props.get("mag")
            if mag is None or lat is None or lon is None:
                continue

            nearby = footprint.near(lat, lon, RADIUS_KM)
            if not nearby:
                continue  # strong quake, but nowhere near our network

            nearest_site, nearest_km = nearby[0]
            suppliers_hit = [s.name for s, d in nearby if s.kind == "supplier"][:5]
            countries = sorted({s.country for s, _ in nearby})

            if mag >= 7.0 or (mag >= 6.5 and nearest_km <= 150):
                severity = "critical"
            elif mag >= 6.3 or (mag >= 6.0 and nearest_km <= 200):
                severity = "high"
            elif mag >= 6.0:
                severity = "medium"
            else:
                severity = "low"

            observed = datetime.fromtimestamp(
                props.get("time", 0) / 1000, tz=timezone.utc
            ).isoformat()
            records.append(SignalRecord(
                event_id=f"USGS-{feature.get('id')}",
                observed_at=observed,
                source="usgs_live",
                type="earthquake",
                severity_hint=severity,
                headline=f"M{mag:.1f} earthquake {props.get('place', 'unknown location')}",
                body=(
                    f"USGS reports a magnitude {mag:.1f} earthquake "
                    f"{props.get('place', '')}. Nearest network site: {nearest_site.name} "
                    f"({nearest_site.kind}, {nearest_km} km from epicenter). "
                    f"{len(nearby)} network site(s) within {int(RADIUS_KM)} km. "
                    f"Details: {props.get('url', '')}"
                ),
                locations=countries,
                suppliers_mentioned=suppliers_hit,
            ))
        return records
