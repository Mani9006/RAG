"""NOAA / National Weather Service alerts connector.

Feed: https://api.weather.gov/alerts/active (free, no key). US coverage —
relevant to the AMER leg of the network (US/MX suppliers, Long Beach port).
Only operationally meaningful event classes pass the filter; marine chatter
and minor advisories are dropped before they ever reach the triage agent.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from sentinel.connectors.base import Footprint, SignalRecord, http_get_json

API_URL = "https://api.weather.gov/alerts/active"

# NWS event name fragment -> our signal type
EVENT_TYPES = {
    "hurricane": "typhoon",
    "typhoon": "typhoon",
    "tropical storm": "typhoon",
    "tornado": "severe_weather",
    "flood": "flood",
    "fire": "wildfire",
    "winter storm": "severe_weather",
    "blizzard": "severe_weather",
    "ice storm": "severe_weather",
    "high wind": "severe_weather",
}
SEVERITY_MAP = {"Extreme": "high", "Severe": "medium"}
RADIUS_KM = 300.0


def _classify(event_name: str) -> str | None:
    lowered = event_name.lower()
    for fragment, signal_type in EVENT_TYPES.items():
        if fragment in lowered:
            return signal_type
    return None


def _centroid(geometry: dict | None) -> tuple[float, float] | None:
    if not geometry or not geometry.get("coordinates"):
        return None
    coords = geometry["coordinates"]
    # Polygon: [[[lon, lat], ...]]; MultiPolygon nests one level deeper.
    while isinstance(coords[0][0], list):
        coords = coords[0]
    lons = [point[0] for point in coords]
    lats = [point[1] for point in coords]
    return sum(lats) / len(lats), sum(lons) / len(lons)


class NoaaConnector:
    name = "noaa"

    def fetch(self, window_hours: int = 24) -> dict:  # noqa: ARG002 (API returns active only)
        return http_get_json(API_URL, params={
            "status": "actual",
            "severity": "Extreme,Severe",
            "message_type": "alert",
        })

    def normalize(self, payload: dict, footprint: Footprint) -> list[SignalRecord]:
        records = []
        for feature in payload.get("features", []):
            props = feature.get("properties", {})
            signal_type = _classify(props.get("event", ""))
            if signal_type is None:
                continue
            severity = SEVERITY_MAP.get(props.get("severity", ""))
            if severity is None:
                continue

            locations = ["United States"]
            suppliers_hit: list[str] = []
            centroid = _centroid(feature.get("geometry"))
            if centroid:
                nearby = footprint.near(centroid[0], centroid[1], RADIUS_KM)
                suppliers_hit = [s.name for s, _ in nearby if s.kind == "supplier"][:5]
                locations += sorted({s.country for s, _ in nearby} - {"United States"})
                if not nearby:
                    severity = "low"  # real alert, but outside our footprint

            alert_id = props.get("id") or feature.get("id", "")
            digest = hashlib.sha1(alert_id.encode()).hexdigest()[:12]
            observed = props.get("sent") or datetime.now(timezone.utc).isoformat()
            area = props.get("areaDesc", "")
            records.append(SignalRecord(
                event_id=f"NOAA-{digest}",
                observed_at=observed,
                source="noaa_live",
                type=signal_type,
                severity_hint=severity,
                headline=props.get("headline") or f"{props.get('event')} — {area[:80]}",
                body=(props.get("description") or "")[:1500]
                + (f"\n\nAffected areas: {area}" if area else ""),
                locations=locations,
                suppliers_mentioned=suppliers_hit,
            ))
        return records
