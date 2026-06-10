"""GDELT 2.0 news-event connector.

Feed: https://api.gdeltproject.org/api/v2/doc/doc (free, no key). GDELT
monitors global news in 100+ languages; we query a supply-chain disruption
vocabulary and classify each article by keyword. Articles that name one of
our actual suppliers are escalated — that is the highest-signal match a news
feed can produce.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from sentinel.connectors.base import Footprint, SignalRecord, http_get_json

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

QUERY = (
    '("supply chain" OR factory OR port OR semiconductor) '
    "(strike OR fire OR ransomware OR sanctions OR bankruptcy OR shortage "
    "OR \"force majeure\" OR recall OR congestion)"
)

# keyword -> signal type, first match wins (ordered by specificity)
KEYWORD_TYPES = [
    ("ransomware", "cyber_incident"), ("cyberattack", "cyber_incident"),
    ("force majeure", "factory_fire"), ("factory fire", "factory_fire"),
    ("fire", "factory_fire"),
    ("strike", "labor_strike"), ("walkout", "labor_strike"),
    ("sanction", "export_control"), ("export control", "export_control"),
    ("export ban", "export_control"),
    ("bankrupt", "supplier_insolvency"), ("insolven", "supplier_insolvency"),
    ("restructuring", "supplier_insolvency"),
    ("recall", "quality_recall"),
    ("congestion", "port_congestion"), ("port closure", "port_congestion"),
    ("backlog", "port_congestion"),
    ("earthquake", "earthquake"), ("typhoon", "typhoon"), ("hurricane", "typhoon"),
    ("flood", "flood"), ("shortage", "price_shock"),
]


def _classify(title: str) -> str | None:
    lowered = title.lower()
    for keyword, signal_type in KEYWORD_TYPES:
        if keyword in lowered:
            return signal_type
    return None


def _parse_seendate(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc).isoformat()
    except (ValueError, TypeError):
        return datetime.now(timezone.utc).isoformat()


class GdeltConnector:
    name = "gdelt"

    def fetch(self, window_hours: int = 24) -> dict:
        return http_get_json(API_URL, params={
            "query": QUERY,
            "mode": "artlist",
            "format": "json",
            "maxrecords": 75,
            "timespan": f"{window_hours}h",
            "sort": "datedesc",
        })

    def normalize(self, payload: dict, footprint: Footprint) -> list[SignalRecord]:
        supplier_names = footprint.supplier_names()
        records = []
        for article in payload.get("articles", []):
            title = article.get("title", "")
            if not title:
                continue
            signal_type = _classify(title)
            if signal_type is None:
                continue  # not a disruption story

            mentioned = [name for name in supplier_names if name.lower() in title.lower()]
            severity = "high" if mentioned else "medium"

            url = article.get("url", title)
            digest = hashlib.sha1(url.encode()).hexdigest()[:12]
            country = article.get("sourcecountry", "")
            records.append(SignalRecord(
                event_id=f"GDELT-{digest}",
                observed_at=_parse_seendate(article.get("seendate", "")),
                source="gdelt_live",
                type=signal_type,
                severity_hint=severity,
                headline=title[:200],
                body=(
                    f"News-monitored event via GDELT. Source: {article.get('domain', '?')}"
                    f"{f' ({country})' if country else ''}. URL: {url}"
                ),
                locations=[country] if country else [],
                suppliers_mentioned=mentioned,
            ))
        return records
