#!/usr/bin/env python3
"""Generate the seeded synthetic enterprise dataset committed under data/.

The dataset models a mid-size electronics manufacturer ("Vertex Devices")
with a global, multi-tier supplier network. Everything is deterministic
(fixed RNG seed) so regeneration produces byte-identical files.

Usage:
    python scripts/generate_data.py
"""
from __future__ import annotations

import csv
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 90061
TODAY = date(2026, 6, 10)

rng = random.Random(SEED)

REGIONS = {
    "TW": ("Taiwan", "APAC"), "CN": ("China", "APAC"), "VN": ("Vietnam", "APAC"),
    "MY": ("Malaysia", "APAC"), "JP": ("Japan", "APAC"), "KR": ("South Korea", "APAC"),
    "DE": ("Germany", "EMEA"), "PL": ("Poland", "EMEA"), "CZ": ("Czechia", "EMEA"),
    "US": ("United States", "AMER"), "MX": ("Mexico", "AMER"), "BR": ("Brazil", "AMER"),
    "IN": ("India", "APAC"), "TH": ("Thailand", "APAC"),
}

SUPPLIER_NAMES = [
    "Hsinchu Microfab", "Shenzhen BrightCircuit", "Hanoi Polymer Works", "Penang Optics",
    "Osaka Precision Components", "Busan Cell Systems", "Dresden Wafertech", "Krakow Connectors",
    "Brno Enclosures", "Austin Boardworks", "Monterrey Harness Co", "Curitiba Castings",
    "Chennai Passives", "Bangkok Display Group", "Taoyuan Sensor Labs", "Suzhou MagnetCore",
    "Da Nang Battery Pack", "Kulim Substrates", "Nagoya Crystal", "Gumi PowerCells",
    "Leipzig Coatings", "Gdansk PCB House", "Ostrava Metals", "Phoenix Silicon Foundry",
    "Tijuana Assembly Partners", "Sao Paulo Resins", "Pune Relays", "Rayong Mouldings",
    "Kaohsiung Camera Modules", "Dongguan Speakers", "Hai Phong Antennas", "Johor Flex Circuits",
    "Kyoto MEMS Works", "Daejeon RF Labs", "Stuttgart Fasteners", "Wroclaw Cables",
    "Plzen Gaskets", "Raleigh Firmware Modules", "Guadalajara Touch Panels", "Recife Adhesives",
]

PART_CATEGORIES = [
    ("SOC", "Application processor", (12.0, 48.0), True),
    ("MEM", "LPDDR5 memory", (6.0, 22.0), True),
    ("PMIC", "Power management IC", (1.2, 6.5), True),
    ("DSP", "Display panel", (14.0, 60.0), True),
    ("BAT", "Li-ion battery pack", (4.0, 18.0), True),
    ("CAM", "Camera module", (5.0, 35.0), False),
    ("RF", "RF front-end module", (2.0, 9.0), True),
    ("CON", "Board-to-board connector", (0.2, 1.8), False),
    ("PCB", "Main logic board PCB", (3.0, 12.0), False),
    ("ENC", "Machined enclosure", (2.5, 14.0), False),
    ("SEN", "MEMS sensor", (0.8, 4.5), False),
    ("ANT", "Antenna assembly", (0.4, 2.2), False),
    ("SPK", "Speaker assembly", (0.6, 3.0), False),
    ("FLX", "Flex circuit", (0.5, 2.5), False),
    ("PAS", "Passive component kit", (0.1, 0.9), False),
]

PRODUCTS = [
    ("PRD-100", "Vertex Tab 11", 329.0, 18000),
    ("PRD-110", "Vertex Tab 11 Pro", 549.0, 9000),
    ("PRD-200", "Vertex Phone S", 449.0, 42000),
    ("PRD-210", "Vertex Phone S Ultra", 899.0, 16000),
    ("PRD-300", "Vertex Watch 4", 249.0, 22000),
    ("PRD-310", "Vertex Watch 4 LTE", 329.0, 11000),
    ("PRD-400", "Vertex Buds ANC", 129.0, 55000),
    ("PRD-500", "Vertex Home Hub", 179.0, 13000),
    ("PRD-600", "Vertex Edge Gateway", 1299.0, 2400),
    ("PRD-700", "Vertex Fleet Tracker", 199.0, 8000),
    ("PRD-800", "Vertex Med Monitor", 1599.0, 1500),
    ("PRD-900", "Vertex Retail Kiosk", 2499.0, 900),
]

PORTS = ["Kaohsiung", "Shanghai", "Hai Phong", "Port Klang", "Busan", "Hamburg",
         "Gdansk", "Long Beach", "Manzanillo", "Santos", "Chennai", "Laem Chabang"]


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def main() -> None:
    country_codes = list(REGIONS)

    # --- suppliers -----------------------------------------------------------
    suppliers = []
    for i, name in enumerate(SUPPLIER_NAMES, start=1):
        cc = country_codes[i % len(country_codes)]
        country, region = REGIONS[cc]
        suppliers.append([
            f"SUP-{i:03d}", name, country, region,
            rng.choice(PORTS),
            rng.choice(["strategic", "preferred", "approved"]),
            round(rng.uniform(0.82, 0.995), 3),          # on-time delivery rate
            rng.randint(14, 75),                          # avg lead time days
            round(rng.uniform(0.5, 9.5), 1),              # annual spend musd
            rng.choice(["low", "low", "medium", "medium", "high"]),  # geo risk
            "yes" if rng.random() > 0.12 else "no",       # iso9001 certified
        ])
    write_csv(DATA / "suppliers.csv",
              ["supplier_id", "name", "country", "region", "primary_port", "tier_class",
               "on_time_rate", "avg_lead_time_days", "annual_spend_musd", "geo_risk", "iso9001"],
              suppliers)

    # --- parts (each with primary + optional alternate supplier) -------------
    parts = []
    pid = 0
    for cat, desc, (lo, hi), critical in PART_CATEGORIES:
        for variant in range(1, 9):
            pid += 1
            primary = suppliers[rng.randrange(len(suppliers))][0]
            alt = suppliers[rng.randrange(len(suppliers))][0]
            if alt == primary or rng.random() < 0.28:
                alt = ""  # single-sourced — the interesting failure mode
            parts.append([
                f"PART-{pid:04d}", f"{cat}-{variant:02d}", f"{desc} v{variant}",
                cat, primary, alt,
                round(rng.uniform(lo, hi), 2),
                "yes" if critical else "no",
                rng.randint(10, 60),                      # replenishment days
            ])
    write_csv(DATA / "parts.csv",
              ["part_id", "part_number", "description", "category", "primary_supplier_id",
               "alternate_supplier_id", "unit_cost_usd", "critical", "replenish_days"],
              parts)

    # --- bill of materials ----------------------------------------------------
    by_cat: dict[str, list[str]] = {}
    for p in parts:
        by_cat.setdefault(p[3], []).append(p[0])
    bom = []
    for prod_id, _, _, _ in PRODUCTS:
        n_cats = rng.randint(6, 11)
        cats = rng.sample(list(by_cat), n_cats)
        for cat in cats:
            part_id = rng.choice(by_cat[cat])
            bom.append([prod_id, part_id, rng.choice([1, 1, 1, 2, 2, 4])])
    write_csv(DATA / "bom.csv", ["product_id", "part_id", "qty_per_unit"], bom)

    # --- products --------------------------------------------------------------
    write_csv(DATA / "products.csv",
              ["product_id", "name", "unit_price_usd", "monthly_demand_units"],
              [list(p) for p in PRODUCTS])

    # --- inventory --------------------------------------------------------------
    inventory = []
    for p in parts:
        daily_burn = rng.randint(40, 2200)
        days_cover = rng.randint(6, 55)
        inventory.append([p[0], daily_burn * days_cover, daily_burn,
                          rng.choice(["AUS1", "MEX1", "POL1", "VNM1"])])
    write_csv(DATA / "inventory.csv",
              ["part_id", "on_hand_units", "daily_consumption_units", "warehouse"],
              inventory)

    # --- in-transit shipments ----------------------------------------------------
    shipments = []
    for i in range(1, 221):
        part = parts[rng.randrange(len(parts))]
        sup_id = part[4]
        sup = next(s for s in suppliers if s[0] == sup_id)
        etd = TODAY - timedelta(days=rng.randint(1, 30))
        eta = TODAY + timedelta(days=rng.randint(2, 45))
        shipments.append([
            f"SHP-{i:05d}", part[0], sup_id, sup[4],
            rng.choice(["ocean", "ocean", "ocean", "air"]),
            etd.isoformat(), eta.isoformat(),
            rng.randint(500, 50000),
            rng.choice(["in_transit", "in_transit", "in_transit", "customs_hold"]),
        ])
    write_csv(DATA / "shipments.csv",
              ["shipment_id", "part_id", "supplier_id", "origin_port", "mode",
               "etd", "eta", "units", "status"],
              shipments)

    # --- open purchase orders ------------------------------------------------------
    pos = []
    for i in range(1, 161):
        part = parts[rng.randrange(len(parts))]
        units = rng.randint(1000, 80000)
        pos.append([
            f"PO-{i:05d}", part[0], part[4], units,
            round(units * float(part[6]), 2),
            (TODAY + timedelta(days=rng.randint(5, 90))).isoformat(),
            rng.choice(["open", "open", "open", "partially_received"]),
        ])
    write_csv(DATA / "purchase_orders.csv",
              ["po_id", "part_id", "supplier_id", "units", "value_usd", "due_date", "status"],
              pos)

    # --- disruption signal feed (the pipeline's inbox) ------------------------------
    base = datetime(2026, 6, 9, 4, 30)
    events = [
        {
            "event_id": "EVT-2026-0601", "source": "weather_service",
            "type": "typhoon", "severity_hint": "critical",
            "headline": "Super Typhoon Halong forecast to make landfall near Kaohsiung within 48h",
            "body": "Category 5-equivalent storm tracking toward southern Taiwan. Kaohsiung port "
                    "authority announces preemptive closure for at least 5 days. Industrial parks in "
                    "Kaohsiung and Tainan beginning controlled shutdowns.",
            "locations": ["Taiwan", "Kaohsiung"], "suppliers_mentioned": [],
        },
        {
            "event_id": "EVT-2026-0602", "source": "supplier_portal",
            "type": "factory_fire", "severity_hint": "high",
            "headline": "Fire reported at Gumi PowerCells cell line 2",
            "body": "Supplier declared force majeure. Line 2 (≈40% of site capacity) offline, "
                    "preliminary restoration estimate 6-8 weeks. Lines 1 and 3 unaffected.",
            "locations": ["South Korea", "Gumi"], "suppliers_mentioned": ["Gumi PowerCells"],
        },
        {
            "event_id": "EVT-2026-0603", "source": "logistics_provider",
            "type": "port_congestion", "severity_hint": "medium",
            "headline": "Shanghai berth wait times rise to 9 days",
            "body": "Container backlog after week-long fog closures. Carriers quoting 7-10 day "
                    "delays on trans-pacific lanes departing Shanghai through end of month.",
            "locations": ["China", "Shanghai"], "suppliers_mentioned": [],
        },
        {
            "event_id": "EVT-2026-0604", "source": "news_wire",
            "type": "export_control", "severity_hint": "high",
            "headline": "New export licensing requirement announced for advanced RF components",
            "body": "Effective in 30 days, RF front-end modules above 6GHz require export licenses "
                    "for several destination markets. Industry groups expect 3-5 week licensing lead "
                    "times during the initial rollout.",
            "locations": ["South Korea", "Japan"], "suppliers_mentioned": ["Daejeon RF Labs"],
        },
        {
            "event_id": "EVT-2026-0605", "source": "supplier_portal",
            "type": "quality_recall", "severity_hint": "medium",
            "headline": "Krakow Connectors issues containment notice on lot K-2261",
            "body": "Plating defect detected in board-to-board connector lot shipped May 18-29. "
                    "Supplier requests quarantine of affected lot pending sort instructions.",
            "locations": ["Poland", "Krakow"], "suppliers_mentioned": ["Krakow Connectors"],
        },
        {
            "event_id": "EVT-2026-0606", "source": "news_wire",
            "type": "labor_strike", "severity_hint": "medium",
            "headline": "Dockworkers union announces 72-hour strike at Gdansk",
            "body": "Strike begins Monday. Feeder vessels being diverted to Hamburg with 4-6 day "
                    "rebooking delays expected.",
            "locations": ["Poland", "Gdansk"], "suppliers_mentioned": [],
        },
        {
            "event_id": "EVT-2026-0607", "source": "financial_monitor",
            "type": "supplier_insolvency", "severity_hint": "high",
            "headline": "Curitiba Castings enters judicial restructuring",
            "body": "Tier-1 castings supplier filed for court-supervised restructuring. Production "
                    "continuing for now; credit insurers have withdrawn coverage. 60-90 day supply "
                    "continuity risk flagged by financial monitor.",
            "locations": ["Brazil", "Curitiba"], "suppliers_mentioned": ["Curitiba Castings"],
        },
        {
            "event_id": "EVT-2026-0608", "source": "weather_service",
            "type": "flood", "severity_hint": "low",
            "headline": "Monsoon flooding in Chennai industrial corridor",
            "body": "Localized flooding, most plants operating normally. Chennai Passives reports "
                    "no impact but road freight to port slowed by 1-2 days.",
            "locations": ["India", "Chennai"], "suppliers_mentioned": ["Chennai Passives"],
        },
        {
            "event_id": "EVT-2026-0609", "source": "logistics_provider",
            "type": "customs_delay", "severity_hint": "low",
            "headline": "New customs inspection regime at Manzanillo adds 24-48h",
            "body": "Random inspection rate increased. Brokers advise adding 2 days of buffer for "
                    "northbound clearance.",
            "locations": ["Mexico", "Manzanillo"], "suppliers_mentioned": [],
        },
        {
            "event_id": "EVT-2026-0610", "source": "news_wire",
            "type": "cyber_incident", "severity_hint": "high",
            "headline": "Ransomware attack disrupts Penang Optics ERP systems",
            "body": "Supplier confirms ransomware incident. Production lines running but order "
                    "processing and outbound logistics manual; shipping confirmations delayed "
                    "indefinitely. No customer data exposure reported.",
            "locations": ["Malaysia", "Penang"], "suppliers_mentioned": ["Penang Optics"],
        },
        {
            "event_id": "EVT-2026-0611", "source": "commodity_desk",
            "type": "price_shock", "severity_hint": "medium",
            "headline": "Cobalt spot price up 31% month-over-month",
            "body": "Battery-grade cobalt rally driven by mine output cuts. Cell suppliers signaling "
                    "8-12% pack price increases at next quarterly negotiation.",
            "locations": [], "suppliers_mentioned": [],
        },
        {
            "event_id": "EVT-2026-0612", "source": "weather_service",
            "type": "earthquake", "severity_hint": "critical",
            "headline": "M6.9 earthquake near Hsinchu Science Park",
            "body": "Strong quake 40km from Hsinchu. Fabs executing post-seismic inspection "
                    "protocols; historically 5-14 day requalification on advanced nodes. Aftershock "
                    "risk elevated for 72h. Hsinchu Microfab and Taoyuan Sensor Labs in affected zone.",
            "locations": ["Taiwan", "Hsinchu"],
            "suppliers_mentioned": ["Hsinchu Microfab", "Taoyuan Sensor Labs"],
        },
    ]
    feed_path = DATA / "disruption_feed.jsonl"
    with feed_path.open("w") as f:
        for i, ev in enumerate(events):
            ev["observed_at"] = (base + timedelta(minutes=37 * i)).isoformat() + "Z"
            f.write(json.dumps(ev, sort_keys=True) + "\n")
    print(f"wrote {feed_path.relative_to(ROOT)} ({len(events)} events)")


if __name__ == "__main__":
    main()
