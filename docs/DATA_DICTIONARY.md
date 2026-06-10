# Data Dictionary

All datasets are synthetic, deterministic (seeded RNG), and committed under
`data/`. They model "Vertex Devices", a mid-size electronics manufacturer with
a global multi-tier supplier network. Regenerate with
`python scripts/generate_data.py` — output is byte-identical.

## suppliers.csv (40 rows)
| Column | Meaning |
|---|---|
| supplier_id | `SUP-NNN` primary key |
| name, country, region | identity; country derived from the plant city; region ∈ APAC/EMEA/AMER |
| city, lat, lon | plant location — used by live-signal geo-matching and graph analytics |
| primary_port | export port used by the supplier |
| tier_class | strategic / preferred / approved |
| on_time_rate | trailing on-time delivery rate (0–1) |
| avg_lead_time_days | standard replenishment lead time |
| annual_spend_musd | our annual spend with them |
| geo_risk | low / medium / high country-risk rating |
| iso9001 | quality certification flag (drives policy SR-004) |

## parts.csv (120 rows)
| Column | Meaning |
|---|---|
| part_id / part_number | `PART-NNNN` / human SKU |
| category | SOC, MEM, PMIC, DSP, BAT, CAM, RF, CON, PCB, ENC, SEN, ANT, SPK, FLX, PAS |
| primary_supplier_id | current source |
| alternate_supplier_id | qualified alternate — **empty = single-sourced** (the key risk dimension) |
| unit_cost_usd | standard cost |
| critical | critical-category flag (drives policy SR-001/SR-004) |
| replenish_days | standard replenishment time |

## bom.csv (95 rows)
`product_id, part_id, qty_per_unit` — which parts each product consumes.

## products.csv (12 rows)
`product_id, name, unit_price_usd, monthly_demand_units` — the revenue side of
blast-radius math.

## inventory.csv (120 rows)
`part_id, on_hand_units, daily_consumption_units, warehouse` — days of cover =
on_hand / daily consumption.

## shipments.csv (220 rows)
In-transit freight: `shipment_id, part_id, supplier_id, origin_port, mode,
etd, eta, units, status (in_transit | customs_hold)`.

## purchase_orders.csv (160 rows)
Open commitments: `po_id, part_id, supplier_id, units, value_usd, due_date,
status (open | partially_received)`.

## disruption_feed.jsonl (12 events)
The pipeline's inbox. Each line:
`event_id, observed_at, source, type, severity_hint, headline, body,
locations[], suppliers_mentioned[]`.

Event types span the realistic threat catalogue: typhoon, earthquake,
factory fire, cyber incident, supplier insolvency, export control,
labor strike, port congestion, quality recall, price shock, customs delay,
flood. Severity hints are *source* hints — the triage agent re-scores against
actual network exposure (e.g. a "critical" event with zero footprint overlap
is downgraded).

## geo/locations.csv (12 rows)
Port gazetteer (`kind, name, country, lat, lon`) used together with supplier
coordinates as the geo-matching footprint for live connectors (USGS / NOAA /
GDELT). Tests use committed connector fixtures under `tests/fixtures/`.

## scenarios/*.yaml (4 scenarios)
War-gamed what-if specs for the Monte Carlo digital twin. Each declares a
supplier selection (`suppliers` / `countries` / `ports` — union applies) and
triangular distributions for `outage_days`, `alt_lead_multiplier`, and
`demand_multiplier`. Run via `sentinel simulate --scenario <name>`.

## evals/*.jsonl (golden datasets)
Labeled cases for the agent evaluation harness: `triage_golden.jsonl`
(escalation ground truth, severity bands, score calibration, entity
resolution) and `compliance_golden.jsonl` (expected verdicts + required rule
citations). Supplier/part references use `@selector` placeholders
(`@name:…`, `@iso9001:yes`, `@cover_lt:18`) resolved against the live dataset
at eval time. Run via `sentinel eval`.

## policies/procurement_policy.yaml
Governance consumed by the compliance agent and the approval gate:
spend-authority tiers, sourcing rules SR-001…SR-004, restricted entities,
escalation SLAs.

## Derived store (.sentinel/sentinel.db — not committed)
Runtime tables: `signals` (inbox + status), `incidents`, `actions`
(approval lifecycle), `audit_log`.
