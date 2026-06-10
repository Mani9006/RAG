# Research Foundation

Why this problem, why agents, and why most attempts at this fail. This document
grounds Sentinel SCM's design decisions in the published evidence.

## 1. The problem is enormous and getting worse

- Organizations lose an average of **$184M per year** to global supply chain
  disruptions (Interos Annual Global Supply Chain Report); procurement-tied
  disruptions alone average **$16M/year** per company (Coupa).
- The Economist Intelligence Unit puts recurring disruption costs at **6–10% of
  annual revenue**; McKinsey estimates disruptions consume **~45% of one year's
  profits per decade**.
- 44.5% of organizations lost 3–4% of annual revenue to disruption over the
  last three years; automotive alone absorbs ~$13B/year.
- The supply chain risk management (SCRM) software market is growing at
  15–21% CAGR, projected from ~$4.5B (2025) toward $9–56B depending on scope
  (Mordor Intelligence, Market Research Future) — money follows pain.

Sources: [Interos via Supply Chain 24/7](https://www.supplychain247.com/article/procurement-disruptions-cost-16-million-supply-chain-coupa-report),
[EIU](https://insights.economistenterprise.com/projects/next-gen-supply-chains/reports/the-business-costs-of-supply-chain-disruption/),
[Conexiom statistics roundup](https://conexiom.com/blog/the-cost-of-supply-chain-disruptions-20-statistics/),
[Mordor Intelligence](https://www.mordorintelligence.com/industry-reports/supply-chain-risk-management-market),
[Market Research Future](https://www.marketresearchfuture.com/reports/supply-chain-risk-management-software-market-26455).

## 2. What incumbents do — and where the gap is

| Vendor | Strength | Gap Sentinel targets |
|---|---|---|
| **Resilinc** (EventWatchAI) | Sub-tier mapping, AI event monitoring | Monitoring → *alerts*. A human still does triage → impact → plan → approval. |
| **Interos** ($1B valuation) | Real-time network mapping, predictive risk scores | Scores describe risk; they don't *plan and execute* governed responses. |
| **Everstream** (Discover/Explore/Reveal) | Live disruption feeds, probabilistic impact | Same: the "last mile" from insight to a compliance-checked, costed action is manual. |
| SAP / IBM / Oracle | Risk dashboards embedded in planning suites | Dashboards, not agents; weeks-long change processes to act on a signal. |

**The whitespace:** every incumbent stops at *detection and scoring*. The
expensive part — triage, blast-radius quantification against the company's own
BOM, costed mitigation planning, compliance checking, and approval routing —
remains human-hours. That is precisely the part a governed agentic pipeline
automates. Sentinel's thesis: **close the loop from signal to approved action,
with code-enforced governance so the loop can be trusted.**

Sources: [Z2Data tool comparison](https://www.z2data.com/insights/top-7-supply-chain-risk-management-software-tools-for-2026),
[Spherical Insights top-20 vendor analysis](https://www.sphericalinsights.com/blogs/top-20-companies-in-global-supply-chain-risk-management-software-market-2026-2035-competitive-analysis-forecast),
[Everstream](https://www.everstream.ai/).

## 3. Why companies take years (and why a demo takes an afternoon)

The honest answer to "why do companies spend years on this?":

1. **Data readiness is the killer, not models.** Gartner: 85% of AI projects
   fail on data quality; 60% of projects lacking AI-ready data will be
   abandoned through 2026. Enterprise supplier/BOM/inventory data lives in
   5–15 systems (ERP, MES, TMS, supplier portals) with conflicting IDs and
   no single source of truth. Sentinel sidesteps this with a clean synthetic
   system of record — a real deployment spends 60–80% of its calendar time here.
2. **The pilot-to-production chasm.** RAND (2025): 80.3% of AI projects fail to
   deliver intended value; MIT NANDA: 95% of generative-AI pilots show zero
   measurable P&L return; enterprises abandoning most AI initiatives jumped
   from 17% (2024) to 42% (2025). The causes are organizational: no end-user
   co-design, no governance story, no audit trail, no answer to "what happens
   when the model is wrong about $2M of spend?"
3. **Trust and governance.** A system that can propose POs needs separation of
   duties, spend authority enforcement, restricted-entity screening, and an
   immutable audit trail before any CFO signs off. Most pilots bolt this on
   late; Sentinel builds the approval gate *into the orchestrator* from day one.
4. **Integration and change management.** Approvals must land where people
   work (Slack/ServiceNow), actions must write back to ERPs, and category
   managers must trust the numbers — each a months-long workstream.

Sources: [RAND via Pertama Partners](https://www.pertamapartners.com/insights/ai-project-failure-statistics-2026),
[MIT NANDA via Fortune](https://fortune.com/2025/08/18/mit-report-95-percent-generative-ai-pilots-at-companies-failing-cfo/),
[SR Analytics on Gartner data findings](https://sranalytics.io/blog/why-95-of-ai-projects-fail/).

**Design consequences for Sentinel** (each maps to a roadmap phase):

| Failure mode in the wild | Sentinel countermeasure |
|---|---|
| Bad/missing data kills the project | Clean repo-resident system of record now; connector seams for real systems (Phase 2+) |
| No measurable value | Every incident carries revenue-at-risk and action cost — value is a query away |
| No governance → no production | Code-enforced approval gate + audit log since Phase 1 |
| Model errors are unbounded | Tool-grounded numbers, budgets, iteration caps, eval harness (Phase 4) |
| Alerts nobody acts on | The pipeline's *output* is an approval queue, not a dashboard |

## 4. Free real-time intelligence sources (Phase 2)

Commercial feeds (Resilinc EventWatch, Everstream) cost six figures. The same
public primary sources they aggregate are free:

| Source | What | API |
|---|---|---|
| **USGS** | Global earthquakes, real-time GeoJSON | `earthquake.usgs.gov/fdsnws/event/1/query` and rolling feeds — [docs](https://earthquake.usgs.gov/fdsnws/event/1/) |
| **NOAA / NWS** | US severe-weather alerts (typhoon/hurricane, flood) | `api.weather.gov/alerts/active` |
| **GDELT 2.0** | Global news events: strikes, fires, sanctions, unrest | `api.gdeltproject.org/api/v2/doc/doc` |

Precedent that this exact stack works: [AUGOR](https://augor.ai/) commercializes
GDELT + NOAA + USGS (+ FIRMS/OFAC/IODA) feeds into scored supply-chain alerts.
Sentinel's Phase 2 implements the same primary-source strategy as open code:
each connector normalizes raw events into the signal schema, geo-matches them
against the company's actual supplier/port footprint, and dedupes into the
signal inbox — so the agentic pipeline downstream is identical for replayed
fixtures and live-world events.

## 5. Differentiation summary

1. **Closed loop, not alerts** — signal → triage → quantified impact →
   costed plan → compliance verdict → approval queue → briefing.
2. **Governance as code** — the gate is in the orchestrator, not in a prompt.
3. **Tool-grounded numbers** — every figure traceable to a query against the
   system of record; auditable end to end.
4. **Network science, not just lists** (Phase 2) — single-point-of-failure
   detection and outage propagation simulation over the supplier/BOM graph.
5. **Zero-cost adoption path** — simulation backend means anyone can run the
   full system before spending a dollar on inference, attacking the #1 cause
   of pilot abandonment (no demonstrated value).
