"""Supply network science.

Builds the supplier → part → product dependency graph from the system of
record and computes the analytics that turn lists into insight:

- single-point-of-failure detection (critical, single-sourced parts)
- a composite supplier criticality index
- deterministic outage propagation: simulate a supplier going dark for N days
  and compute per-part stockout timelines and revenue loss bounded by product
  (no double counting when several affected parts feed the same product).

All math is deterministic and documented so results are auditable and the
agents can cite them.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

_GEO_RISK_WEIGHT = {"low": 0.2, "medium": 0.5, "high": 0.9}

# Criticality index weights — revenue dependence dominates by design.
_W_REVENUE, _W_SINGLE_SOURCE, _W_GEO, _W_OTD = 0.5, 0.25, 0.15, 0.10


@dataclass
class PartNode:
    part_id: str
    category: str
    critical: bool
    single_sourced: bool
    primary_supplier_id: str
    alternate_supplier_id: str
    unit_cost_usd: float
    on_hand_units: int = 0
    daily_consumption_units: int = 0
    inbound_units: int = 0
    products: list[tuple[str, int]] = field(default_factory=list)  # (product_id, qty_per_unit)


class SupplyGraph:
    def __init__(self, conn: sqlite3.Connection):
        self.suppliers = {
            r["supplier_id"]: dict(r) for r in conn.execute("SELECT * FROM suppliers")
        }
        self.products = {
            r["product_id"]: dict(r) for r in conn.execute("SELECT * FROM products")
        }
        self.parts: dict[str, PartNode] = {}
        for r in conn.execute("SELECT * FROM parts"):
            self.parts[r["part_id"]] = PartNode(
                part_id=r["part_id"], category=r["category"],
                critical=r["critical"] == "yes",
                single_sourced=r["alternate_supplier_id"] == "",
                primary_supplier_id=r["primary_supplier_id"],
                alternate_supplier_id=r["alternate_supplier_id"],
                unit_cost_usd=r["unit_cost_usd"],
            )
        for r in conn.execute("SELECT * FROM inventory"):
            node = self.parts.get(r["part_id"])
            if node:
                node.on_hand_units = r["on_hand_units"]
                node.daily_consumption_units = r["daily_consumption_units"]
        for r in conn.execute(
            "SELECT part_id, COALESCE(SUM(units),0) AS units FROM shipments"
            " WHERE status = 'in_transit' GROUP BY part_id"
        ):
            node = self.parts.get(r["part_id"])
            if node:
                node.inbound_units = r["units"]
        for r in conn.execute("SELECT product_id, part_id, qty_per_unit FROM bom"):
            node = self.parts.get(r["part_id"])
            if node:
                node.products.append((r["product_id"], r["qty_per_unit"]))

    # ------------------------------------------------------------- helpers

    def _part_monthly_revenue(self, node: PartNode) -> float:
        """Monthly revenue of all products that consume this part."""
        return round(sum(
            self.products[pid]["unit_price_usd"] * self.products[pid]["monthly_demand_units"]
            for pid, _ in node.products if pid in self.products
        ), 2)

    def days_of_cover(self, node: PartNode, include_inbound: bool = True) -> float | None:
        if not node.daily_consumption_units:
            return None
        units = node.on_hand_units + (node.inbound_units if include_inbound else 0)
        return round(units / node.daily_consumption_units, 1)

    # ----------------------------------------------------- network metrics

    def single_points_of_failure(self, top: int = 10) -> list[dict]:
        """Critical, single-sourced parts that feed products — the riskiest
        structural positions in the network, ranked by revenue dependence."""
        spofs = []
        for node in self.parts.values():
            if not (node.critical and node.single_sourced and node.products):
                continue
            supplier = self.suppliers.get(node.primary_supplier_id, {})
            spofs.append({
                "part_id": node.part_id,
                "category": node.category,
                "supplier_id": node.primary_supplier_id,
                "supplier_name": supplier.get("name"),
                "supplier_country": supplier.get("country"),
                "monthly_revenue_dependent_usd": self._part_monthly_revenue(node),
                "days_of_cover": self.days_of_cover(node),
                "products": [pid for pid, _ in node.products],
            })
        spofs.sort(key=lambda s: s["monthly_revenue_dependent_usd"], reverse=True)
        return spofs[:top]

    def supplier_criticality(self, top: int = 10) -> list[dict]:
        """Composite 0–1 criticality index per supplier.

        score = 0.50 * normalized monthly revenue dependence (as primary source)
              + 0.25 * share of its parts that are single-sourced
              + 0.15 * geo-risk weight
              + 0.10 * (1 - on-time delivery rate, rescaled)
        """
        per_supplier: dict[str, dict] = {}
        for node in self.parts.values():
            sid = node.primary_supplier_id
            bucket = per_supplier.setdefault(sid, {
                "revenue": 0.0, "parts": 0, "single_sourced": 0, "spof": 0,
            })
            bucket["revenue"] += self._part_monthly_revenue(node)
            bucket["parts"] += 1
            bucket["single_sourced"] += node.single_sourced
            bucket["spof"] += node.single_sourced and node.critical and bool(node.products)

        max_revenue = max((b["revenue"] for b in per_supplier.values()), default=0.0) or 1.0
        ranked = []
        for sid, bucket in per_supplier.items():
            supplier = self.suppliers.get(sid)
            if supplier is None:
                continue
            revenue_score = bucket["revenue"] / max_revenue
            single_share = bucket["single_sourced"] / bucket["parts"] if bucket["parts"] else 0.0
            geo = _GEO_RISK_WEIGHT.get(supplier["geo_risk"], 0.5)
            # on_time_rate spans ~[0.82, 1.0]; rescale so the spread matters.
            otd_risk = min(max((1.0 - supplier["on_time_rate"]) / 0.18, 0.0), 1.0)
            score = (
                _W_REVENUE * revenue_score + _W_SINGLE_SOURCE * single_share
                + _W_GEO * geo + _W_OTD * otd_risk
            )
            ranked.append({
                "supplier_id": sid,
                "name": supplier["name"],
                "country": supplier["country"],
                "criticality_score": round(score, 4),
                "monthly_revenue_dependent_usd": round(bucket["revenue"], 2),
                "parts_supplied": bucket["parts"],
                "single_sourced_parts": bucket["single_sourced"],
                "single_points_of_failure": bucket["spof"],
                "geo_risk": supplier["geo_risk"],
                "on_time_rate": supplier["on_time_rate"],
            })
        ranked.sort(key=lambda s: s["criticality_score"], reverse=True)
        return ranked[:top]

    # --------------------------------------------------- outage simulation

    def simulate_outage(self, supplier_id: str, outage_days: int) -> dict:
        """Propagate a full supplier outage of `outage_days` through the network.

        Per primary-sourced part:
          cover_days   = (on-hand + already-in-transit units) / daily consumption
          relief_day   = alternate's avg lead time if an alternate exists
                         (capped at the outage end), else the outage end
          shortage     = max(0, relief_day - cover_days) days without supply

        Revenue loss is computed per *product* using the worst shortage among
        its affected parts (a product missing two parts is still one stopped
        line — no double counting), prorated on monthly demand.
        """
        if supplier_id not in self.suppliers:
            return {"error": f"unknown supplier: {supplier_id}"}

        affected_parts = []
        product_shortage: dict[str, float] = {}
        for node in self.parts.values():
            if node.primary_supplier_id != supplier_id:
                continue
            cover = self.days_of_cover(node)
            if cover is None:
                continue
            if node.single_sourced:
                relief_day = float(outage_days)
            else:
                alt = self.suppliers.get(node.alternate_supplier_id)
                alt_lead = float(alt["avg_lead_time_days"]) if alt else float(outage_days)
                relief_day = min(float(outage_days), alt_lead)
            shortage_days = round(max(0.0, relief_day - cover), 1)
            affected_parts.append({
                "part_id": node.part_id,
                "category": node.category,
                "critical": node.critical,
                "single_sourced": node.single_sourced,
                "days_of_cover": cover,
                "relief_day": relief_day,
                "shortage_days": shortage_days,
                "products": [pid for pid, _ in node.products],
            })
            if shortage_days > 0:
                for pid, _ in node.products:
                    product_shortage[pid] = max(product_shortage.get(pid, 0.0), shortage_days)

        product_losses = []
        for pid, shortage in sorted(product_shortage.items()):
            product = self.products.get(pid)
            if product is None:
                continue
            loss = round(
                product["unit_price_usd"] * product["monthly_demand_units"] * shortage / 30.0, 2
            )
            product_losses.append({
                "product_id": pid,
                "name": product["name"],
                "shortage_days": shortage,
                "estimated_revenue_loss_usd": loss,
            })
        product_losses.sort(key=lambda p: p["estimated_revenue_loss_usd"], reverse=True)

        affected_parts.sort(key=lambda p: p["shortage_days"], reverse=True)
        return {
            "supplier_id": supplier_id,
            "supplier_name": self.suppliers[supplier_id]["name"],
            "outage_days": outage_days,
            "parts_affected": len(affected_parts),
            "parts_in_shortage": sum(1 for p in affected_parts if p["shortage_days"] > 0),
            "affected_parts": affected_parts,
            "product_losses": product_losses,
            "total_estimated_revenue_loss_usd": round(
                sum(p["estimated_revenue_loss_usd"] for p in product_losses), 2
            ),
            "model": (
                "Deterministic propagation: shortage = max(0, relief_day - days_of_cover); "
                "relief via qualified alternate's lead time when one exists; revenue loss "
                "prorated per product on its worst part shortage (no double counting)."
            ),
        }
