"""Probabilistic digital twin.

Deterministic propagation (graph.py) answers "what does a 30-day outage
cost?". This module answers the question executives actually ask: "how bad
could it get, and how likely is that?" — by Monte Carlo sampling the three
quantities nobody knows in advance:

- outage duration       (triangular: min / mode / max days)
- alternate-ramp time   (multiplier on alternates' lead times)
- demand                (multiplier on monthly demand)

Every run is seeded → results are reproducible and testable. Mitigation
options are evaluated with *paired* trials (identical random draws with and
without the option) so the expected loss reduction isolates the option's
effect rather than sampling noise.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from sentinel.config import get_settings
from sentinel.graph import SupplyGraph

DEFAULT_TRIALS = 2000
EXPEDITE_TRANSIT_DAYS = 5.0


@dataclass(frozen=True)
class Triangular:
    minimum: float
    mode: float
    maximum: float

    def sample(self, rng: random.Random) -> float:
        return rng.triangular(self.minimum, self.maximum, self.mode)

    @classmethod
    def from_dict(cls, d: dict) -> Triangular:
        return cls(float(d["min"]), float(d["mode"]), float(d["max"]))


@dataclass
class ScenarioSpec:
    name: str
    description: str
    outage_days: Triangular
    alt_lead_multiplier: Triangular = field(
        default_factory=lambda: Triangular(0.8, 1.0, 1.6))
    demand_multiplier: Triangular = field(
        default_factory=lambda: Triangular(0.9, 1.0, 1.15))
    # Supplier selection — any combination; the union is affected.
    supplier_names: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    ports: list[str] = field(default_factory=list)

    def resolve_suppliers(self, graph: SupplyGraph) -> list[str]:
        selected = set()
        for sid, supplier in graph.suppliers.items():
            if supplier["name"] in self.supplier_names:
                selected.add(sid)
            if supplier["country"] in self.countries:
                selected.add(sid)
            if supplier["primary_port"] in self.ports:
                selected.add(sid)
        return sorted(selected)

    @classmethod
    def from_dict(cls, d: dict) -> ScenarioSpec:
        spec = cls(
            name=d["name"],
            description=d.get("description", ""),
            outage_days=Triangular.from_dict(d["outage_days"]),
        )
        if "alt_lead_multiplier" in d:
            spec.alt_lead_multiplier = Triangular.from_dict(d["alt_lead_multiplier"])
        if "demand_multiplier" in d:
            spec.demand_multiplier = Triangular.from_dict(d["demand_multiplier"])
        select = d.get("select", {})
        spec.supplier_names = list(select.get("suppliers", []))
        spec.countries = list(select.get("countries", []))
        spec.ports = list(select.get("ports", []))
        return spec


def load_scenarios(scenario_dir: Path | None = None) -> dict[str, ScenarioSpec]:
    scenario_dir = scenario_dir or get_settings().data_dir / "scenarios"
    library = {}
    for path in sorted(scenario_dir.glob("*.yaml")):
        spec = ScenarioSpec.from_dict(yaml.safe_load(path.read_text()))
        library[spec.name] = spec
    return library


def _quantile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank quantile on a pre-sorted list (deterministic, no interp)."""
    if not sorted_values:
        return 0.0
    index = min(int(q * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


class MonteCarloEngine:
    def __init__(self, graph: SupplyGraph, seed: int = 90061):
        self.graph = graph
        self.seed = seed

    # ------------------------------------------------------------ scenarios

    def run_scenario(self, spec: ScenarioSpec, trials: int = DEFAULT_TRIALS) -> dict:
        supplier_ids = spec.resolve_suppliers(self.graph)
        if not supplier_ids:
            return {"error": f"scenario '{spec.name}' selects no suppliers"}

        rng = random.Random(self.seed)
        losses: list[float] = []
        product_totals: dict[str, float] = {}
        for _ in range(trials):
            result = self.graph.simulate_multi_outage(
                supplier_ids,
                outage_days=spec.outage_days.sample(rng),
                alt_lead_multiplier=spec.alt_lead_multiplier.sample(rng),
                demand_multiplier=spec.demand_multiplier.sample(rng),
            )
            losses.append(result["total_estimated_revenue_loss_usd"])
            for product in result["product_losses"]:
                product_totals[product["product_id"]] = (
                    product_totals.get(product["product_id"], 0.0)
                    + product["estimated_revenue_loss_usd"]
                )

        losses.sort()
        expected = sum(losses) / trials
        top_products = sorted(
            (
                {
                    "product_id": pid,
                    "name": self.graph.products[pid]["name"],
                    "expected_loss_usd": round(total / trials, 2),
                }
                for pid, total in product_totals.items()
            ),
            key=lambda p: p["expected_loss_usd"], reverse=True,
        )[:5]
        return {
            "scenario": spec.name,
            "description": spec.description,
            "suppliers_affected": supplier_ids,
            "trials": trials,
            "seed": self.seed,
            "loss_usd": {
                "expected": round(expected, 2),
                "p50": round(_quantile(losses, 0.50), 2),
                "p90": round(_quantile(losses, 0.90), 2),
                "p95": round(_quantile(losses, 0.95), 2),
                "max": round(losses[-1], 2),
            },
            "probability_loss_exceeds": {
                "1m_usd": round(sum(loss > 1e6 for loss in losses) / trials, 3),
                "10m_usd": round(sum(loss > 1e7 for loss in losses) / trials, 3),
                "50m_usd": round(sum(loss > 5e7 for loss in losses) / trials, 3),
            },
            "top_products_by_expected_loss": top_products,
            "model": (
                "Seeded Monte Carlo over outage duration, alternate-ramp and demand "
                "uncertainty; per-trial losses from deterministic network propagation."
            ),
        }

    # ------------------------------------------ mitigation ranking under MC

    def rank_options(
        self, supplier_ids: list[str], options: list[dict],
        spec: ScenarioSpec, trials: int = 500,
    ) -> list[dict]:
        """Re-rank mitigation options by expected loss reduction per dollar.

        Paired trials: each trial draws one set of random parameters and
        evaluates the network with and without the option, so the difference
        is attributable to the option alone.
        """
        ranked = []
        for option in options:
            override = self._relief_override(option)
            if override is None:
                ranked.append({**option, "expected_loss_reduction_usd": 0.0,
                               "reduction_per_dollar": 0.0})
                continue

            rng = random.Random(self.seed)  # identical draws per option
            total_reduction = 0.0
            for _ in range(trials):
                outage = spec.outage_days.sample(rng)
                alt_mult = spec.alt_lead_multiplier.sample(rng)
                demand_mult = spec.demand_multiplier.sample(rng)
                base = self.graph.simulate_multi_outage(
                    supplier_ids, outage,
                    alt_lead_multiplier=alt_mult, demand_multiplier=demand_mult,
                )["total_estimated_revenue_loss_usd"]
                mitigated = self.graph.simulate_multi_outage(
                    supplier_ids, outage,
                    alt_lead_multiplier=alt_mult, demand_multiplier=demand_mult,
                    relief_overrides=override,
                )["total_estimated_revenue_loss_usd"]
                total_reduction += base - mitigated

            expected_reduction = round(total_reduction / trials, 2)
            cost = max(float(option.get("value_usd", 0.0)), 1.0)
            ranked.append({
                **option,
                "expected_loss_reduction_usd": expected_reduction,
                "reduction_per_dollar": round(expected_reduction / cost, 3),
            })
        ranked.sort(key=lambda o: o["reduction_per_dollar"], reverse=True)
        return ranked

    @staticmethod
    def _relief_override(option: dict) -> dict[str, float] | None:
        """Model an option as a cap on its part's relief day."""
        part_id = option.get("part_id")
        if not part_id:
            return None
        kind = option.get("kind")
        if kind == "expedite_freight":
            return {part_id: EXPEDITE_TRANSIT_DAYS}
        if kind == "alternate_source_po":
            return {part_id: float(option.get("lead_time_days", 30.0))}
        return None
