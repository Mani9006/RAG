"""SQL validation guardrail.

Every generated statement passes through :func:`validate_sql` before it is
allowed anywhere near the database. Three independent checks:

1. **Shape** -- exactly one statement, and it must be a read (``SELECT`` /
   ``WITH``). Any DDL/DML verb (``DROP``, ``DELETE``, ``UPDATE``, ``INSERT``,
   ``ALTER``, ``CREATE``, ``ATTACH``, ``COPY``, ``PRAGMA`` ...) is rejected.
2. **Columns** -- every ``table.column`` reference must exist in the semantic
   model's physical catalog, so a hallucinated column name is caught before
   execution rather than as a runtime error.
3. **Parse** -- optionally, DuckDB's ``EXPLAIN`` binds the statement against the
   live schema (catches typos, bad joins) without executing it.

This is the "text-to-SQL is only safe if it's validated" story: an LLM could be
swapped in for the generator and this guardrail would still hold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Verbs that must never appear as a *statement* -- mutation / DDL / side effects.
_FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "replace", "attach", "detach", "copy", "pragma", "call", "export",
    "import", "install", "load", "set", "grant", "revoke", "merge", "vacuum",
}

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_QCOL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")


class SQLValidationError(ValueError):
    """Raised when a statement fails a safety or correctness check."""


@dataclass
class ValidationReport:
    ok: bool
    reason: str = ""


def _strip(sql: str) -> str:
    return _COMMENT_RE.sub(" ", sql).strip().rstrip(";").strip()


def validate_sql(
    sql: str,
    catalog: dict[str, set[str]] | None = None,
    con=None,
) -> ValidationReport:
    """Validate *sql*. Raises :class:`SQLValidationError` on failure.

    Parameters
    ----------
    sql : str
        The candidate statement.
    catalog : dict[str, set[str]] | None
        ``table -> {columns}``. When provided, qualified column references are
        checked against it.
    con : duckdb connection | None
        When provided, ``EXPLAIN`` binds the statement against the live schema.
    """
    body = _strip(sql)
    if not body:
        raise SQLValidationError("empty statement")

    # (1a) single statement -- no stacked queries.
    if ";" in body:
        raise SQLValidationError("multiple statements are not allowed")

    # (1b) must be a read.
    first = re.match(r"\s*([A-Za-z]+)", body)
    lead = (first.group(1).lower() if first else "")
    if lead not in {"select", "with"}:
        raise SQLValidationError(f"only SELECT/WITH allowed, got '{lead.upper()}'")

    # (1c) no forbidden verb anywhere as a word (defends against CTE tricks like
    # `WITH x AS (...) DELETE ...`).
    words = set(re.findall(r"[A-Za-z_]+", body.lower()))
    hit = _FORBIDDEN & words
    if hit:
        raise SQLValidationError(f"forbidden keyword(s): {', '.join(sorted(hit))}")

    # (2) column-existence check.
    if catalog is not None:
        known_tables = {t.lower(): {c.lower() for c in cols}
                        for t, cols in catalog.items()}
        for tbl, col in _QCOL_RE.findall(body):
            t, c = tbl.lower(), col.lower()
            if t in known_tables and c not in known_tables[t]:
                raise SQLValidationError(
                    f"unknown column '{tbl}.{col}' (not in {tbl})"
                )

    # (3) live parse/bind via DuckDB, no execution.
    if con is not None:
        try:
            con.execute("EXPLAIN " + body)
        except Exception as exc:  # noqa: BLE001 - surface as validation failure
            raise SQLValidationError(f"parse/bind failed: {exc}") from exc

    return ValidationReport(ok=True)


def is_safe(sql: str, catalog: dict[str, set[str]] | None = None, con=None) -> bool:
    """Boolean convenience wrapper around :func:`validate_sql`."""
    try:
        validate_sql(sql, catalog=catalog, con=con)
        return True
    except SQLValidationError:
        return False
