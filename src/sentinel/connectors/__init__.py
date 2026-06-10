from sentinel.connectors.base import Footprint, SignalRecord, ingest_records
from sentinel.connectors.gdelt import GdeltConnector
from sentinel.connectors.noaa import NoaaConnector
from sentinel.connectors.usgs import UsgsConnector

CONNECTORS = {
    "usgs": UsgsConnector,
    "noaa": NoaaConnector,
    "gdelt": GdeltConnector,
}


def run_ingest(conn, source: str = "all", window_hours: int = 24) -> dict:
    """Fetch + normalize + ingest from one or all live connectors.

    Connector failures are isolated: one unreachable feed never blocks the
    others. Returns a per-source report.
    """
    names = list(CONNECTORS) if source == "all" else [source]
    footprint = Footprint.load(conn)
    report: dict[str, dict] = {}
    for name in names:
        connector = CONNECTORS[name]()
        try:
            payload = connector.fetch(window_hours=window_hours)
            records = connector.normalize(payload, footprint)
            report[name] = ingest_records(conn, records)
        except Exception as exc:  # network/parse errors are per-source results
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return report


__all__ = ["CONNECTORS", "Footprint", "SignalRecord", "ingest_records", "run_ingest",
           "UsgsConnector", "NoaaConnector", "GdeltConnector"]
