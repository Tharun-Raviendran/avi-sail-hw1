"""Parse raw BETER capture NDJSON into tidy CSVs (stdlib only).

Input : <data-root>/beter/{trading,scoreboard,incident}/**/*.ndjson
Output: <out>/trading.csv     one row per (message, market, outcome)
        <out>/scoreboard.csv  one row per scoreboard state message
        <out>/incidents.csv   one row per unique incident (match, index)

Dedupe rules (see docs): trading/scoreboard by (matchId, offset) keeping the
EARLIEST arrival — RecoverySnapshots and reconnect snapshots re-send old
state; incidents by (matchId, index). recv_wall_ns/recv_mono_ns are OUR
arrival stamps and are the timestamps to use for lead/lag work.

  python analysis/parse_beter.py --data-root data --out parsed
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None


def open_capture(path: Path):
    """Open .ndjson or .ndjson.zst transparently (hours get compressed)."""
    if path.suffix == ".zst":
        if zstd is not None:
            fh = open(path, "rb")
            return io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh),
                                    encoding="utf-8")
        p = subprocess.Popen(["zstd", "-dc", str(path)],
                             stdout=subprocess.PIPE)
        return io.TextIOWrapper(p.stdout, encoding="utf-8")
    return open(path, encoding="utf-8")


def _iter_messages(root: Path, channel: str):
    """Yield (envelope, payload) for OnUpdate data messages of a channel."""
    for path in sorted((root / "beter" / channel).rglob("*.ndjson*")):
        with open_capture(path) as f:
            for line in f:
                try:
                    env = json.loads(line)
                except ValueError:
                    continue
                if env.get("type") != "msg" or env.get("ch") != "OnUpdate":
                    continue
                try:
                    frame = json.loads(env["raw"])
                except (ValueError, KeyError):
                    continue
                for arg in frame.get("arguments", []):
                    items = arg if isinstance(arg, list) else [arg]
                    for it in items:
                        if isinstance(it, dict):
                            yield env, it


def _probability(outcome: dict) -> str:
    # Live feed (verified 2026-08-28): de-margined probability is a
    # top-level outcome field; `prices` holds only formatted prices.
    p = outcome.get("probability")
    if p is not None:
        return str(p)
    prices = outcome.get("prices")
    if isinstance(prices, dict):
        for k, v in prices.items():
            if "prob" in str(k).lower():
                return str(v)
    return ""


def parse_trading(root: Path, out: Path) -> int:
    seen: set[tuple[str, int]] = set()
    n = 0
    with open(out / "trading.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recv_wall_ns", "recv_mono_ns", "match_id", "offset",
                    "msg_type", "trading_status", "market_id", "market_type",
                    "market_value", "outcome_id", "outcome_type",
                    "outcome_value", "price", "probability",
                    "outcome_status", "outcome_result"])
        for env, m in _iter_messages(root, "trading"):
            mid = str(m.get("id", ""))
            off = m.get("offset", -1)
            key = (mid, off)
            if key in seen:
                continue
            seen.add(key)
            for mk in m.get("markets") or []:
                for o in mk.get("outcomes") or []:
                    w.writerow([env["recv_wall_ns"], env["recv_mono_ns"],
                                mid, off, m.get("messageType"),
                                m.get("tradingStatus"), mk.get("id"),
                                mk.get("marketType"), mk.get("marketValue"),
                                o.get("id"), o.get("outcomeType"),
                                o.get("outcomeValue"), o.get("price"),
                                _probability(o),
                                o.get("status"), o.get("outcomeResult")])
                    n += 1
    return n


def parse_scoreboard(root: Path, out: Path) -> int:
    seen: set[tuple[str, int]] = set()
    n = 0
    with open(out / "scoreboard.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recv_wall_ns", "recv_mono_ns", "match_id", "offset",
                    "stage", "server", "scores_json", "timer_json"])
        for env, m in _iter_messages(root, "scoreboard"):
            mid = str(m.get("id", ""))
            key = (mid, m.get("offset", -1))
            if key in seen:
                continue
            seen.add(key)
            w.writerow([env["recv_wall_ns"], env["recv_mono_ns"], mid,
                        m.get("offset"), m.get("stage"), m.get("server"),
                        json.dumps(m.get("scores"), separators=(",", ":")),
                        json.dumps(m.get("timer"), separators=(",", ":"))])
            n += 1
    return n


def parse_incidents(root: Path, out: Path) -> int:
    seen: set[tuple[str, int]] = set()
    n = 0
    with open(out / "incidents.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recv_wall_ns", "recv_mono_ns", "match_id", "index",
                    "incident_type", "occurred_date", "params_json"])
        for env, m in _iter_messages(root, "incident"):
            mid = str(m.get("id", ""))
            key = (mid, m.get("index", -1))
            if key in seen:
                continue
            seen.add(key)
            w.writerow([env["recv_wall_ns"], env["recv_mono_ns"], mid,
                        m.get("index"), m.get("type"), m.get("date"),
                        json.dumps(m.get("params"), separators=(",", ":"))])
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="parsed")
    args = ap.parse_args()
    root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"trading rows:    {parse_trading(root, out)}")
    print(f"scoreboard rows: {parse_scoreboard(root, out)}")
    print(f"incident rows:   {parse_incidents(root, out)}")


if __name__ == "__main__":
    main()
