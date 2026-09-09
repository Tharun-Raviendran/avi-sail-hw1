"""Parse raw Kalshi capture NDJSON into tidy CSVs (stdlib only).

Input : <data-root>/kalshi/rest/**/*.ndjson   (live poller capture)
        and/or <data-root>/kalshi/backfill/*-trades.ndjson (backfill)
Output: <out>/kalshi_quotes.csv  one row per (snapshot, market)
        <out>/kalshi_trades.csv  one row per unique trade_id

Kalshi is migrating field units: responses may carry cents ints (yes_bid),
dollar strings (yes_bid_dollars), or both. Everything is normalized to
CENTS here.

  python analysis/parse_kalshi.py --data-root data --out parsed
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from parse_beter import open_capture


def _cents(obj: dict, base: str):
    """Prefer '<base>' (cents int); fall back to '<base>_dollars' string."""
    v = obj.get(base)
    if isinstance(v, (int, float)):
        return round(v)
    d = obj.get(base + "_dollars")
    if d is not None:
        try:
            return round(float(d) * 100)
        except (TypeError, ValueError):
            pass
    return ""


def _num(obj: dict, *names):
    for n in names:
        v = obj.get(n)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return ""


def parse_live(root: Path, quotes_writer, trades_writer,
               seen_trades: set) -> tuple[int, int]:
    nq = nt = 0
    for path in sorted((root / "kalshi" / "rest").rglob("*.ndjson*")):
        with open_capture(path) as f:
            for line in f:
                try:
                    env = json.loads(line)
                except ValueError:
                    continue
                if env.get("type") != "msg":
                    continue
                ch = env.get("ch", "")
                try:
                    body = json.loads(env["raw"])
                except (ValueError, KeyError):
                    continue
                if ch == "snapshot":
                    for m in body.get("markets", []):
                        quotes_writer.writerow(
                            [env["recv_wall_ns"], env["recv_mono_ns"],
                             m.get("ticker"), m.get("status"),
                             _cents(m, "yes_bid"), _cents(m, "yes_ask"),
                             _cents(m, "last_price"),
                             _num(m, "volume", "volume_fp"),
                             _num(m, "open_interest", "open_interest_fp"),
                             m.get("result", "")])
                        nq += 1
                elif ch.startswith("trades/"):
                    for tr in body.get("trades", []):
                        tid = tr.get("trade_id")
                        if not tid or tid in seen_trades:
                            continue
                        seen_trades.add(tid)
                        trades_writer.writerow(
                            [tr.get("created_time"), tr.get("ticker"),
                             _cents(tr, "yes_price"),
                             _num(tr, "count", "count_fp"),
                             tr.get("taker_side"), tid,
                             env["recv_wall_ns"]])
                        nt += 1
    return nq, nt


def parse_backfill(root: Path, trades_writer, seen_trades: set) -> int:
    nt = 0
    bf = root / "kalshi" / "backfill"
    for path in sorted(bf.glob("*-trades.ndjson*")) if bf.exists() else []:
        with open_capture(path) as f:
            for line in f:
                try:
                    tr = json.loads(line)
                except ValueError:
                    continue
                tid = tr.get("trade_id")
                if not tid or tid in seen_trades:
                    continue
                seen_trades.add(tid)
                trades_writer.writerow(
                    [tr.get("created_time"), tr.get("ticker"),
                     _cents(tr, "yes_price"), _num(tr, "count", "count_fp"),
                     tr.get("taker_side"), tid, ""])
                nt += 1
    return nt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="parsed")
    args = ap.parse_args()
    root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    with open(out / "kalshi_quotes.csv", "w", newline="") as fq, \
         open(out / "kalshi_trades.csv", "w", newline="") as ft:
        wq, wt = csv.writer(fq), csv.writer(ft)
        wq.writerow(["recv_wall_ns", "recv_mono_ns", "ticker", "status",
                     "yes_bid_c", "yes_ask_c", "last_price_c", "volume",
                     "open_interest", "result"])
        wt.writerow(["created_time", "ticker", "yes_price_c", "count",
                     "taker_side", "trade_id", "recv_wall_ns"])
        nq, nt = parse_live(root, wq, wt, seen)
        nb = parse_backfill(root, wt, seen)
    print(f"quote rows: {nq}   live trades: {nt}   backfill trades: {nb}")


if __name__ == "__main__":
    main()
