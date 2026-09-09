"""Build a session manifest from a captured time_table channel.

For captures made WITHOUT the booking orchestrator (e.g. everything was
already booked on the account, so only the four feed channels ran), there
is no manifest for auto-pairing. This reconstructs one from the
time_table capture: every fixture seen, with English names, start time,
sport and league.

  python analysis/make_manifest.py --data-root data
  -> data/tt_session/manifest-from-timetable.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _en(x):
    if isinstance(x, dict):
        n = x.get("name", x)
        if isinstance(n, dict):
            return n.get("en") or next(iter(n.values()), None)
        return str(n) if n is not None else None
    return str(x) if x is not None else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    root = Path(args.data_root)
    bookings: dict[str, dict] = {}
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from parse_beter import open_capture
    for path in sorted((root / "beter" / "time_table").rglob("*.ndjson*")):
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
                    for it in (arg if isinstance(arg, list) else [arg]):
                        if not isinstance(it, dict):
                            continue
                        mid = str(it.get("id") or "")
                        parts = [_en(p) for p in
                                 (it.get("participants") or [])
                                 if isinstance(p, dict)]
                        parts = [p for p in parts if p]
                        if not mid or not parts:
                            continue
                        bookings[mid] = {
                            "name": " vs ".join(parts),
                            "sport": it.get("sport"),
                            "sportId": it.get("sportId"),
                            "startDate": it.get("startDate"),
                            "league": _en(it.get("league")),
                            "participants": parts,
                        }
    out_dir = root / "tt_session"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "manifest-from-timetable.json"
    out.write_text(json.dumps({"source": "time_table capture",
                               "bookings": bookings}, indent=2))
    by_sport: dict = {}
    for b in bookings.values():
        by_sport[b["sport"]] = by_sport.get(b["sport"], 0) + 1
    print(f"{len(bookings)} fixtures -> {out}")
    for s, n in sorted(by_sport.items()):
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
