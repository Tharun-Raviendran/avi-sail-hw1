"""Starter helpers for Homework 2. Read this before writing anything.

Conventions
  * One price axis per match: P(team1 wins) for esports, P(home wins) for MLB.
    A Kalshi trade on the OTHER team's ticker at yes_price y is a trade at 1-y
    with the taker direction flipped. `trades_on_axis` does this for you.
  * Two clocks. Kalshi `t_ns` is the exchange's clock. Feed `t_recv_ns` is when
    OUR collector (AWS Ohio, NTP-disciplined) received the message. Never use the
    feed's own payload timestamps for ordering.
  * Everything you compute at time t may only use rows with timestamp <= t.
    `asof` enforces that; if you bypass it, you are looking into the future.
"""
import numpy as np, pandas as pd
D = "data/"

def trades_on_axis(sport="esports"):
    kt = pd.read_parquet(D + "kalshi_trades.parquet")
    if sport == "esports":
        mp = pd.read_parquet(D + "map_esports.parquet"); a, b, key = "ticker_team1", "ticker_team2", "beter_match_id"
    else:
        mp = pd.read_parquet(D + "map_mlb.parquet"); a, b, key = "ticker_home", "ticker_away", "sr_match_id"
    x = mp[[a, key]].rename(columns={a: "ticker"}); x["flip"] = False
    y = mp[[b, key]].rename(columns={b: "ticker"}); y["flip"] = True
    k = kt.merge(pd.concat([x, y]), on="ticker").rename(columns={key: "match_id"})
    k["px"] = np.where(k.flip, 1 - k.yes_price, k.yes_price)
    k["dir"] = np.where(k.taker_side == "yes", 1, -1) * np.where(k.flip, -1, 1)   # +1 = taker bought team1/home
    return in_play(k, sport).sort_values("t_ns").reset_index(drop=True)

CUT = pd.Timestamp("2026-09-15", tz="UTC")

def test_match_ids(sport="esports"):
    """THE split rule: a match belongs to the TEST set if its scheduled start (Kalshi's, in
    `kalshi_start`) is on or after 2026-09-15 00:00 UTC; otherwise TRAIN. Split by match, never
    by print timestamp: US-evening MLB games cross midnight UTC and would sit in both."""
    mp = pd.read_parquet(D + ("map_esports.parquet" if sport == "esports" else "map_mlb.parquet"))
    key = "beter_match_id" if sport == "esports" else "sr_match_id"
    return set(mp.loc[pd.to_datetime(mp.kalshi_start, utc=True) >= CUT, key])

def in_play(k, sport="esports"):
    """Flag prints that happened while the match was live according to OUR feed. Prematch
    prints are a different regime; almost everything in this homework is about in-play.
    MLB: a few games have no ENDED event in the delta stream (reconnect); their span ends at the last live row."""
    if sport == "esports":
        tr = pd.read_parquet(D + "beter_esports_trading.parquet"); mw = tr[(tr.interval == 1) & (tr.result_type == 7)]
        span = mw[mw.line_type == 1].groupby("match_id").t_recv_ns.agg(["min", "max"])
        # a few matches keep emitting "live" rows for hours after the result: end the span at the first resulted row
        done = mw[mw.st1 == 3].groupby("match_id").t_recv_ns.min(); span["max"] = np.minimum(span["max"], done.reindex(span.index).fillna(np.inf))
    else:
        ev = pd.read_parquet(D + "sportradar_mlb_events.parquet"); ev = ev[(ev.feedtype == "delta") & ev.matchstatus.notna() & ~ev.matchstatus.isin(["NOT_STARTED", "ENDED"])]
        span = ev.groupby("match_id").t_recv_ns.agg(["min", "max"])
    k = k.merge(span, left_on="match_id", right_index=True, how="left")
    k["inplay"] = (k.t_ns >= k["min"]) & (k.t_ns <= k["max"]); return k.drop(columns=["min", "max"])

def betstop_state(match_id):
    """Sportradar tells bookmakers when to stop taking bets. Returns (t_recv_ns, halted) steps for
    one game: type 1011 = Betstop, 1010 = Betstart. ONLY feedtype == 'delta' rows are real time;
    'full*' rows are history replayed after a reconnect and carry a late t_recv_ns."""
    ev = pd.read_parquet(D + "sportradar_mlb_events.parquet")
    e = ev[(ev.match_id == match_id) & (ev.feedtype == "delta") & ev.type.isin(["1010", "1011"])].sort_values("t_recv_ns")
    return e.t_recv_ns.values, (e.type.values == "1011")

def beter_match_winner(sport_id=3):
    """BETER de-margined P(team1 wins match), state changes only. Prematch rows (line_type 2) are
    included; filter on line_type == 1 and st1 == 1 for the live, open market."""
    tr = pd.read_parquet(D + "beter_esports_trading.parquet")
    tr = tr[(tr.sport_id == sport_id) & (tr.interval == 1) & (tr.result_type == 7)].sort_values("t_recv_ns", kind="stable")
    chg = (tr.p1 != tr.groupby("match_id").p1.shift()) | (tr.st1 != tr.groupby("match_id").st1.shift())
    return tr[chg & tr.p1.notna()][["t_recv_ns", "match_id", "p1", "st1", "line_type"]]

def asof(left, right, left_t, right_t, by="match_id", shift_s=0.0, cols=None):
    """Value of `right` as of left_t + shift_s, per match, returned in the ROW ORDER of `left`
    (index labels are ignored, so a non-unique index cannot misalign the result).
    shift_s > 0 simulates a feed that reaches us shift_s seconds EARLIER than it really did (Part 4)."""
    q = left[[left_t, by]].reset_index(drop=True); q["_k"] = q[left_t] + int(shift_s * 1e9); q["_i"] = np.arange(len(q))
    r = right.rename(columns={right_t: "_k"}); cols = cols or [c for c in r.columns if c not in ("_k", by)]
    m = pd.merge_asof(q.sort_values("_k"), r.sort_values("_k")[["_k", by] + cols], on="_k", by=by, direction="backward")
    return m.sort_values("_i")[cols].reset_index(drop=True)

if __name__ == "__main__":
    k = trades_on_axis("esports"); b = beter_match_winner(3)
    k = k[k.match_id.isin(b.match_id)].copy()
    f = asof(k, b, "t_ns", "t_recv_ns", cols=["p1", "st1"]); k["feed"], k["feed_open"] = f.p1.values, (f.st1.values == 1)   # asof returns rows in k's order
    live = k[k.inplay & k.feed_open & k.px.between(0.03, 0.97)]
    print(live[["t_ns", "match_id", "px", "dir", "count", "feed"]].head())
    print(f"CS2 in-play prints while BETER's market is open: {len(live):,}; corr(trade price, feed) = {live.px.corr(live.feed):.3f}")
