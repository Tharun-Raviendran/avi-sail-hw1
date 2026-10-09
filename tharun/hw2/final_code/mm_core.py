"""Shared print-level fill rule, so results compare. You design the quotes; this file decides
which prints would have filled a quote at a given price and what the fill was worth. The queue
rule Part 6 imposes (improve = front, join = behind displayed size, behind = unfillable) sits ON
TOP of this: apply it to decide which of these fills you would really have received.

Fill rule (print-driven). You rest a bid and an ask on the P(team1/home wins) axis.
A taker BUY print at px >= your ask hits you first (price priority): you SELL
min(print size, q_max) at YOUR ask. A taker SELL print at px <= your bid: you BUY at
your bid. NaN means you are not quoting that side. Marks: tape mid 30 s later, and
settlement.

What this overstates: you are assumed to be first in the queue at your own price, and
your quotes do not change anyone's behaviour. Treat results as a way to RANK policies,
not as a dollar forecast. `strict=True` only counts prints that trade THROUGH your
price, which is the pessimistic bound on queue position - report both.

Your obligation: bid[i] and ask[i] may only use information stamped earlier than
trade i's time minus your reaction latency. Use `visible_index` for the tape and
`starter.asof(..., shift_s=-latency)` for feeds.
"""
import numpy as np, pandas as pd
HALF_SPREAD = 0.01   # print half-spread used to side-correct the last print (a Kalshi tick is 1c; this is not a tick)

def tape_mid(px, direction):
    """Last print corrected for the aggressor side (the best tape-only mid in Part 1)."""
    return px - direction * HALF_SPREAD

def visible_index(t_s, latency_s=0.10):
    """For each print i, index of the last print you could have SEEN before it (or -1)."""
    return np.searchsorted(t_s, t_s - latency_s, side="left") - 1

def simulate_maker(g, bid, ask, settle, q_max=100.0, horizon_s=30.0, strict=False):
    """g: one match of starter.trades_on_axis(), time-sorted. bid/ask: arrays aligned to g.
    Returns one row per fill: side, size, price, pnl to the +horizon tape mid, pnl to settlement."""
    t = g.t_ns.values / 1e9; px = g.px.values; d = g["dir"].values; q = np.minimum(g["count"].values, q_max)
    mid = tape_mid(px, d); fut = mid[np.searchsorted(t, t + horizon_s, side="right") - 1]
    sell = (d > 0) & ((px > ask) if strict else (px >= ask)); buy = (d < 0) & ((px < bid) if strict else (px <= bid))
    rows = []
    for mask, sgn, price, side in ((sell, -1.0, ask, "sell"), (buy, 1.0, bid, "buy")):
        if mask.any():
            rows.append(pd.DataFrame({"t_ns": g.t_ns.values[mask], "side": side, "size": q[mask], "price": price[mask],
                                      "pnl_mid": sgn * (fut[mask] - price[mask]), "pnl_settle": sgn * (settle - price[mask])}))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["t_ns", "side", "size", "price", "pnl_mid", "pnl_settle"])

def summarize(fills, total_contracts):
    """Volume share and cents per contract. Cluster by match before you quote a t-stat."""
    w = fills["size"]
    if len(fills) == 0 or w.sum() == 0:
        return {"fills": 0, "contracts": 0.0, "share_pct": 0.0, "pnl_mid_c": float("nan"), "pnl_settle_c": float("nan")}
    return {"fills": len(fills), "contracts": float(w.sum()), "share_pct": 100 * float(w.sum()) / total_contracts,
            "pnl_mid_c": 100 * float(np.average(fills.pnl_mid, weights=w)), "pnl_settle_c": 100 * float(np.average(fills.pnl_settle, weights=w))}

if __name__ == "__main__":   # baseline: tape-only maker, 2c half-spread, CS2. Your job is to beat this.
    import starter
    k = starter.trades_on_axis("esports"); mp = pd.read_parquet("data/map_esports.parquet")
    mp = mp[mp.sport_id == 3].set_index("beter_match_id"); out = []; tot = 0.0
    for mid, g in k[k.match_id.isin(mp.index)].groupby("match_id"):
        g = g[g.inplay] if "inplay" in g else g
        if len(g) < 100 or mp.result_team1.get(mid) not in ("yes", "no"): continue
        t = g.t_ns.values / 1e9; i = visible_index(t); F = np.where(i >= 0, tape_mid(g.px.values, g["dir"].values)[np.clip(i, 0, None)], np.nan)
        F = np.where((F > 0.05) & (F < 0.95), F, np.nan)
        bid, ask = np.floor((F - 0.02) * 100 + 1e-9) / 100, np.ceil((F + 0.02) * 100 - 1e-9) / 100
        out.append(simulate_maker(g, bid, ask, 1.0 if mp.result_team1[mid] == "yes" else 0.0).assign(match_id=mid)); tot += g["count"].sum()
    print(summarize(pd.concat(out), tot))
    # NOTE: the answer key's mm_sim.py adds a 120 s tape-staleness guard and a 0.03-0.97 price filter,
    # so its "tape only" numbers are a little less negative than this baseline. Both are correct.
