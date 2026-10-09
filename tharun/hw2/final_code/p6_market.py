"""Part 6: make markets against the REAL Kalshi touch with queue priority.

Queue model (the README's rule plus our stated drain model):
  * our quote IMPROVES the touch (bid above the best bid / ask below the best ask)  -> first in line (nothing ahead);
  * our quote JOINS the touch (equal to it)  -> behind the displayed size at that price, measured when we post or
    when the touch comes to our price;
  * our quote is BEHIND the touch  -> not fillable (even if a sweep later prints through it before the touch row
    updates: the conservative reading of the rule);
  * queue drain: prints AT our price consume the size ahead of us first (cancellations by others are only counted
    when the displayed size falls below what we think is ahead); a print BELOW our bid (above our ask) means the
    level was cleared, so we are filled;
  * any change of our price, or a re-post after being fully filled, re-joins at the back.
Prices are integer cents on the match axis. Decisions are made once per second from information <= t and go live
at t + LAT (reaction latency). Fills are marked to the recorded touch mid at +30 s / +120 s and to settlement.
"""
import numpy as np, pandas as pd

from starter import D as DATA, asof
from p1_mid import LAT
from p3_leadlag import touch_mid_at
from p5_forecast import settle_values, wmean_t

TOUCH_MAX_AGE_S = 5.0


# =================================================================================================
# decision grid: one row per in-play second, with the forecasts and the touch in force
# =================================================================================================
def decision_grid(ctx, F):
    """Every in-play second of every book_ok match (first to last in-play print, as in Part 1), with the Part 5
    forecast row for that second (NaN where Part 5 had none: stale / crossed touch) and the touch in force at t."""
    ids = set(F.match_id)
    tr = ctx["tr"][ctx["tr"].match_id.isin(ids)]
    span = tr.groupby("match_id").t_ns.agg(["min", "max"]) / 1e9
    Dg = pd.concat([pd.DataFrame({"match_id": m, "t": np.arange(r["min"], r["max"], 1.0)}) for m, r in span.iterrows()],
                   ignore_index=True)
    Dg["tk"] = np.round(Dg.t * 1000).astype("int64"); F = F.assign(tk=np.round(F.t * 1000).astype("int64"))
    Dg = Dg.merge(F.drop(columns=["t"]), on=["match_id", "tk"], how="left").drop(columns="tk")
    q = Dg[["match_id"]].assign(q=(Dg.t * 1e9).astype("int64"))
    bb = ctx["bb"].assign(bt=ctx["bb"].t_ns)
    r = asof(q, bb, "q", "t_ns", cols=["bid", "ask", "bid_sz", "ask_sz", "bt"])
    Dg["bid_c"] = np.round(100 * r.bid.values); Dg["ask_c"] = np.round(100 * r.ask.values)
    Dg["bid_sz"], Dg["ask_sz"] = r.bid_sz.values, r.ask_sz.values
    Dg["touch_ok"] = ((q.q.values - r.bt.values) / 1e9 <= TOUCH_MAX_AGE_S) & (Dg.bid_c < Dg.ask_c)
    Dg["micro"] = (r.bid * r.ask_sz + r.ask * r.bid_sz).values / (r.bid_sz + r.ask_sz).values
    Dg["test"] = Dg.match_id.isin(ctx["test"])
    return Dg.sort_values(["match_id", "t"]).reset_index(drop=True)

def feed_columns(Dg, sport, k, ev=None):
    """Feed state at each decision second, delivered k s earlier: CS2 BETER p1 and 'market suspended';
    MLB Betstop active and seconds since the last pitch release."""
    a = ((Dg.t - LAT + k) * 1e9).astype("int64")
    q = Dg[["match_id"]].assign(q=a.values)
    out = {}
    if sport == "CS2":
        from starter import beter_match_winner
        bw = beter_match_winner(3); bw = bw[(bw.line_type == 1) & bw.match_id.isin(set(Dg.match_id))]
        r = asof(q, bw, "q", "t_recv_ns", cols=["p1", "st1"])
        out["beter_p1"] = np.where(r.st1.values == 1, r.p1.values, np.nan)
        out["beter_suspended"] = (r.st1.values == 2)
    else:
        d = ev[(ev.feedtype == "delta") & ev.match_id.isin(set(Dg.match_id))]
        bs = d[d.type.isin(["1010", "1011"])].assign(stop=lambda x: (x.type == "1011").astype(float))
        out["betstop"] = asof(q, bs, "q", "t_recv_ns", cols=["stop"]).stop.fillna(0).values == 1
        pr = d[d.type == "2327"].assign(pt=lambda x: x.t_recv_ns)
        r = asof(q, pr, "q", "t_recv_ns", cols=["pt"])
        out["since_pitch_s"] = (a.values - r.pt.values) / 1e9
    return pd.DataFrame(out, index=Dg.index)


# =================================================================================================
# the quoting policy (vectorized over decision seconds) -> the bid / ask we want, integer cents (-1 = none)
# =================================================================================================
def policy_quotes(Dg, fair, h, placement="penny", pull_bid=None, pull_ask=None):
    """fair: fair value in dollars per decision second. h: half-spread in cents, or 'join' (sit on the touch when the
    fair value is inside the spread). placement: 'naive' = post fair -/+ h (post-only: never crossing the touch);
    'penny' = additionally never improve the touch by more than one tick (being first in line needs only one tick).
    pull_bid / pull_ask: boolean masks of seconds when that side is pulled."""
    bb, ba = Dg.bid_c.values, Dg.ask_c.values
    fc = 100 * np.asarray(fair, float)
    ok = Dg.touch_ok.values & np.isfinite(fc) & (fc > 3) & (fc < 97)
    if h == "join":
        wb = np.where(fc > bb, bb, -1); wa = np.where(fc < ba, ba, -1)
    else:
        wb = np.floor(fc - h + 1e-9); wa = np.ceil(fc + h - 1e-9)
        wb = np.minimum(wb, ba - 1); wa = np.maximum(wa, bb + 1)                 # post-only
        if placement == "penny":
            wb = np.minimum(wb, bb + 1); wa = np.maximum(wa, ba - 1)
        wb = np.where(wb >= 1, wb, -1); wa = np.where(wa <= 99, wa, -1)
    wb = np.where(ok, wb, -1); wa = np.where(ok, wa, -1)
    if pull_bid is not None: wb = np.where(pull_bid, -1, wb)
    if pull_ask is not None: wa = np.where(pull_ask, -1, wa)
    return np.nan_to_num(wb, nan=-1).astype(int), np.nan_to_num(wa, nan=-1).astype(int)


# =================================================================================================
# the queue simulator
# =================================================================================================
def _prints(ctx, ids):
    tr = ctx["tr"][ctx["tr"].match_id.isin(ids)].sort_values(["match_id", "t_ns"], kind="stable")
    q = tr[["match_id"]].assign(q=tr.t_ns.values)
    r = asof(q, ctx["bb"], "q", "t_ns", cols=["bid", "ask", "bid_sz", "ask_sz"])      # touch just before the print
    return pd.DataFrame({"match_id": tr.match_id.values, "t": tr.t_ns.values / 1e9, "px_c": np.round(100 * tr.px.values),
                         "dir": tr["dir"].values, "q": tr["count"].values,
                         "bb": np.round(100 * r.bid.values), "ba": np.round(100 * r.ask.values),
                         "bbsz": r.bid_sz.values, "basz": r.ask_sz.values})

class Sim:
    """Holds the prints (with the touch before each) once; run() replays any quote schedule against them."""
    def __init__(self, ctx, Dg, sport):
        self.ctx, self.Dg, self.sport = ctx, Dg, sport
        self.P = _prints(ctx, set(Dg.match_id))
        self.settle = settle_values(sport)
        self.vol = self.P.groupby("match_id").q.sum()                         # in-play contracts per match

    def run(self, wb, wa, Q, inv_limit=np.inf):
        """wb / wa: wanted bid / ask (int cents, -1 = none) per decision second of Dg. Q: post size (contracts), a
        number or an array per decision second. inv_limit: stop buying when long >= inv_limit contracts (stop selling
        when short <= -inv_limit). Returns one row per fill with marks."""
        Dg, P = self.Dg, self.P
        Qa = np.broadcast_to(np.asarray(Q, float), (len(Dg),))
        fills = []
        dgi = Dg.groupby("match_id", sort=False).indices; pri = P.groupby("match_id", sort=False).indices
        for m, di in dgi.items():
            if m not in pri: continue
            pi = pri[m]
            dt = (Dg.t.values[di] + LAT).tolist(); dB = wb[di].tolist(); dA = wa[di].tolist(); dQ = Qa[di].tolist()
            dbb = Dg.bid_c.values[di].tolist(); dba = Dg.ask_c.values[di].tolist()
            dbs = Dg.bid_sz.values[di].tolist(); das = Dg.ask_sz.values[di].tolist()
            pt = P.t.values[pi].tolist(); ppx = P.px_c.values[pi].tolist(); pdr = P["dir"].values[pi].tolist(); pq = P.q.values[pi].tolist()
            pbb = P.bb.values[pi].tolist(); pba = P.ba.values[pi].tolist(); pbs = P.bbsz.values[pi].tolist(); pas = P.basz.values[pi].tolist()
            bp = ap = -1; brem = arem = 0.0; bah = aah = 0.0; bat = aat = False; pos = 0.0
            i = j = 0; nd, npr = len(dt), len(pt)
            out_t, out_side, out_sz, out_px = [], [], [], []
            while i < nd or j < npr:
                if j >= npr or (i < nd and dt[i] <= pt[j]):
                    # ---- decision: (re)post / cancel
                    w = dB[i]
                    if w < 0: bp = -1
                    elif w != bp or brem <= 0:
                        bp, brem = w, dQ[i]
                        if dbb[i] != dbb[i]: bat = False
                        elif w > dbb[i]: bah, bat = 0.0, True
                        elif w == dbb[i]: bah, bat = dbs[i], True
                        else: bat = False
                    w = dA[i]
                    if w < 0: ap = -1
                    elif w != ap or arem <= 0:
                        ap, arem = w, dQ[i]
                        if dba[i] != dba[i]: aat = False
                        elif w < dba[i]: aah, aat = 0.0, True
                        elif w == dba[i]: aah, aat = das[i], True
                        else: aat = False
                    i += 1
                else:
                    # ---- a print: update our queue status against the touch just before it, then fill
                    tb, ta = pbb[j], pba[j]
                    if bp > 0 and tb == tb:
                        if bp > tb: bah, bat = 0.0, True
                        elif bp == tb:
                            if not bat: bah, bat = pbs[j], True
                            else: bah = min(bah, pbs[j])
                        else: bat = False
                    if ap > 0 and ta == ta:
                        if ap < ta: aah, aat = 0.0, True
                        elif ap == ta:
                            if not aat: aah, aat = pas[j], True
                            else: aah = min(aah, pas[j])
                        else: aat = False
                    px, q = ppx[j], pq[j]
                    if pdr[j] < 0 and bp > 0 and bat and brem > 0 and pos < inv_limit:   # taker SELL -> may hit our bid
                        f = 0.0
                        if px < bp: f = min(brem, q)
                        elif px == bp: f = min(brem, max(0.0, q - bah)); bah = max(0.0, bah - q)
                        f = min(f, inv_limit - pos)
                        if f > 0: brem -= f; pos += f; out_t.append(pt[j]); out_side.append(1); out_sz.append(f); out_px.append(bp)
                    elif pdr[j] > 0 and ap > 0 and aat and arem > 0 and pos > -inv_limit:  # taker BUY -> may lift our ask
                        f = 0.0
                        if px > ap: f = min(arem, q)
                        elif px == ap: f = min(arem, max(0.0, q - aah)); aah = max(0.0, aah - q)
                        f = min(f, inv_limit + pos)
                        if f > 0: arem -= f; pos -= f; out_t.append(pt[j]); out_side.append(-1); out_sz.append(f); out_px.append(ap)
                    j += 1
            if out_t:
                fills.append(pd.DataFrame({"match_id": m, "t": out_t, "side": out_side, "size": out_sz, "price_c": out_px}))
        if not fills:
            return pd.DataFrame(columns=["match_id", "t", "side", "size", "price_c", "pnl30", "pnl120", "pnl_settle", "test"])
        Fl = pd.concat(fills, ignore_index=True)
        p = Fl.price_c.values / 100; s = Fl.side.values
        for H in (30, 120):
            Fl[f"pnl{H}"] = s * (touch_mid_at(self.ctx["bb"], Fl.match_id.values, Fl.t.values + H) - p)
        Fl["pnl_settle"] = s * (Fl.match_id.map(self.settle).values - p)
        Fl["test"] = Fl.match_id.isin(self.ctx["test"])
        return Fl

    def summary(self, Fl, split):
        """Volume and P&L per contract (cents) at +30 s, +120 s, settlement, with match-clustered t-stats."""
        d = Fl[Fl.test] if split == "test" else Fl[~Fl.test]
        ids = self.ctx["test"] if split == "test" else self.ctx["train"]
        tot = self.vol[self.vol.index.isin(ids)].sum()
        r = {"fills": len(d), "contracts": d["size"].sum(), "share_pct": 100 * d["size"].sum() / tot if tot else np.nan}
        for c in ("pnl30", "pnl120", "pnl_settle"):
            v, t = wmean_t(d[c].values, d["size"].values, d.match_id.values) if len(d) else (np.nan, np.nan)
            r[c + "_c"], r[c + "_t"] = v, t
        r["total_pnl30_$"] = np.nansum(d.pnl30 * d["size"])
        return r


# =================================================================================================
# versions, frontier, break-even volume
# =================================================================================================
def break_even_volume(points, vol="vol", pnl="pnl"):
    """Largest volume on the upper envelope of (volume, P&L per contract) points with P&L >= 0, interpolating
    between the last point at or above zero and the next one below it. 0 if no point breaks even."""
    P = points.sort_values(vol).reset_index(drop=True)
    env = []                                                    # upper envelope: best P&L at each volume or more
    best = -np.inf
    for _, r in P.iloc[::-1].iterrows():
        best = max(best, r[pnl]); env.append((r[vol], best))
    env = env[::-1]                                             # increasing volume, non-increasing P&L
    ok = [i for i, (v, p) in enumerate(env) if p >= 0]
    if not ok: return 0.0
    i = max(ok)
    if i == len(env) - 1: return env[i][0]
    (v0, p0), (v1, p1) = env[i], env[i + 1]
    return v0 + (v1 - v0) * p0 / (p0 - p1) if p0 != p1 else v0


# =================================================================================================
# the four versions (shared by part6.ipynb and the summary notebook)
# =================================================================================================
FAIR = {"CS2": "micro", "MLB": "mtape"}
VERSIONS = ["tape only", "feed as received", "feed k s earlier", "Part 5 model"]

def version_quotes(s, version, h, params, Dg, FD, K, return_masks=False):
    """Bid / ask (int cents, -1 = none) for one version. params[s]: the train-chosen components.
    FD[(s, k)]: feed_columns for k = 0 and the Part 4 k. With return_masks, also returns the pull masks."""
    d, p = Dg, params[s]
    fair = d[FAIR[s]].values.copy()
    pb = np.zeros(len(d), bool); pa = np.zeros(len(d), bool)
    if version in ("feed as received", "feed k s earlier", "Part 5 model"):
        fd = FD[(s, K[s] if version == "feed k s earlier" else 0.0)]
        if s == "CS2":
            if p.get("w_beter", 0) > 0:
                fair = np.where(np.isfinite(fd.beter_p1.values), fair + p["w_beter"] * (fd.beter_p1.values - fair), fair)
            if p.get("pull_suspended"): pb |= fd.beter_suspended.values; pa |= fd.beter_suspended.values
        else:
            pull = np.zeros(len(d), bool)
            if p.get("pitch_window", 0) > 0: pull |= (fd.since_pitch_s.values >= 0) & (fd.since_pitch_s.values <= p["pitch_window"])
            if p.get("pull_betstop"): pull |= fd.betstop.values
            pb |= pull; pa |= pull
    if version == "Part 5 model":
        x = p.get("size_q")
        if x is not None:
            big = (d.size10 > d.loc[~d.test, "size10"].quantile(x)).values
            if p.get("one_sided", True): pb |= big & (d.dir10.values < 0); pa |= big & (d.dir10.values > 0)
            else: pb |= big; pa |= big
        fair = fair + p.get("beta", 0) * np.nan_to_num(d.dir10.values) / 100
    wb, wa = policy_quotes(d, fair, h, "penny", pb, pa)
    if return_masks:
        wb0, wa0 = policy_quotes(d, fair, h, "penny")              # same fair value, no pulls
        return wb, wa, wb0, wa0
    return wb, wa


# =================================================================================================
# selection on TRAIN, frontier and break-even volume on TEST
# =================================================================================================
def greedy_select(score_fn, base, options, min_volume_kept=0.6):
    """Add components one at a time: keep the best candidate of each option only if it raises TRAIN P&L per contract
    and keeps >= min_volume_kept of the current volume. score_fn(params) -> (pnl_c, volume)."""
    cur = dict(base); p0, v0 = score_fn(cur); log = [("start", dict(cur), p0, v0)]
    for name, cands in options:
        best = None
        for c in cands:
            p, v = score_fn({**cur, **c})
            if p > p0 and v >= min_volume_kept * v0 and (best is None or p > best[1]): best = (c, p, v)
        if best: cur = {**cur, **best[0]}; p0, v0 = best[1], best[2]
        log.append((name, dict(cur), p0, v0))
    return cur, log

def frontier_points(sim, quotes_fn, versions, hs, Qs, test_mids):
    """Every (version, half-spread, post size) on TEST: summary + per-match sums for the bootstrap."""
    rows, PM = [], {}
    for v in versions:
        for h in hs:
            wb, wa = quotes_fn(v, h)
            for Q in Qs:
                Fl = sim.run(wb, wa, Q); r = sim.summary(Fl, "test")
                te = Fl[Fl.test].dropna(subset=["pnl30"])
                g = te.assign(x=te.pnl30 * te["size"]).groupby("match_id")[["x", "size"]].sum().reindex(test_mids).fillna(0)
                PM[(v, h, Q)] = (g.x.values, g["size"].values)
                rows.append({"version": v, "h": h, "Q": Q, "share": r["share_pct"], "contracts": r["contracts"], "pnl": r["pnl30_c"], "t": r["pnl30_t"]})
    return pd.DataFrame(rows), PM

def bootstrap_break_even(FR, PM, vol_m, volcol, versions, B=500, seed=0):
    """Break-even volume per version with match-bootstrap draws (the same draws for every version, so differences are
    paired). Returns ({version: point estimate}, {version: B draws})."""
    rng_ = np.random.default_rng(seed); n = len(vol_m)
    Wt = rng_.multinomial(n, np.ones(n) / n, size=B).astype(float)
    point, boot = {}, {}
    for v in versions:
        d = FR[FR.version == v].reset_index(drop=True)
        point[v] = break_even_volume(d.rename(columns={volcol: "vol"}))
        X = np.array([PM[(v, r.h, r.Q)][0] for _, r in d.iterrows()]); Wm = np.array([PM[(v, r.h, r.Q)][1] for _, r in d.iterrows()])
        bx, bw, bv = Wt @ X.T, Wt @ Wm.T, Wt @ vol_m
        pnl_b = 100 * bx / np.maximum(bw, 1e-12)
        vol_b = 100 * bw / bv[:, None] if volcol == "share" else bw
        boot[v] = np.array([break_even_volume(pd.DataFrame({"vol": vol_b[b], "pnl": pnl_b[b]})) for b in range(B)])
    return point, boot

def frontier_figure(FR_all, BE, path):
    import matplotlib.pyplot as plt
    from p3_leadlag import INK, INK2, GRID, C_TEAM1, C_TEAM2
    cols = {"tape only": "#8f8e89", "feed as received": C_TEAM1, "feed k s earlier": "#7a5bd6", "Part 5 model": C_TEAM2}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, s in zip(axes, ("CS2", "MLB")):
        volcol = "share" if s == "CS2" else "contracts"
        for v in VERSIONS:
            d = FR_all[(FR_all.sport == s) & (FR_all.version == v)].sort_values(volcol)
            ax.scatter(d[volcol], d.pnl, s=14, color=cols[v], alpha=0.35, lw=0)
            env, best = [], -np.inf
            for _, r in d.iloc[::-1].iterrows():
                if r.pnl > best: best = r.pnl; env.append((r[volcol], r.pnl))
            env = np.array(env[::-1]); ax.plot(env[:, 0], env[:, 1], "-o", ms=4, color=cols[v], lw=2, label=v)
            bev = BE[(BE.sport == s) & (BE.version == v)]["break-even volume"].item()
            if bev > 0: ax.plot([bev], [0], "|", ms=14, mew=2.5, color=cols[v])
        ax.axhline(0, color=INK, lw=0.9); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xscale("log")
        ax.set_ylim((-0.9, 0.7) if s == "CS2" else (-0.6, 0.25))
        ax.set_xlabel("volume: share of in-play volume (%)" if s == "CS2" else "volume: contracts filled (test)")
        ax.set_title(s, loc="left", fontweight="bold")
    axes[0].set_ylabel("P&L per contract, marked to touch mid +30 s (cents)"); axes[0].legend(frameon=False, fontsize=9)
    fig.text(0.01, 0.985, "Part 6 · volume vs P&L frontier on TEST, four versions", fontsize=13, fontweight="bold", va="top")
    fig.text(0.01, 0.94, "points = every half-spread (join, 1–3c) × post size; line = upper envelope; tick on the zero line = break-even volume · "
             "points far below the envelopes are off the axis", fontsize=9, color=INK2, va="top")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.11, wspace=0.18); fig.savefig(path, dpi=130); plt.show()

def quote_time_row(Dg, wb, wa, wb0, wa0, fair_col):
    """Shares of TEST in-play seconds: quoting both / one / neither side, pulled by a rule, fillable (on or inside
    the touch), and no usable touch or extreme price."""
    te = Dg.test.values
    wb, wa, wb0, wa0 = wb[te], wa[te], wb0[te], wa0[te]
    bb, ba, fair = Dg.bid_c.values[te], Dg.ask_c.values[te], Dg[fair_col].values[te]
    usable = Dg.touch_ok.values[te] & np.isfinite(fair) & (fair > 0.03) & (fair < 0.97)
    qb, qa = wb >= 0, wa >= 0
    pulled = usable & (((wb0 >= 0) & ~qb) | ((wa0 >= 0) & ~qa))
    return {"quoting both sides": (qb & qa).mean(), "quoting one side": (qb ^ qa).mean(), "quoting neither": (~qb & ~qa).mean(),
            "any side pulled by a rule": pulled.mean(), "fillable (on or inside the touch)": ((qb & (wb >= bb)) | (qa & (wa <= ba))).mean(),
            "no usable touch / price < 3% or > 97%": (~usable).mean()}
