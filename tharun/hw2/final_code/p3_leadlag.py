"""Part 3 (final): the lead/lag event study.

Clock: everything is on OUR collector clock. Feed events use `t_recv_ns` (when the message reached us, the only
feed time we can act on). Kalshi prints use `t_ns`, the exchange stamp, which reaches us about 6 ms later
(DATA.md), so it is the collector clock to within a few ms. x = minutes since the match went in play, where
"in play" is the same span `starter.in_play` uses (first live BETER match-winner row / first live Sportradar row).
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from starter import D, test_match_ids, beter_match_winner


SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_TEAM1, C_TEAM2, C_GREEN, C_PURPLE, C_GRAY = "#2a78d6", "#eb6834", "#1f9e6e", "#7a5bd6", "#8f8e89"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


# =================================================================================================
# in-play spans (the same definition as starter.in_play)
# =================================================================================================
def inplay_spans(sport):
    """DataFrame indexed by match_id with start / end (t_recv_ns) of the in-play span."""
    if sport == "esports":
        tr = pd.read_parquet(D + "beter_esports_trading.parquet"); mw = tr[(tr.interval == 1) & (tr.result_type == 7)]
        span = mw[mw.line_type == 1].groupby("match_id").t_recv_ns.agg(["min", "max"])
        done = mw[mw.st1 == 3].groupby("match_id").t_recv_ns.min()
        span["max"] = np.minimum(span["max"], done.reindex(span.index).fillna(np.inf))
    else:
        ev = pd.read_parquet(D + "sportradar_mlb_events.parquet")
        ev = ev[(ev.feedtype == "delta") & ev.matchstatus.notna() & ~ev.matchstatus.isin(["NOT_STARTED", "ENDED"])]
        span = ev.groupby("match_id").t_recv_ns.agg(["min", "max"])
    return span.rename(columns={"min": "start", "max": "end"}).astype("int64")


# =================================================================================================
# PART 3 EVENT STUDY: who knows first?
# =================================================================================================
# Every event carries a DIRECTION taken from the feed itself (never from Kalshi's own move), so a curve that rises
# before t = 0 means Kalshi priced the information before our feed delivered it, not a mechanical artefact.
# Moves are measured on the match axis (P(team1) / P(home)), signed by the event's direction, from the
# window start t = -W. "Fraction of the eventual move" at t = (sum of signed moves to t) / (sum of signed moves to +H),
# pooled over events (a ratio of sums), with a match-bootstrap CI.

def offsets(W, H, fine=5.0, df=0.25, dc=1.0):
    """Event-time grid (seconds): 1 s steps far from the event, 0.25 s steps inside +-fine s."""
    return np.unique(np.round(np.concatenate([np.arange(-W, -fine, dc), np.arange(-fine, fine, df),
                                              np.arange(fine, H + dc / 2, dc)]), 4))

def touch_mid_at(bbo, match_ids, t_s, max_stale_s=30.0):
    """Kalshi touch mid in force at each (match_id, t seconds), collector clock; NaN if no touch or the last change
    is older than max_stale_s (a recording gap, DATA.md)."""
    out = np.full(len(t_s), np.nan)
    by = bbo.groupby("match_id").indices
    q = pd.DataFrame({"m": match_ids, "t": t_s, "i": np.arange(len(t_s))})
    for m, g in q.groupby("m"):
        if m not in by: continue
        b = bbo.iloc[by[m]]; tb = b.t_ns.values / 1e9; mid = ((b.bid + b.ask) / 2).values
        k = np.searchsorted(tb, g.t.values, side="right") - 1
        ok = (k >= 0); k = np.clip(k, 0, None)
        ok &= (g.t.values - tb[k]) <= max_stale_s
        out[g.i.values[ok]] = mid[k[ok]]
    return out

def tape_mid_at(tr, match_ids, t_s, predict, params):
    """Part 1 winner (Model 3b / Model 6) from every print traded up to t (no extra latency: this measures what has
    TRADED by t). NaN before the first in-play print."""
    from p1_mid import query_prints
    q = pd.DataFrame({"match_id": match_ids, "t": t_s, "qi": np.arange(len(t_s))})
    Q = query_prints(tr, q, lat=0.0)
    Q["est"] = predict(Q, tr, params)
    out = np.full(len(t_s), np.nan); out[Q.qi.values] = Q.est.values
    return out

def event_matrix(ev, tau, mid_fn):
    """mid_fn(match_ids, t_s) at every (event, tau). Returns array (events x len(tau))."""
    t = (ev.t_ns.values[:, None] / 1e9 + tau[None, :]).ravel()
    m = np.repeat(ev.match_id.values, len(tau))
    return mid_fn(m, t).reshape(len(ev), len(tau))

def taker_matrix(tr, ev, edges):
    """Contracts traded in each [edges[k], edges[k+1]) bin around each event: total, and with the event's direction."""
    tot = np.zeros((len(ev), len(edges) - 1)); withm = np.zeros_like(tot)
    by = tr.groupby("match_id").indices
    for m, idx in ev.groupby("match_id").indices.items():
        if m not in by: continue
        g = tr.iloc[by[m]]; tp = g.t_ns.values / 1e9
        c_all = np.concatenate([[0], np.cumsum(g["count"].values)])
        c_up = np.concatenate([[0], np.cumsum(g["count"].values * (g["dir"].values > 0))])
        e = ev.iloc[idx]
        k = np.searchsorted(tp, e.t_ns.values[:, None] / 1e9 + edges[None, :], side="left")
        all_b, up_b = np.diff(c_all[k], axis=1), np.diff(c_up[k], axis=1)
        s = e.sign.values[:, None]
        tot[idx], withm[idx] = all_b, np.where(s > 0, up_b, all_b - up_b)
    return tot, withm


def run_study(ev, tau, tape_fn, touch_fn, tr, W, H, edges, clock="t_recv_ns"):
    """ev: match_id, t_recv_ns, t_prov_ns, sign. Aligns on `clock`. Keeps events whose tape AND touch are known over
    the whole window (same events on both lines). Returns a dict of per-match sums for bootstrapping."""
    e = ev.assign(t_ns=ev[clock].values).reset_index(drop=True)
    tape, touch = event_matrix(e, tau, tape_fn), event_matrix(e, tau, touch_fn)
    ok = np.isfinite(tape).all(1) & np.isfinite(touch).all(1)
    e, tape, touch = e[ok].reset_index(drop=True), tape[ok], touch[ok]
    s = e.sign.values[:, None]
    i0 = 0                                                   # tau[0] = -W: the baseline
    mv_tape, mv_touch = s * (tape - tape[:, [i0]]), s * (touch - touch[:, [i0]])
    tot, withm = taker_matrix(tr, e, edges)
    g = e.match_id.values
    agg = lambda X: pd.DataFrame(X).groupby(g).sum()
    return dict(events=e, tau=tau, edges=edges, H=H, W=W, n=pd.Series(1, index=g).groupby(level=0).sum(),
                tape=agg(mv_tape), touch=agg(mv_touch), tot=agg(tot), withm=agg(withm),
                move_ev_touch=mv_touch[:, np.searchsorted(tau, H)])

def _t_half(F, tau, i_H):
    """First event time at which F reaches 0.5 (linear interpolation); NaN if it never does by +H."""
    above = np.where(F[: i_H + 1] >= 0.5)[0]
    if not len(above): return np.nan
    k = above[0]
    if k == 0: return tau[0]
    return tau[k - 1] + (0.5 - F[k - 1]) * (tau[k] - tau[k - 1]) / (F[k] - F[k - 1])

def summarize(R, B=1000, seed=0):
    """Pooled curves + match-bootstrap 95% bands and the headline numbers."""
    rng = np.random.default_rng(seed)
    tau, i_H, i_0 = R["tau"], np.searchsorted(R["tau"], R["H"]), np.searchsorted(R["tau"], 0.0)
    mids = R["n"].index
    Wt = rng.multinomial(len(mids), np.ones(len(mids)) / len(mids), size=B).astype(float)   # B x matches
    out = {}
    for line in ("tape", "touch"):
        S = R[line].loc[mids].values                                   # matches x tau (summed signed moves)
        F = S.sum(0) / S[:, i_H].sum()
        Sb = Wt @ S; Fb = Sb / Sb[:, [i_H]]
        th = np.array([_t_half(f, tau, i_H) for f in Fb])
        out[line] = dict(F=F, lo=np.nanpercentile(Fb, 2.5, 0), hi=np.nanpercentile(Fb, 97.5, 0),
                         t_half=_t_half(F, tau, i_H), t_half_lo=np.nanpercentile(th, 2.5), t_half_hi=np.nanpercentile(th, 97.5),
                         t_half_nan=np.isnan(th).mean(), F0=F[i_0], F0_lo=np.percentile(Fb[:, i_0], 2.5), F0_hi=np.percentile(Fb[:, i_0], 97.5))
    # eventual move (touch), cents, SE clustered by match
    n = R["n"].loc[mids].values.astype(float); Sm = R["touch"].loc[mids].values[:, i_H]
    mean = Sm.sum() / n.sum(); se = np.sqrt(((Sm - mean * n) ** 2).sum()) / n.sum()
    out["move_c"], out["move_se_c"] = 100 * mean, 100 * se
    # takers with the move: contract-weighted share per bin
    T, Wm = R["tot"].loc[mids].values, R["withm"].loc[mids].values
    share = Wm.sum(0) / np.maximum(T.sum(0), 1e-12)
    sb = (Wt @ Wm) / np.maximum(Wt @ T, 1e-12)
    out["taker"] = dict(share=share, lo=np.percentile(sb, 2.5, 0), hi=np.percentile(sb, 97.5, 0), contracts=T.sum(0))
    ed = R["edges"]; mid_e = (ed[:-1] + ed[1:]) / 2
    for name, sel in [("before", (mid_e < 0) & (mid_e > -10)), ("after", (mid_e > 0) & (mid_e < 10))]:
        a, b = Wm[:, sel].sum(1), T[:, sel].sum(1)
        p = a.sum() / b.sum(); pb = (Wt @ a) / (Wt @ b)
        out[f"taker_{name}"], out[f"taker_{name}_lo"], out[f"taker_{name}_hi"] = p, np.percentile(pb, 2.5), np.percentile(pb, 97.5)
    out["events"], out["matches"] = int(n.sum()), len(mids)
    return out


# ---- feature events, each with a feed-side direction (+1 = good for team1 / home) ----------------------------------
def _declustered(e, gap_s):
    """Keep an event only if the previous event of the same feature in the same match is >= gap_s earlier."""
    e = e.sort_values(["match_id", "t_recv_ns"], kind="stable")
    prev = e.groupby("match_id").t_recv_ns.shift()
    return e[(e.t_recv_ns - prev).isna() | ((e.t_recv_ns - prev) >= gap_s * 1e9)]

def beter_jumps(match_ids, which, theta, gap_s=10.0):
    """BETER price moves of at least `theta` (probability points) on the live, open market, first of each cluster.
    which = 'match' (match winner, interval 1 / result_type 7) or 'map' (winner of the map being played,
    interval 10 + N / result_type 8; N from the series score in real-time incidents). Provider clock = state_ts_ms."""
    trd = pd.read_parquet(D + "beter_esports_trading.parquet")
    trd = trd[(trd.sport_id == 3) & (trd.line_type == 1) & trd.match_id.isin(match_ids)]
    trd = trd.sort_values("t_recv_ns", kind="stable").drop_duplicates(["match_id", "interval", "result_type", "offset"])
    if which == "match":
        x = trd[(trd.interval == 1) & (trd.result_type == 7)]
    else:
        x = trd[(trd.result_type == 8) & trd.interval.between(11, 15)].copy()
        inc = pd.read_parquet(D + "beter_esports_incident.parquet")
        inc = inc[(inc.sport_id == 3) & (inc.msg_type == 1) & inc.map_score.notna() & inc.match_id.isin(match_ids)].copy()
        inc["map_no"] = inc.map_score.str.split(":", expand=True).astype(float).sum(axis=1) + 1
        from starter import asof
        x["map_no"] = asof(x, inc, "t_recv_ns", "t_recv_ns", cols=["map_no"]).map_no.values
        x = x[x.interval == 10 + x.map_no]
    x = x[x.st1 == 1].copy()
    x["dp"] = x.groupby(["match_id", "interval"]).p1.diff()
    j = x[x.dp.abs() >= theta]
    j = _declustered(j, gap_s)
    return pd.DataFrame({"match_id": j.match_id.values, "t_recv_ns": j.t_recv_ns.values,
                         "t_prov_ns": (j.state_ts_ms.values * 1e6).astype("int64"), "sign": np.sign(j.dp.values).astype(int),
                         "size": j.dp.abs().values})

def round_won(match_ids):
    """BETER CS2 RoundWon (type 9), real-time rows. Direction = the round winner. Provider clock = `date`."""
    inc = pd.read_parquet(D + "beter_esports_incident.parquet")
    r = inc[(inc.sport_id == 3) & (inc.msg_type == 1) & (inc.type == "9") & inc.match_id.isin(match_ids)
            & inc.participant.isin(["1", "2"])]
    e = pd.DataFrame({"match_id": r.match_id.values, "t_recv_ns": r.t_recv_ns.values,
                      "t_prov_ns": pd.to_datetime(r.date, utc=True, format="ISO8601").astype("int64").values,
                      "sign": np.where(r.participant.values == "1", 1, -1)})
    return _declustered(e, 10.0)          # ~5% of rounds are re-sent within seconds (corrections): keep the first

def mlb_feature_events(ev, match_ids, we_states, run_gap_s=30.0, min_dwe=0.005):
    """Sportradar real-time (delta) events with a direction (+1 = good for home):
       run      1720, the scoring side; first run of a multi-run play only
       betstop  1011, the BATTING team (a Betstop says 'a run may be coming'); `before_run` flags those followed by a
                run before the next Betstart
       bip      1031 ball in play; pitch 2327 release: the sign of the change in win expectancy (our Part 2 structural
                model) from the state at the event to the state just before the NEXT pitch: what the play turned out
                to be, according to the feed. Plays that leave win expectancy unchanged (< min_dwe) are dropped.
    we_states: match_id, t_recv_ns, we (one row per Sportradar state change)."""
    from starter import asof
    d = ev[(ev.feedtype == "delta") & ev.match_id.isin(match_ids)].sort_values(["match_id", "t_recv_ns"], kind="stable").copy()
    d["half"] = d.groupby("match_id").inninghalf.ffill()
    d["t_prov_ns"] = (pd.to_numeric(d.stime, errors="coerce") * 1e6).astype("int64")
    base = lambda x, s: pd.DataFrame({"match_id": x.match_id.values, "t_recv_ns": x.t_recv_ns.values,
                                      "t_prov_ns": x.t_prov_ns.values, "sign": s})
    out = {}
    runs = d[(d.type == "1720") & d.side.isin(["home", "away"])]
    run_ev = base(runs, np.where(runs.side.values == "home", 1, -1))
    out["run"] = _declustered(run_ev, run_gap_s)
    bs = d[(d.type == "1011") & d.half.isin(["T", "B"])]
    b = base(bs, np.where(bs.half.values == "B", 1, -1))
    # followed by a run before the next Betstart?
    nxt = lambda kind: asof(b.assign(t=-b.t_recv_ns), d[d.type == kind].assign(t=-d[d.type == kind].t_recv_ns, tk=d[d.type == kind].t_recv_ns), "t", "t", cols=["tk"]).tk.values
    next_run, next_start = nxt("1720"), nxt("1010")
    b["before_run"] = np.isfinite(next_run) & ((next_run < next_start) | ~np.isfinite(next_start))
    out["betstop_all"], out["betstop_run"] = b, b[b.before_run]
    pitches = d[d.type == "2327"][["match_id", "t_recv_ns"]].rename(columns={"t_recv_ns": "tp"})
    for name, kind in [("pitch", "2327"), ("bip", "1031")]:
        x = d[d.type == kind]
        e = base(x, 0)
        # the next pitch release strictly after the event (search backwards in negated time = forward asof)
        q = e.assign(t=-(e.t_recv_ns + 1))
        nx = asof(q, pitches.assign(t=-pitches.tp), "t", "t", cols=["tp"]).tp.values
        t_after = np.where(np.isfinite(nx), nx - 1, e.t_recv_ns + 300e9).astype("int64")
        t_after = np.minimum(t_after, e.t_recv_ns.values + int(300e9))
        we0 = asof(e, we_states, "t_recv_ns", "t_recv_ns", cols=["we"]).we.values
        we1 = asof(e.assign(ta=t_after), we_states, "ta", "t_recv_ns", cols=["we"]).we.values
        dwe = we1 - we0
        e["sign"], e["dwe"] = np.sign(dwe), dwe
        out[name] = e[np.abs(dwe) >= min_dwe].astype({"sign": int})
    return out


# ---- running one feature on both clocks ----------------------------------------------------------------------------
def study(ev, ctx, W, H, step_taker=2.0, B=1000):
    """Event study for one feature on our receipt clock and on the provider's own clock.
    ctx: dict(tr, tape_fn, touch_fn). Returns dict(recv, prov, delay_s, tau, edges)."""
    tau, edges = offsets(W, H), np.arange(-W, H + 1e-9, step_taker)
    out = {"tau": tau, "edges": edges, "W": W, "H": H}
    for clock, key in [("t_recv_ns", "recv"), ("t_prov_ns", "prov")]:
        R = run_study(ev, tau, ctx["tape_fn"], ctx["touch_fn"], ctx["tr"], W, H, edges, clock=clock)
        out[key] = summarize(R, B=B); out[key + "_events"] = R["events"]
    e = out["recv_events"]; out["delay_s"] = (e.t_recv_ns - e.t_prov_ns) / 1e9      # network delay, per event
    return out


# ---- charts ----------------------------------------------------------------------------------------------------------
C_TOUCH, C_TAPE, C_TAKER = C_TEAM1, C_TEAM2, C_PURPLE

def feature_figure(res, title, direction, path=None, xlim=(-60, 60)):
    """Top: share of the eventual Kalshi move already done (touch, tape; touch on the feed's own clock dashed).
    Bottom: share of taker contracts trading WITH the event's direction. Both with 95% match-bootstrap bands."""
    tau, ed = res["tau"], res["edges"]; r, p = res["recv"], res["prov"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True, gridspec_kw=dict(height_ratios=[2.3, 1], hspace=0.08))
    for line, col, lab in [("tape", C_TAPE, "tape: Part 1 mid from prints"), ("touch", C_TOUCH, "touch: Kalshi bid/ask mid")]:
        d = r[line]
        a1.fill_between(tau, d["lo"], d["hi"], color=col, alpha=0.15, lw=0)
        a1.plot(tau, d["F"], color=col, lw=2, label=lab)
    a1.plot(tau, p["touch"]["F"], color=C_TOUCH, lw=1.1, ls=(0, (3, 2)), label="touch, aligned on the feed's own timestamp")
    for a in (a1, a2):
        a.axvline(0, color=INK, lw=1); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    a1.axhline(0.5, color=INK2, lw=0.8, ls=(0, (4, 3))); a1.axhline(0, color=INK2, lw=0.6); a1.axhline(1, color=INK2, lw=0.6)
    th, lo, hi = r["touch"]["t_half"], r["touch"]["t_half_lo"], r["touch"]["t_half_hi"]
    if np.isfinite(th):
        a1.plot([th], [0.5], "o", color=INK, ms=6, zorder=5)
        a1.annotate(f"half the move in the touch at {th:+.1f} s\n(95% CI {lo:+.1f} to {hi:+.1f})", (th, 0.5), xytext=(-12, 22),
                    textcoords="offset points", ha="right", fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    f0 = r["touch"]
    a1.annotate(f"{f0['F0']:.0%} already in the touch\nwhen the feed reaches us\n(95% CI {f0['F0_lo']:.0%}–{f0['F0_hi']:.0%})",
                (0, f0["F0"]), xytext=(14, -38), textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a1.plot([0], [f0["F0"]], "o", color=C_TOUCH, ms=6, zorder=5, markeredgecolor=SURFACE)
    a1.set_ylim(-0.2, 1.25); a1.set_ylabel("share of the eventual\nKalshi move already done")
    a1.legend(loc="upper left", fontsize=8.5, frameon=False)
    a1.text(0, 1.2, " ← before we receive it | after →", fontsize=8.5, color=INK2, ha="center", va="top")
    t = r["taker"]; x = (ed[:-1] + ed[1:]) / 2
    a2.fill_between(x, t["lo"], t["hi"], color=C_TAKER, alpha=0.15, lw=0, step="mid")
    a2.step(x, t["share"], where="mid", color=C_TAKER, lw=1.6)
    a2.axhline(0.5, color=INK2, lw=0.8, ls=(0, (4, 3))); a2.text(xlim[1] - 0.5, 0.505, "50% = no edge", ha="right", va="bottom", fontsize=8, color=INK2)
    a2.set_ylim(0.15, 1.0); a2.set_ylabel(f"takers WITH the move\n(share of contracts, {ed[1]-ed[0]:g} s bins)")
    a2.set_xlabel("seconds relative to the moment OUR collector received the feed event (t = 0)")
    a2.set_xlim(*xlim)
    fig.text(0.075, 0.975, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.075, 0.94, f"TEST · {r['events']:,} events in {r['matches']} matches · direction = {direction} · "
             f"eventual move (touch, −{res['W']} s → +{res['H']} s) {r['move_c']:+.2f}c ± {1.96*r['move_se_c']:.2f}",
             fontsize=9, color=INK2, va="top")
    fig.subplots_adjust(left=0.1, right=0.93, top=0.89, bottom=0.08)
    if path: fig.savefig(path, dpi=130)
    return fig

def lead_table(results, train_results=None):
    """One row per feature, ranked by when the Kalshi touch had done half the move (most negative = Kalshi first)."""
    rows = []
    for (sport, name), res in results.items():
        r, p, d = res["recv"], res["prov"], res["delay_s"]
        row = {"sport": sport, "feature": name, "events": r["events"], "matches": r["matches"],
               "move_c": round(r["move_c"], 2), "move_ci_c": round(1.96 * r["move_se_c"], 2),
               "t_half_touch_s": r["touch"]["t_half"], "t_half_ci": f"[{r['touch']['t_half_lo']:+.1f}, {r['touch']['t_half_hi']:+.1f}]",
               "t_half_tape_s": r["tape"]["t_half"], "in_touch_at_0": r["touch"]["F0"], "in_tape_at_0": r["tape"]["F0"],
               "takers_with_10s_before": r["taker_before"], "takers_with_10s_after": r["taker_after"],
               "t_half_touch_feed_clock_s": p["touch"]["t_half"],
               "net_delay_med_s": d.median(), "net_delay_p90_s": d.quantile(0.9)}
        if train_results is not None: row["t_half_touch_TRAIN_s"] = train_results[(sport, name)]["recv"]["touch"]["t_half"]
        rows.append(row)
    T = pd.DataFrame(rows).sort_values("t_half_touch_s").reset_index(drop=True)
    T.insert(2, "kalshi_lead_s", -T.t_half_touch_s)
    return T

def ranking_figure(T, path=None):
    """Forest plot: when the Kalshi touch had done half the move, per feature, receipt clock (filled) and the feed's own
    clock (hollow). Left of 0 = Kalshi got there before our feed. Right-hand column: share already in the touch at receipt."""
    T = T.iloc[::-1].reset_index(drop=True)
    h = 0.55 * len(T) + 2.0
    fig, ax = plt.subplots(figsize=(12, h))
    for i, r in T.iterrows():
        col = C_TOUCH if r.sport == "CS2" else C_TAPE
        lo, hi = [float(v) for v in r.t_half_ci.strip("[]").split(",")]
        ax.plot([lo, hi], [i, i], color=col, lw=2.4, solid_capstyle="round")
        ax.plot(r.t_half_touch_s, i, "o", color=col, ms=9, markeredgecolor=SURFACE, zorder=3)
        ax.plot(r.t_half_touch_feed_clock_s, i, "o", ms=7, markerfacecolor="none", markeredgecolor=INK, mew=1, zorder=4)
        ax.text(1.02, i, f"{r.t_half_touch_s:+.1f} s", transform=ax.get_yaxis_transform(), va="center", fontsize=9.5, fontweight="bold")
        ax.text(1.10, i, f"{r.in_touch_at_0:.0%}", transform=ax.get_yaxis_transform(), va="center", fontsize=9.5)
    ax.text(1.02, len(T) - 0.3, "t½", transform=ax.get_yaxis_transform(), fontsize=8.5, color=INK2, va="bottom")
    ax.text(1.10, len(T) - 0.3, "in touch\nat receipt", transform=ax.get_yaxis_transform(), fontsize=8.5, color=INK2, va="bottom")
    ax.set_yticks(np.arange(len(T))); ax.set_yticklabels([f"{r.sport} · {r.feature}" for _, r in T.iterrows()])
    ax.axvline(0, color=INK, lw=1); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    lows = [float(v.strip("[]").split(",")[0]) for v in T.t_half_ci]; highs = [float(v.strip("[]").split(",")[1]) for v in T.t_half_ci]
    ax.set_xlim(min(lows) - 1, max(highs) + 1.5)
    ax.text(-0.3, len(T) - 0.45, "← Kalshi moved first", ha="right", fontsize=9, color=INK2)
    ax.text(0.3, len(T) - 0.45, "feed first →", ha="left", fontsize=9, color=INK2)
    ax.set_ylim(-0.6, len(T) - 0.2)
    ax.set_xlabel("t½: when the Kalshi touch had done HALF of the eventual move, seconds relative to our receipt of the feed")
    fig.text(0.01, 1 - 0.25 / h, "Who knows first? Lead/lag by feed feature (TEST)", fontsize=13, fontweight="bold", va="top")
    fig.text(0.01, 1 - 0.6 / h, "dot + bar: our receipt clock, 95% match-bootstrap CI · hollow ring: aligned on the feed's own timestamp · "
             "blue = CS2 (BETER), orange = MLB (Sportradar)", fontsize=8.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.24, right=0.84, top=1 - 1.1 / h, bottom=0.75 / h)
    if path: fig.savefig(path, dpi=130)
    return fig

# ---- context per sport: in-play prints with the Part 1 estimators, the book_ok touch, mid functions ----------------
def sport_context(sport, P):
    """Everything Parts 3-6 need for 'CS2' or 'MLB': every mapped in-play print with the Part 1 estimators (P = the
    sport's Part 1 params), the book_ok touch, and train / test ids restricted to book_ok matches, so tape and touch
    are always computed on the same matches."""
    from p1_mid import add_estimators, predict_lasso3b, predict_lasso6
    if sport == "CS2":
        mp = pd.read_parquet(D + "map_esports.parquet"); ids = set(mp[mp.sport_id == 3].beter_match_id)
        test = test_match_ids("esports") & ids; from starter import trades_on_axis; tr = trades_on_axis("esports"); pred = predict_lasso3b
    else:
        ids = set(pd.read_parquet(D + "map_mlb.parquet").sr_match_id); test = test_match_ids("mlb") & ids
        from starter import trades_on_axis; tr = trades_on_axis("mlb"); pred = predict_lasso6
    tr = tr[tr.match_id.isin(ids) & tr.inplay].reset_index(drop=True)
    add_estimators(tr, P["H"], P["W"])
    bbo = pd.read_parquet(D + "kalshi_bbo.parquet"); bq = pd.read_parquet(D + "book_quality.parquet")
    bb = bbo[(bbo.sport == sport) & bbo.event_ticker.isin(bq.loc[bq.book_ok, "event_ticker"])].sort_values("t_ns").reset_index(drop=True)
    book = set(bb.match_id) & set(tr.match_id)
    return dict(sport=sport, tr=tr, bb=bb, train=book - test, test=book & test,
                tape_fn=lambda m, t: tape_mid_at(tr, m, t, pred, P), touch_fn=lambda m, t: touch_mid_at(bb, m, t))
