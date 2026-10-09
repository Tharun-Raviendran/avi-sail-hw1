"""Part 4 helpers: normalize for latency. Pretend the feed reached us k seconds earlier (starter.asof shift_s=k) and redo
Part 2 (the blend weight the feed earns) and Part 3 (the lead/lag event study) at each k.

Nothing on the Kalshi side moves: the mid (Part 1), the touch target and the prints are untouched. Only the feed's
arrival time is moved earlier by k.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from starter import D, asof, beter_match_winner
from p1_mid import LAT, query_prints, predict_lasso3b, predict_lasso6, clustered_mean_se
from p2_feed import boot_matches
from p3_leadlag import run_study, summarize, offsets, SURFACE, INK, INK2, GRID, C_TEAM1, C_TEAM2

MAX_BOOK_AGE_S = 5.0          # same touch-target rule as Part 2


# =================================================================================================
# Part 2 at shift k: the 1 s comparison grids, exactly as in part2.ipynb
# =================================================================================================
def _grid(tr, predict, P):
    span = tr.groupby("match_id").t_ns.agg(["min", "max"]) / 1e9
    times = pd.concat([pd.DataFrame({"match_id": m, "t": np.arange(r["min"], r["max"], 1.0)}) for m, r in span.iterrows()],
                      ignore_index=True)
    G = query_prints(tr, times)                                # last print visible at t - LAT
    G["mid"] = predict(G, tr, P)                               # Part 1 winner from prints <= t - LAT
    G = G[["match_id", "t", "mid"]].copy()
    G["a_ns"] = ((G.t - LAT) * 1e9).astype("int64")           # information cutoff a = t - latency
    return G

def _touch_target(G, bb):
    """book_0: the recorded touch mid at t (fresh <= 5 s, not locked/crossed), NaN otherwise. TARGET only."""
    book = bb.assign(bmid=(bb.bid + bb.ask) / 2, uncrossed=bb.bid < bb.ask, book_t_ns=bb.t_ns)
    q = G.assign(_q=(G.t * 1e9).astype("int64"))
    r = asof(q, book, "_q", "t_ns", cols=["bmid", "uncrossed", "book_t_ns"])
    fresh = (q._q.values - r.book_t_ns.values) / 1e9 <= MAX_BOOK_AGE_S
    return np.where(r.uncrossed.fillna(False).astype(bool).values & fresh, r.bmid.values, np.nan)

def cs2_grid(ctx, P):
    """Part 2 CS2 grid: every in-play second of every mapped CS2 match, Model 3b mid, touch target."""
    G = _grid(ctx["tr"], predict_lasso3b, P)
    G["book_0"] = _touch_target(G, ctx["bb"])
    return G

def mlb_grid(ctx, P):
    G = _grid(ctx["tr"], predict_lasso6, P)
    G["book_0"] = _touch_target(G, ctx["bb"])
    return G

def cs2_feed():
    f = beter_match_winner(3); return f[f.line_type == 1][["match_id", "t_recv_ns", "p1", "st1"]]

def feed_at_k(G, feed, k, value, keep=None):
    """The feed value as of a + k (the feed delivered k s earlier). keep(df) -> bool mask of rows to keep."""
    f = asof(G, feed, "a_ns", "t_recv_ns", shift_s=k, cols=[c for c in feed.columns if c not in ("match_id", "t_recv_ns")])
    out = G.assign(feed=f[value].values)
    m = out.feed.notna().values & out.book_0.notna().values
    if keep is not None: m &= keep(f)
    out = out[m].copy(); out["gap"] = out.feed - out.mid
    return out

def blend_at_k(F, test_ids):
    """Part 2's blend: w = sum gap*(touch - mid) / sum gap^2 on TRAIN seconds; TEST RMSE of mid alone vs blend."""
    F = F.assign(test=F.match_id.isin(test_ids))
    tr_, te_ = F[~F.test], F[F.test]
    s = pd.DataFrame({"xy": tr_.gap * (tr_.book_0 - tr_.mid), "xx": tr_.gap ** 2, "g": tr_.match_id.values}).groupby("g").sum()
    fn = lambda v: v[0] / v[1]
    w = fn(s.values.sum(0)); w_se = boot_matches(s, fn, B=500)
    sq = lambda ww: 1e4 * (te_.mid + ww * te_.gap - te_.book_0) ** 2
    gain, gain_se = clustered_mean_se(sq(0) - sq(w), te_.match_id)
    r0, rw = np.sqrt(sq(0).mean()), np.sqrt(sq(w).mean())
    w_te = (te_.gap * (te_.book_0 - te_.mid)).sum() / (te_.gap ** 2).sum()
    return {"w_train": w, "w_train_se": w_se, "test_RMSE_mid_c": r0, "test_RMSE_blend_c": rw,
            "rmse_change_pct": 100 * (rw / r0 - 1), "gain_t": gain / gain_se if gain_se else np.nan, "w_test_refit": w_te,
            "train_seconds": len(tr_), "test_seconds": len(te_)}


# =================================================================================================
# Part 3 at shift k
# =================================================================================================
def lead_at_k(ev, ctx, k, W=60, H=60, B=400):
    """The Part 3 event study with every feed event received k s earlier."""
    e = ev.assign(t_recv_ns=ev.t_recv_ns - int(round(k * 1e9)))
    tau, edges = offsets(W, H), np.arange(-W, H + 1e-9, 2.0)
    s = summarize(run_study(e, tau, ctx["tape_fn"], ctx["touch_fn"], ctx["tr"], W, H, edges), B=B)
    t = s["touch"]
    return {"k": k, "t_half": t["t_half"], "t_half_lo": t["t_half_lo"], "t_half_hi": t["t_half_hi"],
            "in_touch_at_0": t["F0"], "in_touch_lo": t["F0_lo"], "in_touch_hi": t["F0_hi"], "events": s["events"]}

def crossing(x, y, level=0.0, rising=True):
    """First x at which y crosses `level` (linear interpolation). 0 if already past it at x[0]; NaN if never."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (y >= level) if rising else (y <= level)
    if ok[0]: return 0.0
    i = np.argmax(ok)
    if not ok[i]: return np.nan
    return x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])

def k_needed(L):
    """From one feature's lead-vs-k table: the k at which the feed stops lagging (t_half >= 0, i.e. the feed arrives
    before the touch has done half the move), with a CI from the bootstrap band, and the stricter k at which at most
    10% of the move is already in the touch at receipt."""
    return {"k_stop_lagging": crossing(L.k, L.t_half), "k_lo": crossing(L.k, L.t_half_hi), "k_hi": crossing(L.k, L.t_half_lo),
            "k_10pct": crossing(L.k, L.in_touch_at_0, 0.10, rising=False)}


# =================================================================================================
# charts
# =================================================================================================
def lead_vs_k_figure(LK, KN, net_delay, path=None):
    """Small multiples: Kalshi lead (= -t_half) vs k per feature, with CI band; the zero crossing is where the feed
    stops lagging. Shaded: the network delay we measured (what co-location could remove)."""
    keys = list(LK)
    n = len(keys); cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.3 * rows + 1.1), sharex=True)
    for ax, key in zip(axes.ravel(), keys):
        L = LK[key]; col = C_TEAM1 if key[0] == "CS2" else C_TEAM2
        ax.axvspan(0, net_delay, color=INK2, alpha=0.12, lw=0)
        ax.fill_between(L.k, -L.t_half_hi, -L.t_half_lo, color=col, alpha=0.18, lw=0)
        ax.plot(L.k, -L.t_half, color=col, lw=2)
        ax.axhline(0, color=INK, lw=0.9)
        kk = KN[key]["k_stop_lagging"]
        if np.isfinite(kk) and kk > 0:
            ax.plot([kk], [0], "o", color=INK, ms=6, zorder=4)
            right = kk > 5
            ax.annotate(f"stops lagging at k ≈ {kk:.1f} s", (kk, 0), xytext=(-6 if right else 6, 10), textcoords="offset points",
                        ha="right" if right else "left", fontsize=8.5)
        elif not np.isfinite(kk):
            ax.text(0.97, 0.06, f"still lagging at k = {L.k.max():g} s", transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)
        elif kk == 0:
            ax.text(0.97, 0.9, "already ahead at k = 0", transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)
        ax.set_title(f"{key[0]} · {key[1]}", fontsize=10, loc="left")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for ax in axes.ravel()[n:]: ax.axis("off")
    for ax in axes[:, 0]: ax.set_ylabel("Kalshi lead (s)\n+ = Kalshi first")
    for ax in axes[-1]: ax.set_xlabel("k: feed delivered k s earlier")
    fig.text(0.01, 0.985, "Part 4 · Kalshi's lead over the feed as the feed is delivered k seconds earlier (TEST)", fontsize=13,
             fontweight="bold", va="top")
    fig.text(0.01, 0.985 - 0.38 / (3.3 * rows + 1.1), "lead = −t½ (Part 3), 95% match-bootstrap band · grey strip = the network delay we "
             f"measured (median {net_delay:.2f} s): the most that co-location could remove", fontsize=9, color=INK2, va="top")
    fig.subplots_adjust(left=0.07, right=0.98, top=1 - 1.0 / (3.3 * rows + 1.1), bottom=0.6 / (3.3 * rows + 1.1) + 0.04, hspace=0.35, wspace=0.25)
    if path: fig.savefig(path, dpi=130)
    return fig

def blend_vs_k_figure(BK, plateaus, lags, net_delay, path=None):
    """Per sport: train blend weight vs k (±1.96 SE) and the TEST RMSE change of the blend vs the mid alone."""
    fig, axes = plt.subplots(2, len(BK), figsize=(6.2 * len(BK), 7), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    axes = np.atleast_2d(axes).reshape(2, len(BK))
    for j, (name, Bt) in enumerate(BK.items()):
        col = C_TEAM1 if name.startswith("CS2") else C_TEAM2
        a1, a2 = axes[0, j], axes[1, j]
        for a in (a1, a2):
            a.axvspan(0, net_delay, color=INK2, alpha=0.12, lw=0); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
        a1.fill_between(Bt.k, Bt.dw_lo, Bt.dw_hi, color=col, alpha=0.18, lw=0)
        a1.plot(Bt.k, Bt.dw, color=col, lw=2, marker="o", ms=3.5)
        a1.axhline(0, color=INK, lw=0.8)
        pk = plateaus[name]
        a1.axvline(pk["k_peak"], color=INK, lw=0.9, ls=(0, (3, 2)))
        right = pk["k_peak"] > 0.6 * Bt.k.max()
        a1.annotate(f"weight stops rising at k ≈ {pk['k_peak']:g} s\n(95% CI {pk['k_peak_lo']:g}–{pk['k_peak_hi']:g} s)\n"
                    f"w: {pk['w_at_0']:.4f} at k = 0 → {pk['w_peak']:.4f}", (pk["k_peak"], pk["rise"]),
                    xytext=(-10 if right else 10, -48), textcoords="offset points", ha="right" if right else "left", fontsize=8.5)
        if name in lags:
            a1.axvline(lags[name], color=col, lw=0.9, ls=(0, (1, 2)))
            a1.annotate(f"Part 3 lag {lags[name]:.1f} s ", (lags[name], 0), xytext=(0, 4), textcoords="offset points",
                        ha="right", fontsize=8, color=INK2)
        a1.set_title(name, loc="left", fontsize=11, fontweight="bold")
        a1.set_ylabel("change in blend weight\nsince k = 0 (TRAIN fit)")
        a2.plot(Bt.k, Bt.rmse_change_pct, color=col, lw=2, marker="o", ms=3.5)
        a2.axhline(0, color=INK, lw=0.8)
        a2.set_ylabel("TEST RMSE change vs mid alone (%)\n← blend better")
        a2.set_xlabel("k: feed delivered k s earlier")
    fig.text(0.01, 0.985, "Part 4 · Blend weight the feed earns as it is delivered k seconds earlier", fontsize=13, fontweight="bold", va="top")
    fig.text(0.01, 0.945, "target = Kalshi touch mid now; w fitted on train, band = 95% paired match-bootstrap CI of the change; RMSE on test · "
             f"grey strip = measured network delay ({net_delay:.2f} s)", fontsize=9, color=INK2, va="top")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.08, hspace=0.12, wspace=0.3)
    if path: fig.savefig(path, dpi=130)
    return fig

def k_needed_figure(T, net_delay, path=None, kmax=10):
    """Bar per feature: the k needed for the feed to stop lagging the touch (with CI), against the measured network delay."""
    T = T.iloc[::-1].reset_index(drop=True)
    h = 0.5 * len(T) + 1.9
    fig, ax = plt.subplots(figsize=(11, h))
    for i, r in T.iterrows():
        col = C_TEAM1 if r.sport == "CS2" else C_TEAM2
        v = r.k_stop_lagging
        if not np.isfinite(v):
            ax.barh(i, kmax, color=col, height=0.55, alpha=0.35, hatch="///", edgecolor=col, lw=0)
            ax.text(kmax + 0.15, i, f"> {kmax:g} s (never within the range tested)", va="center", fontsize=9)
            continue
        ax.barh(i, v, color=col, height=0.55)
        if np.isfinite(r.k_lo) and np.isfinite(r.k_hi) and v > 0:
            ax.plot([r.k_lo, r.k_hi], [i, i], color=INK, lw=1.2)
        ax.text((r.k_hi if np.isfinite(r.k_hi) and v > 0 else v) + 0.35, i,
                "already ahead at k = 0" if v == 0 else f"{v:.1f} s", va="center", fontsize=9)
    ax.axvline(net_delay, color=INK, lw=1, ls=(0, (3, 2)))
    ax.text(net_delay + 0.1, -0.55, f"measured network delay {net_delay:.2f} s: the most co-location can remove", fontsize=8.5, color=INK2, va="bottom")
    ax.set_xlim(0, kmax + 5)
    ax.set_yticks(range(len(T))); ax.set_yticklabels([f"{r.sport} · {r.feature}" for _, r in T.iterrows()])
    ax.set_xlabel("k needed for the feed to stop lagging the Kalshi touch (s)  ·  bar = point estimate, line = 95% CI")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(-0.75, len(T) - 0.4)
    fig.text(0.01, 1 - 0.25 / h, "How much earlier would each feed have to arrive?", fontsize=13, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.27, right=0.97, top=1 - 0.75 / h, bottom=0.75 / h)
    if path: fig.savefig(path, dpi=130)
    return fig


def blend_curve(Fk, test_ids, B=2000, seed=0):
    """Fk: {k: grid with feed shifted by k}. Returns (table per k, bootstrap summary of where the TRAIN weight peaks).
    The bootstrap resamples TRAIN matches once and reuses the same draw at every k (paired), so the shape of w(k)
    is estimated far more precisely than its level."""
    rows = {k: blend_at_k(F, test_ids) for k, F in Fk.items()}
    Bt = pd.DataFrame(rows).T.rename_axis("k").reset_index()
    ks = list(Fk)
    stats = []
    for k in ks:
        F = Fk[k]; tr_ = F[~F.match_id.isin(test_ids)]
        stats.append(pd.DataFrame({"xy": tr_.gap * (tr_.book_0 - tr_.mid), "xx": tr_.gap ** 2, "g": tr_.match_id.values}).groupby("g").sum())
    mids = sorted(set().union(*[s.index for s in stats]))
    XY = np.column_stack([s.xy.reindex(mids).fillna(0).values for s in stats])
    XX = np.column_stack([s.xx.reindex(mids).fillna(0).values for s in stats])
    rng = np.random.default_rng(seed)
    Wt = rng.multinomial(len(mids), np.ones(len(mids)) / len(mids), size=B).astype(float)
    wb = (Wt @ XY) / (Wt @ XX)                                     # B x k
    kb = np.array(ks)[wb.argmax(1)]
    rise = wb.max(1) - wb[:, 0]
    w0 = XY[:, 0].sum() / XX[:, 0].sum(); w = XY.sum(0) / XX.sum(0)
    dw = wb - wb[:, [0]]
    Bt["dw"], Bt["dw_lo"], Bt["dw_hi"] = w - w0, np.percentile(dw, 2.5, 0), np.percentile(dw, 97.5, 0)
    summary = {"k_peak": float(ks[int(np.argmax(w))]), "k_peak_lo": float(np.percentile(kb, 2.5)), "k_peak_hi": float(np.percentile(kb, 97.5)),
               "w_at_0": w0, "w_peak": w.max(), "rise": w.max() - w0, "rise_lo": float(np.percentile(rise, 2.5)), "rise_hi": float(np.percentile(rise, 97.5))}
    return Bt, summary
