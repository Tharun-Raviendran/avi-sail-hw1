"""Part 2 helpers: the MLB win-probability model (Sportradar game state -> P(home wins)) and the
feed-vs-mid analysis (comparison, box plots, blend, when to stop trusting the feed) for any sport.

Analysis functions take a frame F with one row per in-play second and columns
    match_id, test (bool), feed, mid, gap (= feed - mid), since_print_s, feed_age_s, book_0 / book_30 / book_120
(the Kalshi touch mid now / later: TARGETS only), plus whatever bucket columns are passed as `by`.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from p1_mid import clustered_mean_se
from starter import D as D_DATA

# ---- chart style (validated palette, light surface) --------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BOX, POOLED, NONE_C = "#2a78d6", "#eb6834", "#8f8e89"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(0)


# =================================================================================================
# MLB win-probability model
# =================================================================================================
STATE_COLS = ["inning", "half", "outs", "b1", "b2", "b3", "home", "away"]
WP_FEATS = ["we", "diff", "inning", "bottom", "outs", "b1", "b2", "b3"]
WP_MONO = tuple(1 if c in ("we", "diff") else 0 for c in WP_FEATS)   # P(home) must rise with we and the home lead
KMAX = 12                                                            # runs capped at 12 in the run distributions

def mlb_state_changes(ev, match_ids):
    """Game state after every Sportradar change, from REAL-TIME rows only (feedtype 'delta'), keyed on t_recv_ns.
    score from matchscore ('home:away'); inning / half forward-filled; outs forward-filled within the half-inning;
    bases only on 1716 'Play over' rows, forward-filled within the half-inning; a new half starts at 0 outs, bases empty.
    Returns one row per change of (inning, half, outs, bases, score), live rows only."""
    d = ev[(ev.feedtype == "delta") & ev.match_id.isin(match_ids)].sort_values(["match_id", "t_recv_ns"], kind="stable").copy()
    sc = d.matchscore.str.split(":", expand=True)
    d["home"], d["away"] = pd.to_numeric(sc[0], errors="coerce"), pd.to_numeric(sc[1], errors="coerce")
    d["inning"], d["half"] = pd.to_numeric(d.periodnumber, errors="coerce"), d.inninghalf
    g = d.groupby("match_id")
    for c in ["home", "away", "inning", "half"]: d[c] = g[c].ffill()
    d["outs"] = pd.to_numeric(d.outs, errors="coerce")
    play_over = d.type == "1716"
    for b, col in [("b1", "firstbaseloaded"), ("b2", "secondbaseloaded"), ("b3", "thirdbaseloaded")]:
        d[b] = np.where(play_over, pd.to_numeric(d[col], errors="coerce"), np.nan)
    half_key = [d.match_id, d.inning, d.half]
    for c in ["outs", "b1", "b2", "b3"]: d[c] = d.groupby(half_key, dropna=False)[c].ffill()
    d["outs"] = d.outs.fillna(0).clip(0, 3); d[["b1", "b2", "b3"]] = d[["b1", "b2", "b3"]].fillna(0)
    live = d.matchstatus.notna() & ~d.matchstatus.isin(["NOT_STARTED", "ENDED"])
    d = d[live & d.inning.notna() & d.half.isin(["T", "B"]) & d.home.notna()]
    d = d[["match_id", "t_recv_ns", "type"] + STATE_COLS].reset_index(drop=True)
    changed = (d[STATE_COLS] != d.groupby("match_id")[STATE_COLS].shift()).any(axis=1)
    S = d[changed].reset_index(drop=True)
    S["base_state"] = (S.b1 + 2 * S.b2 + 4 * S.b3).astype(int)
    S["diff"] = S.home - S.away
    return S

def _dist(x):
    c = np.bincount(np.clip(np.asarray(x, int), 0, KMAX), minlength=KMAX + 1).astype(float)
    return c / c.sum()

def fit_run_distributions(S_train):
    """From TRAIN games: runs the batting team scores (a) from each (outs, bases) state to the end of that half-inning
    and (b) in a full half-inning. Returns {'rest': {(outs, base_state): pmf}, 'full': pmf, 're': expected runs}."""
    S = S_train.copy()
    S["bat_runs"] = np.where(S.half == "B", S.home, S.away)
    S["runs_after"] = S.groupby(["match_id", "inning", "half"]).bat_runs.transform("last") - S.bat_runs
    live = S[S.outs < 3]
    rest = {k: _dist(g.runs_after) for k, g in live.groupby(["outs", "base_state"])}
    full = _dist(S.groupby(["match_id", "inning", "half"]).head(1).runs_after)
    re = live.groupby(["outs", "base_state"]).runs_after.mean()
    return {"rest": rest, "full": full, "re": re}

def _conv_pow(p, n):
    out = np.array([1.0])
    for _ in range(n): out = np.convolve(out, p)
    return out

def win_expectancy(inning, half, outs, base_state, diff, dists):
    """P(home wins) if both teams score like the TRAIN average from here on: the rest of the current half-inning
    from its (outs, bases) state, then independent full half-innings to the end of the 9th. A tie after regulation
    counts 1/2 (extra innings as a coin flip). diff = home - away runs now."""
    after = max(9 - int(inning), 0)
    rest = dists["rest"].get((int(outs), int(base_state)), np.array([1.0])) if outs < 3 else np.array([1.0])
    full = dists["full"]
    if half == "T": A, H = np.convolve(rest, _conv_pow(full, after)), _conv_pow(full, after + 1)
    else:           A, H = _conv_pow(full, after), np.convolve(rest, _conv_pow(full, after))
    d = np.convolve(H, A[::-1]); final = np.arange(len(d)) - (len(A) - 1) + diff       # final home - away margin
    return d[final > 0].sum() + 0.5 * d[final == 0].sum()

def wp_features(S, dists):
    """WP_FEATS for every state row: structural win expectancy `we` plus the raw state."""
    key = ["inning", "half", "outs", "base_state", "diff"]
    u = S[key].drop_duplicates()
    u["we"] = [win_expectancy(min(r.inning, 12), r.half, r.outs, r.base_state, r.diff, dists) for r in u.itertuples()]
    F = S[key].merge(u, on=key, how="left")
    F["bottom"] = (S.half.values == "B").astype(float)
    F["inning"] = S.inning.clip(upper=10).values
    for b in ["outs", "b1", "b2", "b3"]: F[b] = S[b].values
    F.index = S.index
    return F[WP_FEATS]

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

def predict_wp(model, F):
    """XGBoost boosted from the structural win expectancy: base_margin = logit(we), trees add corrections."""
    return model.predict_proba(F[WP_FEATS].values, base_margin=logit(F.we.values))[:, 1]

def betstop_intervals(ev, match_ids):
    """Sportradar Betstop (1011) / Betstart (1010) steps from real-time rows: t_recv_ns, match_id, betstop (bool)."""
    e = ev[(ev.feedtype == "delta") & ev.match_id.isin(match_ids) & ev.type.isin(["1010", "1011"])]
    return e.assign(betstop=e.type == "1011")[["match_id", "t_recv_ns", "betstop"]].sort_values("t_recv_ns")


# =================================================================================================
# Feed vs mid: comparison
# =================================================================================================
def boot_matches(stats, fn, B=1000):
    """Cluster bootstrap: stats has one row of summed sufficient statistics per match; resample matches."""
    k = rng.multinomial(len(stats), np.ones(len(stats)) / len(stats), size=B)
    return np.std([fn((stats.values * w[:, None]).sum(0)) for w in k])

def slope(d):
    """OLS slope of mid on feed and its match-bootstrap SE."""
    s = pd.DataFrame({"n": 1.0, "f": d.feed, "m": d.mid, "fm": d.feed * d.mid, "ff": d.feed ** 2,
                      "g": d.match_id.values}).groupby("g").sum()
    fn = lambda v: (v[3] - v[1] * v[2] / v[0]) / (v[4] - v[1] ** 2 / v[0])
    return fn(s.values.sum(0)), boot_matches(s, fn)

def compare(d):
    b, b_se = clustered_mean_se(d.gap, d.match_id)
    mse, _ = clustered_mean_se(d.gap ** 2, d.match_id)
    b1, b1_se = slope(d)
    return {"bias_c": 100 * b, "bias_se_c": 100 * b_se, "RMSE_c": 100 * np.sqrt(mse),
            "slope": b1, "slope_se": b1_se, "matches": d.match_id.nunique(), "seconds": len(d)}

def compare_by(T, by):
    return pd.DataFrame({k: compare(d) for k, d in T.groupby(by, observed=True)}).T

def bias_by_binning(T, levels):
    """Bias by price bin under three binning choices (robustness): (f+m)/2, mid, feed."""
    rob = {}
    for name, x in [("bin on (f+m)/2", (T.feed + T.mid) / 2), ("bin on mid", T.mid), ("bin on feed", T.feed)]:
        rob[name] = T.gap.groupby(pd.cut(x, levels, right=False), observed=True).mean() * 100
    return pd.DataFrame(rob)


# ---- box plots ----------------------------------------------------------------------------------
MIN_SECONDS, MIN_FEED_SD = 120, 0.005

def per_match(T, by):
    rows = []
    for (k, m), d in T.groupby([by, "match_id"], observed=True):
        if len(d) < MIN_SECONDS: continue
        b1 = np.cov(d.feed, d.mid)[0, 1] / d.feed.var() if d.feed.std() >= MIN_FEED_SD else np.nan
        rows.append({"bucket": k, "match_id": m, "bias_c": 100 * d.gap.mean(), "slope": b1})
    return pd.DataFrame(rows)

def _box_panel(ax, pm, stat, pooled, se, order, ref, ylabel):
    data = [pm.loc[(pm.bucket == k) & pm[stat].notna(), stat].values for k in order]
    bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True, whis=1.5,
                    medianprops={"color": INK, "linewidth": 2},
                    boxprops={"facecolor": BOX + "33", "edgecolor": BOX, "linewidth": 1.5},
                    whiskerprops={"color": BOX, "linewidth": 1.5}, capprops={"color": BOX, "linewidth": 1.5})
    x = np.arange(1, len(order) + 1) + 0.33
    ax.errorbar(x, [pooled[k] for k in order], yerr=[1.96 * se[k] for k in order], fmt="D", ms=6, color=POOLED,
                mec=SURFACE, mew=1.2, ecolor=POOLED, elinewidth=2, capsize=4, zorder=3,
                label="pooled estimate ± 95% CI (SE clustered by match)")
    ax.axhline(ref, color=INK2, linestyle="--", linewidth=1, zorder=0)
    lo = min(np.min(w.get_ydata()) for w in bp["whiskers"]); hi = max(np.max(w.get_ydata()) for w in bp["whiskers"])
    ci_lo = [pooled[k] - 1.96 * se[k] for k in order]; ci_hi = [pooled[k] + 1.96 * se[k] for k in order]
    lo, hi = min(lo, ref, *ci_lo), max(hi, ref, *ci_hi); pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{k}\nn={len(d)}" for k, d in zip(order, data)])
    ax.set_ylabel(ylabel); ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)

def bias_slope_figure(F, by, title, xlabel, path, subtitle):
    T = F[F.test & F[by].notna()]
    cats = T[by].cat.categories if hasattr(T[by], "cat") else sorted(T[by].unique())
    order = [k for k in cats if (T[by] == k).any()]
    pm = per_match(T, by)
    pooled = {k: compare(T[T[by] == k]) for k in order}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    _box_panel(a1, pm, "bias_c", {k: v["bias_c"] for k, v in pooled.items()}, {k: v["bias_se_c"] for k, v in pooled.items()},
               order, 0.0, "bias: feed − mid (cents)")
    _box_panel(a2, pm, "slope", {k: v["slope"] for k, v in pooled.items()}, {k: v["slope_se"] for k, v in pooled.items()},
               order, 1.0, "slope of mid on feed")
    fig.legend(*a1.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.995, 0.925), frameon=False)
    a2.set_xlabel(xlabel)
    a2.set_xticklabels([f"{k}\nn={len(pm[(pm.bucket == k) & pm.slope.notna()])}" for k in order])
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.935, subtitle, color=INK2, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(path, dpi=150); plt.show()


# =================================================================================================
# Blend and when to stop trusting the feed
# =================================================================================================
def w_stats(d, target):
    d = d[d[target].notna()]
    return pd.DataFrame({"xy": d.gap * (d[target] - d.mid), "xx": d.gap ** 2, "g": d.match_id.values}).groupby("g").sum()

def w_fit(d, target):
    """Blend weight w in  target - mid = w (feed - mid): closed-form least squares, match-bootstrap SE."""
    s = w_stats(d, target); fn = lambda v: v[0] / v[1]
    return fn(s.values.sum(0)), boot_matches(s, fn), len(s)

def blend_table(F, horizons=(0, 30, 120)):
    """w fitted on TRAIN; errors on TEST (mid only, blend at the train w, feed only); paired gain; w refit on test."""
    rows = []
    for h in horizons:
        target = f"book_{h}"
        w_tr, w_tr_se, n_tr = w_fit(F[~F.test], target)
        w_te, w_te_se, n_te = w_fit(F[F.test], target)
        d = F[F.test & F[target].notna()]
        sq = lambda w: 1e4 * (d.mid + w * d.gap - d[target]) ** 2
        gain, gain_se = clustered_mean_se(sq(0) - sq(w_tr), d.match_id)
        rows.append({"target": "touch now" if h == 0 else f"touch +{h} s",
                     "w_train": w_tr, "w_train_se": w_tr_se, "train_matches": n_tr,
                     "test_RMSE_mid_c": np.sqrt(sq(0).mean()), "test_RMSE_blend_c": np.sqrt(sq(w_tr).mean()),
                     "test_RMSE_feed_c": np.sqrt(sq(1).mean()), "gain_c2": gain, "gain_se_c2": gain_se, "gain_t": gain / gain_se,
                     "w_test_refit": w_te, "w_test_se": w_te_se, "test_matches": n_te, "test_seconds": len(d)})
    return pd.DataFrame(rows).set_index("target")

def weight_curve_figure(F, w_tr, path, title, subtitle):
    ws = np.linspace(0, 1, 41)
    d = F[F.test & F.book_0.notna()]
    curve = [np.sqrt((1e4 * (d.mid + w * d.gap - d.book_0) ** 2).mean()) for w in ws]
    at = np.interp(w_tr, ws, curve)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ws, curve, color=BOX, linewidth=2)
    ax.axvline(w_tr, color=POOLED, linewidth=1.5, linestyle="--")
    ax.annotate(f"train-chosen w = {w_tr:.2f}\ntest RMSE {at:.2f}c", xy=(w_tr, at),
                xytext=(w_tr + 0.08, at + 0.15 * (max(curve) - min(curve))), color=INK, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8})
    ax.set_xlabel("feed weight w  (0 = our mid only, 1 = feed only)"); ax.set_ylabel("test RMSE vs touch mid (cents)")
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.9, subtitle, color=INK2, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(path, dpi=150); plt.show()

def trust_table(F, by, target="book_0"):
    """Within each bucket: w fitted on TRAIN; on TEST the gain of the blend (at that w) over the mid alone, and w refit."""
    out = {}
    base = F[F[target].notna()]
    for k, d in base.groupby(by, observed=True):
        tr_, te_ = d[~d.test], d[d.test]
        n_tr, n_te = tr_.match_id.nunique(), te_.match_id.nunique()
        w_tr = w_tr_se = w_te = w_te_se = g = g_se = rm0 = rmw = np.nan
        if n_tr >= 5: w_tr, w_tr_se, _ = w_fit(tr_, target)
        if n_te >= 5 and np.isfinite(w_tr):
            w_te, w_te_se, _ = w_fit(te_, target)
            sq = lambda w: 1e4 * (te_.mid + w * te_.gap - te_[target]) ** 2
            g, g_se = clustered_mean_se(sq(0) - sq(w_tr), te_.match_id)
            rm0, rmw = np.sqrt(sq(0).mean()), np.sqrt(sq(w_tr).mean())
        out[k] = {"share_of_seconds": len(d) / len(base), "w_train": w_tr, "w_train_se": w_tr_se,
                  "w_test_refit": w_te, "w_test_se": w_te_se, "test_RMSE_mid_c": rm0, "test_RMSE_blend_c": rmw,
                  "test_gain_c2": g, "gain_t": g / g_se if g_se else np.nan, "train_m": n_tr, "test_m": n_te}
    return pd.DataFrame(out).T

def state_dependent_blend(F, w_const, target="book_0"):
    """w_t = b0 + b1 log(1+feed age) + b2 log(1+tape staleness) + b3 |gap| (cents), fitted on TRAIN, clipped to [0, 1];
    compared on TEST with the constant-weight blend. Returns (coefficients, results dict)."""
    design = lambda d: np.column_stack([np.ones(len(d)), np.log1p(d.feed_age_s), np.log1p(d.since_print_s), 100 * d.gap.abs()])
    tr_ = F[~F.test & F[target].notna()]; te_ = F[F.test & F[target].notna()]
    beta, *_ = np.linalg.lstsq(design(tr_) * tr_.gap.values[:, None], (tr_[target] - tr_.mid).values, rcond=None)
    w = np.clip(design(te_) @ beta, 0, 1)
    sq_mid = 1e4 * (te_.mid - te_[target]) ** 2
    sq_c = 1e4 * (te_.mid + w_const * te_.gap - te_[target]) ** 2
    sq_s = 1e4 * (te_.mid + w * te_.gap - te_[target]) ** 2
    g, g_se = clustered_mean_se(sq_c - sq_s, te_.match_id)
    coef = dict(zip(["intercept", "log1p(feed age s)", "log1p(tape staleness s)", "|gap| (c)"], beta))
    return coef, {"test_RMSE_mid_c": np.sqrt(sq_mid.mean()), "test_RMSE_constant_c": np.sqrt(sq_c.mean()),
                  "test_RMSE_state_c": np.sqrt(sq_s.mean()), "gain_vs_constant_c2": g, "gain_t": g / g_se,
                  "w_median": np.median(w), "w_p10": np.quantile(w, .1), "w_p90": np.quantile(w, .9)}

def rmse_change(d_te, w, target="book_0", B=2000):
    """% change in TEST RMSE from adding the feed at weight w; 95% CI by resampling test matches."""
    s = pd.DataFrame({"s0": 1e4 * (d_te.mid - d_te[target]) ** 2, "sw": 1e4 * (d_te.mid + w * d_te.gap - d_te[target]) ** 2,
                      "g": d_te.match_id.values}).groupby("g").sum()
    est = np.sqrt(s.sw.sum() / s.s0.sum()) - 1
    k = rng.multinomial(len(s), np.ones(len(s)) / len(s), size=B)
    bs = np.sqrt((k * s.sw.values).sum(1) / (k * s.s0.values).sum(1)) - 1
    return 100 * est, 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)

def forest_table(F, groups, target="book_0"):
    rows, base = [], F[F[target].notna()]
    for gname, col in groups:
        buckets = [("all seconds", base)] if col is None else list(base.groupby(col, observed=True))
        for k, d in buckets:
            tr_, te_ = d[~d.test], d[d.test]
            if tr_.match_id.nunique() < 5 or te_.match_id.nunique() < 5: continue
            w, _, _ = w_fit(tr_, target)
            e, lo, hi = rmse_change(te_, w, target)
            rows.append({"group": gname, "bucket": str(k), "w_train": w, "rmse_change_pct": e, "lo": lo, "hi": hi,
                         "share": len(d) / len(base), "test_m": te_.match_id.nunique()})
    return pd.DataFrame(rows)

def forest_figure(T, groups, path, title, subtitle, xmax=25):
    HELP, NONE, HURT = BOX, NONE_C, POOLED
    fig, ax = plt.subplots(figsize=(9.5, 0.34 * (len(T) + 2 * len(groups)) + 1.4))
    y, yt, yl, heads = 0, [], [], []
    xmin = T.lo.min() - 2
    for gname, _ in groups:
        sub = T[T.group == gname]
        if sub.empty: continue
        if gname != "all seconds": heads.append((y, gname)); y += 1
        for _, r in sub.iterrows():
            col = HELP if r.hi < 0 else (HURT if r.lo > 0 else NONE)
            ax.plot([r.lo, min(r.hi, xmax)], [y, y], color=col, linewidth=2.5, solid_capstyle="round")
            if r.hi > xmax:
                ax.annotate("", xy=(xmax + 1.2, y), xytext=(xmax - 0.5, y), arrowprops={"arrowstyle": "-|>", "color": col, "lw": 2.5})
            if r.rmse_change_pct <= xmax:
                ax.plot(r.rmse_change_pct, y, "o", ms=8, color=col, mec=SURFACE, mew=1.5)
            note = f"w={r.w_train:.2f} · {100 * r.share:.0f}% of seconds"
            if r.hi > xmax: note = f"{r.rmse_change_pct:+.0f}% [{r.lo:+.0f}, {r.hi:+.0f}] · " + note
            ax.text(xmax + 3, y, note, va="center", fontsize=8, color=INK2)
            yt.append(y); yl.append(r.bucket); y += 1
        y += 0.5
    for yy, g in heads:
        ax.text(xmin + 0.3, yy, g, fontsize=9.5, fontweight="bold", color=INK, va="center", ha="left")
    ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9); ax.invert_yaxis()
    ax.set_xlim(xmin, xmax + 2); ax.set_ylim(y - 0.2, -0.8)
    ax.set_xlabel("change in TEST RMSE from adding the feed (%)     ←  feed helps   |   feed hurts  →")
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], color=HELP, marker="o", lw=2.5, label="feed lowers error (95% CI below 0)"),
               Line2D([], [], color=NONE, marker="o", lw=2.5, label="no evidence either way (CI spans 0)"),
               Line2D([], [], color=HURT, marker="o", lw=2.5, label="feed raises error (CI above 0)")]
    H = fig.get_size_inches()[1]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1 - 0.78 / H), ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle(title, x=0.01, y=1 - 0.12 / H, ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.01, 1 - 0.62 / H, subtitle, color=INK2, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0, 0.80, 1 - 1.1 / H))
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.show()


# =================================================================================================
# The final Part 2 pipeline (moved here from part2.ipynb)
# =================================================================================================
LEVELS = [0, .1, .25, .4, .6, .75, .9, 1.0001]
MAX_BOOK_AGE_S = 5.0          # touch target: last change at most 5 s old (DATA.md: older = unknown) and not locked/crossed

def _second_grid(tr):
    span = tr.groupby("match_id").t_ns.agg(["min", "max"]) / 1e9
    return pd.concat([pd.DataFrame({"match_id": m, "t": np.arange(r["min"], r["max"], 1.0)}) for m, r in span.iterrows()],
                     ignore_index=True)

def add_touch_targets(F, bb, horizons=(0, 30, 120)):
    """book_h = the recorded touch mid at t + h (TARGET only), NaN when stale (> 5 s) or locked / crossed."""
    from starter import asof
    book = bb.assign(bmid=(bb.bid + bb.ask) / 2, uncrossed=bb.bid < bb.ask, book_t_ns=bb.t_ns)
    for h in horizons:
        q = F[["match_id"]].assign(_q=((F.t + h) * 1e9).astype("int64"))
        r = asof(q, book, "_q", "t_ns", cols=["bmid", "uncrossed", "book_t_ns"])
        fresh = (q._q.values - r.book_t_ns.values) / 1e9 <= MAX_BOOK_AGE_S
        F[f"book_{h}"] = np.where(r.uncrossed.fillna(False).astype(bool).values & fresh, r.bmid.values, np.nan)
    return F

def cs2_comparison_grid(tr, bb, P, test_ids):
    """One row per in-play second of every mapped CS2 match while BETER's live match-winner market is open:
    Model 3b mid (prints <= t - LAT), BETER p1 as received (<= t - LAT), map / round phase from real-time incidents,
    the touch targets, and the 'when to trust' buckets."""
    from starter import asof, beter_match_winner
    from p1_mid import LAT, query_prints, predict_lasso3b
    feed = beter_match_winner(3); feed = feed[feed.line_type == 1].copy(); feed["feed_t_ns"] = feed.t_recv_ns
    inc = pd.read_parquet(D_DATA + "beter_esports_incident.parquet")
    inc = inc[(inc.sport_id == 3) & (inc.msg_type == 1) & inc.map_score.notna() & inc.round_score.notna()].copy()
    ms = inc.map_score.str.split(":", expand=True).astype(float); rs = inc.round_score.str.split(":", expand=True).astype(float)
    inc["map_no"], inc["rounds_played"] = ms[0] + ms[1] + 1, rs[0] + rs[1]
    C = query_prints(tr, _second_grid(tr)); C["mid"] = predict_lasso3b(C, tr, P)
    C = C[["match_id", "t", "since_print_s", "tape_mid", "mid"]].copy()
    C["a_ns"] = ((C.t - LAT) * 1e9).astype("int64")
    f = asof(C, feed, "a_ns", "t_recv_ns", cols=["p1", "st1", "feed_t_ns"])
    C["feed"], C["st1"], C["feed_age_s"] = f.p1.values, f.st1.values, (C.a_ns.values - f.feed_t_ns.values) / 1e9
    C = C[C.st1 == 1].reset_index(drop=True)                                    # BETER live market open
    g = asof(C, inc, "a_ns", "t_recv_ns", cols=["map_no", "rounds_played"])
    C["map_no"], C["rounds_played"] = g.map_no.values, g.rounds_played.values
    C["gap"], C["test"] = C.feed - C.mid, C.match_id.isin(test_ids)
    add_touch_targets(C, bb)
    C["level"] = pd.cut((C.feed + C.mid) / 2, LEVELS, right=False)
    C["map_phase"] = C.map_no.clip(upper=3).map({1: "map 1", 2: "map 2", 3: "map 3+"})
    C["feed_age_b"] = pd.cut(C.feed_age_s, [0, 5, 30, 120, 600, np.inf], right=False, labels=["<5 s", "5-30 s", "30 s-2 min", "2-10 min", "10+ min"])
    C["gap_b"] = pd.cut(100 * C.gap.abs(), [0, 1, 2, 5, 10, np.inf], right=False, labels=["<1c", "1-2c", "2-5c", "5-10c", "10c+"])
    C["tape_b"] = pd.cut(C.since_print_s, [0, 1, 5, 30, 120, np.inf], right=False, labels=["<1 s", "1-5 s", "5-30 s", "30 s-2 min", "2+ min"])
    return C

CS2_TRUST_GROUPS = [("all seconds", None), ("BETER quiet for", "feed_age_b"), ("|feed − mid|", "gap_b"),
                    ("time since last Kalshi print", "tape_b"), ("map", "map_phase")]
MLB_TRUST_GROUPS = [("all seconds", None), ("Sportradar state unchanged for", "feed_age_b"), ("|feed − mid|", "gap_b"),
                    ("time since last Kalshi print", "tape_b"), ("inning", "inning_phase"), ("Sportradar Betstop", "betstop_b")]

def fit_wp_model(ev, mlb_ids, test_ids, seed_grid=True):
    """The MLB win-probability model: structural win expectancy from run distributions fitted on TRAIN games, plus an
    XGBoost correction boosted from logit(we) with monotone constraints. Depth / min_child_weight / trees chosen by
    grouped 5-fold CV log loss on TRAIN (0 trees = the structural model alone). Returns (S with wp / we, model, cv table)."""
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import log_loss
    mp = pd.read_parquet(D_DATA + "map_mlb.parquet")
    S = mlb_state_changes(ev, mlb_ids)
    S["y"] = (S.match_id.map(mp.set_index("sr_match_id").result_home) == "yes").astype(float)
    S["test"] = S.match_id.isin(test_ids)
    dists = fit_run_distributions(S[~S.test]); X = wp_features(S, dists)
    trm = ~S.test.values; Xtr, ytr, gtr = X[trm], S.y.values[trm], S.match_id.values[trm]
    make = lambda md, mcw, ne: xgb.XGBClassifier(n_estimators=ne, max_depth=md, min_child_weight=mcw, learning_rate=0.03, subsample=0.8,
                                                 reg_lambda=10, monotone_constraints=WP_MONO, eval_metric="logloss", n_jobs=4)
    cv = []
    for md, mcw, ne in [(1, 20, 0)] + [(m, c, n) for m in (1, 2, 3) for c in (20, 80, 200) for n in (25, 75, 200)]:
        ll = []
        for a, b in GroupKFold(5).split(Xtr, ytr, gtr):
            if ne == 0: p = Xtr.we.values[b]
            else: p = predict_wp(make(md, mcw, ne).fit(Xtr.iloc[a][WP_FEATS].values, ytr[a], base_margin=logit(Xtr.we.values[a])), Xtr.iloc[b])
            ll.append(log_loss(ytr[b], np.clip(p, 1e-4, 1 - 1e-4), labels=[0, 1]))
        cv.append({"max_depth": md, "min_child_weight": mcw, "n_estimators": ne, "cv_logloss": np.mean(ll)})
    cv = pd.DataFrame(cv).sort_values("cv_logloss"); best = cv.iloc[0]
    model = make(int(best.max_depth), int(best.min_child_weight), max(int(best.n_estimators), 1))
    if best.n_estimators > 0:
        model.fit(Xtr[WP_FEATS].values, ytr, base_margin=logit(Xtr.we.values)); S["wp"] = predict_wp(model, X)
    else:
        S["wp"] = X.we.values
    S["we"] = X.we.values
    return S, model, cv

def mlb_comparison_grid(tr, bb, P, S, ev, test_ids):
    """One row per in-play second of every MLB game: Model 6 mid, the win-probability model as received, Betstop,
    the touch targets and the 'when to trust' buckets."""
    from starter import asof
    from p1_mid import LAT, query_prints, predict_lasso6
    M = query_prints(tr, _second_grid(tr)); M["mid"] = predict_lasso6(M, tr, P)
    M = M[["match_id", "t", "since_print_s", "mid"]].copy()
    M["a_ns"] = ((M.t - LAT) * 1e9).astype("int64")
    f = asof(M, S.assign(state_t_ns=S.t_recv_ns), "a_ns", "t_recv_ns", cols=["wp", "we", "inning", "half", "outs", "diff", "state_t_ns"])
    M["feed"], M["we"], M["inning"], M["half"] = f.wp.values, f.we.values, f.inning.values, f.half.values
    M["feed_age_s"] = (M.a_ns.values - f.state_t_ns.values) / 1e9
    M["betstop"] = asof(M, betstop_intervals(ev, set(M.match_id)), "a_ns", "t_recv_ns", cols=["betstop"]).betstop.fillna(False).astype(bool).values
    M = M[M.feed.notna()].reset_index(drop=True)                               # no state before the first pitch
    M["gap"], M["test"] = M.feed - M.mid, M.match_id.isin(test_ids)
    add_touch_targets(M, bb)
    M["level"] = pd.cut((M.feed + M.mid) / 2, LEVELS, right=False)
    M["inning_phase"] = pd.cut(M.inning, [0, 3, 6, 8, 9, 99], labels=["1-3", "4-6", "7-8", "9th", "extras"])
    M["feed_age_b"] = pd.cut(M.feed_age_s, [0, 10, 30, 120, 300, np.inf], right=False, labels=["<10 s", "10-30 s", "30 s-2 min", "2-5 min", "5+ min"])
    M["gap_b"] = pd.cut(100 * M.gap.abs(), [0, 1, 2, 5, 10, 20, np.inf], right=False, labels=["<1c", "1-2c", "2-5c", "5-10c", "10-20c", "20c+"])
    M["tape_b"] = pd.cut(M.since_print_s, [0, 1, 5, 30, np.inf], right=False, labels=["<1 s", "1-5 s", "5-30 s", "30 s+"])
    M["betstop_b"] = np.where(M.betstop, "Betstop active", "betting open")
    return M

def level_labels(F):
    """Readable price-level labels for the box plots."""
    lab = [f"{a:g}–{b:g}" for a, b in zip(LEVELS[:-1], [.1, .25, .4, .6, .75, .9, 1])]
    return pd.Categorical(F.level.astype(str).str.replace(r"\[(.*), (.*)\)", r"\1–\2", regex=True).str.replace("1.0001", "1"), lab, ordered=True)
