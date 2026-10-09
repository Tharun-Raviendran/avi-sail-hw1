"""Part 5 helpers: forecast the change in the Kalshi touch mid over the next 10 s and 30 s.

Rules every feature obeys (see part5.ipynb):
  * causal: prints stamped <= t - LAT, touch rows received <= t, feed messages received <= t - LAT + k
    (k > 0 only in the "feed delivered k s earlier" fit);
  * no feature contains the touch mid (the target's starting point). "Feed minus mid" uses the TAPE mid
    (Part 1 Model 3b / Model 6), never the touch mid.
"""
import numpy as np, pandas as pd

from starter import D, asof, beter_match_winner
from p1_mid import LAT, build_grid, predict_lasso3b, predict_lasso6
from p4_latency import _touch_target

HORIZONS = (10, 30)
FLOW_WINDOWS = (2, 10, 30, 120)


# =================================================================================================
# the grid and the targets
# =================================================================================================
def base_grid(ctx, P, sport):
    """1 s in-play grid on book_ok matches (Part 1's build_grid: touch at t fresh <= 5 s and uncrossed), the Part 1
    tape mid, and the targets y10 / y30 = touch mid(t + h) - touch mid(t) in cents (NaN if the touch at t + h is
    stale or crossed)."""
    ids = ctx["train"] | ctx["test"]
    G = build_grid(ctx["tr"], ctx["bb"], ids)
    G["mtape"] = (predict_lasso3b if sport == "CS2" else predict_lasso6)(G, ctx["tr"], P)
    G = G[["match_id", "t", "mid", "stale_s", "since_print_s", "tape_mid", "mtape"]].rename(columns={"mid": "touch_mid"})
    for h in HORIZONS:
        G[f"y{h}"] = 100 * (_touch_target(G.assign(t=G.t + h), ctx["bb"]) - G.touch_mid.values)
    G["test"] = G.match_id.isin(ctx["test"])
    return G.reset_index(drop=True)


# =================================================================================================
# features
# =================================================================================================
def _per_match(G, tr):
    """Yield (row positions in G, that match's prints) for every match in G."""
    by = tr.groupby("match_id").indices
    for m, idx in G.groupby("match_id").indices.items():
        if m in by: yield idx, tr.iloc[by[m]]

def tape_features(G, tr, t):
    """Signed taker flow, print intensity and tape momentum from prints stamped <= t - LAT (t: query times, s)."""
    out = {f"imb_{w}s": np.zeros(len(G)) for w in FLOW_WINDOWS}
    out.update({"log_prints_30s": np.zeros(len(G)), "tape_mom_10s_c": np.zeros(len(G)), "tape_mom_30s_c": np.zeros(len(G))})
    for idx, g in _per_match(G, tr):
        tp = g.t_ns.values / 1e9; c = g["count"].values; d = g["dir"].values
        S = np.concatenate([[0], np.cumsum(c * d)]); V = np.concatenate([[0], np.cumsum(c)])
        tm = np.concatenate([[np.nan], g.px.values - d * 0.01])          # tape_mid after each print (index 0 = none yet)
        q = t[idx] - LAT
        hi = np.searchsorted(tp, q, side="right")
        for w in FLOW_WINDOWS:
            lo = np.searchsorted(tp, q - w, side="right")
            v = V[hi] - V[lo]
            out[f"imb_{w}s"][idx] = np.where(v > 0, (S[hi] - S[lo]) / np.maximum(v, 1e-12), 0.0)   # in [-1, 1]
        out["log_prints_30s"][idx] = np.log1p(hi - np.searchsorted(tp, q - 30, side="right"))
        for w in (10, 30):
            past = tm[np.searchsorted(tp, q - w, side="right")]
            out[f"tape_mom_{w}s_c"][idx] = np.nan_to_num(100 * (tm[hi] - past))
    return pd.DataFrame(out, index=G.index)

def touch_features(G, bb, t):
    """Quoted spread, queue imbalance and touch age from the touch in force at t. Never the touch mid itself."""
    q = G[["match_id"]].assign(q=(t * 1e9).astype("int64"))
    r = asof(q, bb.assign(bt=bb.t_ns), "q", "t_ns", cols=["bid", "ask", "bid_sz", "ask_sz", "bt"])
    tot = (r.bid_sz + r.ask_sz).values
    out = pd.DataFrame({"spread_c": 100 * (r.ask - r.bid).values,
                        "queue_imb": np.where(tot > 0, (r.bid_sz - r.ask_sz).values / np.maximum(tot, 1e-12), 0.0),
                        "log_touch_age": np.log1p(np.maximum(t - r.bt.values / 1e9, 0))}, index=G.index)
    # no touch recorded yet at t (only possible in the 60 s-stale placebo): neutral values
    return out.fillna({"spread_c": out.spread_c.median(), "queue_imb": 0.0, "log_touch_age": out.log_touch_age.max()})

def _last_event(G, ev, t_feed_ns, cols):
    """Last feed event received <= the (shifted) information time, per row: its columns and age in seconds."""
    q = G[["match_id"]].assign(q=t_feed_ns)
    r = asof(q, ev.assign(et=ev.t_recv_ns), "q", "t_recv_ns", cols=cols + ["et"])
    r["age"] = (t_feed_ns - r.et.values) / 1e9
    return r

def cs2_feed_features(G, k, t):
    """BETER, as received k s earlier: BETER - tape mid, last BETER jump (>= 3c), last RoundWon."""
    from p3_leadlag import round_won
    a = ((t - LAT + k) * 1e9).astype("int64")
    ids = set(G.match_id)
    bw = beter_match_winner(3); bw = bw[(bw.line_type == 1) & bw.match_id.isin(ids)].copy()
    f = _last_event(G, bw, a, ["p1", "st1"])
    open_ = (f.st1 == 1).values
    gap = np.where(open_, 100 * (f.p1.values - G.mtape.values), 0.0)
    live = bw[bw.st1 == 1].copy(); live["dp"] = live.groupby("match_id").p1.diff()
    jumps = live[live.dp.abs() >= 0.03]
    j = _last_event(G, jumps, a, ["dp"])
    jump_10 = np.where(j.age.values <= 10, 100 * j.dp.fillna(0).values, 0.0)
    rw = round_won(ids); r = _last_event(G, rw, a, ["sign"])
    return pd.DataFrame({"feed_gap_c": gap, "feed_open": open_.astype(float),
                         "feed_gap_x_stale": gap * np.log1p(G.since_print_s.values),
                         "beter_jump_10s_c": jump_10, "log_beter_jump_age": np.log1p(j.age.fillna(3600).clip(upper=3600).values),
                         "round_won_10s": np.where(r.age.values <= 10, r.sign.fillna(0).values, 0.0),
                         "log_round_age": np.log1p(r.age.fillna(3600).clip(upper=3600).values)}, index=G.index)

def mlb_feed_features(G, k, t, S, ev, tr):
    """Sportradar, as received k s earlier: win-probability change (anchored), the level gap, last run, Betstop
    active x batting side, and taker flow since the last process message (ball in play / pitch release)."""
    a = ((t - LAT + k) * 1e9).astype("int64")
    wp = S[["match_id", "t_recv_ns", "wp", "half"]]
    now = _last_event(G, wp, a, ["wp", "half"])
    d_wp = {}
    for w in (10, 30):
        past = _last_event(G, wp, a - int(w * 1e9), ["wp"])
        d_wp[w] = np.nan_to_num(100 * (now.wp.values - past.wp.values))
    d = ev[(ev.feedtype == "delta") & ev.match_id.isin(set(G.match_id))]
    runs = d[(d.type == "1720") & d.side.isin(["home", "away"])].assign(sign=lambda x: np.where(x.side == "home", 1, -1))
    rn = _last_event(G, runs, a, ["sign"])
    bs = d[d.type.isin(["1010", "1011"])].assign(stop=lambda x: (x.type == "1011").astype(float))
    b = _last_event(G, bs, a, ["stop"])
    bat = np.where(now.half.values == "B", 1.0, np.where(now.half.values == "T", -1.0, 0.0))     # +1 = home batting
    proc = d[d.type.isin(["1031", "2327"])].assign(bip=lambda x: (x.type == "1031").astype(float))
    p = _last_event(G, proc, a, ["bip"])
    # signed taker flow from the last process message's (shifted) arrival to t - LAT, if that message is <= 10 s old
    flow = np.zeros(len(G))
    p_t = (p.et.values - k * 1e9) / 1e9
    for idx, g in _per_match(G, tr):
        tp = g.t_ns.values / 1e9; c = g["count"].values; dd = g["dir"].values
        Sx = np.concatenate([[0], np.cumsum(c * dd)]); V = np.concatenate([[0], np.cumsum(c)])
        hi = np.searchsorted(tp, t[idx] - LAT, side="right"); lo = np.searchsorted(tp, np.nan_to_num(p_t[idx], nan=-1e18), side="right")
        lo = np.minimum(lo, hi); v = V[hi] - V[lo]
        flow[idx] = np.where(v > 0, (Sx[hi] - Sx[lo]) / np.maximum(v, 1e-12), 0.0)
    recent = p.age.values <= 10
    return pd.DataFrame({"wp_gap_c": np.nan_to_num(100 * (now.wp.values - G.mtape.values)),
                         "d_wp_10s_c": d_wp[10], "d_wp_30s_c": d_wp[30],
                         "run_10s": np.where(rn.age.values <= 10, rn.sign.fillna(0).values, 0.0),
                         "betstop_x_bat": np.where(b.stop.fillna(0).values == 1, bat, 0.0),
                         "bip_10s": np.where(recent, p.bip.fillna(0).values, 0.0),
                         "flow_since_play": np.where(recent, flow, 0.0),
                         "flow_since_bip": np.where(recent & (p.bip.fillna(0).values == 1), flow, 0.0)}, index=G.index)

def context_features(G, t, start_s):
    m = G.mtape.values
    return pd.DataFrame({"dist_edge_c": 100 * np.minimum(m, 1 - m), "minutes_in_play": (t - start_s) / 60}, index=G.index)

def feature_table(G, ctx, sport, k, lag_s=0.0, S=None, ev=None):
    """All features at time t - lag_s (lag_s > 0 = the stale-information placebo). Target columns stay at t."""
    from p3_leadlag import inplay_spans
    t = G.t.values - lag_s
    start = inplay_spans("esports" if sport == "CS2" else "mlb").start.reindex(G.match_id).values / 1e9
    parts = [tape_features(G, ctx["tr"], t), touch_features(G, ctx["bb"], t), context_features(G, t, start)]
    parts.append(cs2_feed_features(G, k, t) if sport == "CS2" else mlb_feed_features(G, k, t, S, ev, ctx["tr"]))
    X = pd.concat(parts, axis=1)
    # flow x intensity: the same imbalance means more when many prints are behind it
    X["imb_10s_x_prints"] = X.imb_10s * X.log_prints_30s
    return X


# =================================================================================================
# models and scoring
# =================================================================================================
def r2_boot(y, yhat, ybar, groups, B=1000, seed=0):
    """Test R^2 = 1 - SSE / SST around the TRAIN mean, with a match-bootstrap 95% CI."""
    s = pd.DataFrame({"sse": (y - yhat) ** 2, "sst": (y - ybar) ** 2, "g": groups}).groupby("g").sum()
    r2 = 1 - s.sse.sum() / s.sst.sum()
    rng = np.random.default_rng(seed)
    Wt = rng.multinomial(len(s), np.ones(len(s)) / len(s), size=B)
    rb = 1 - (Wt @ s.sse.values) / (Wt @ s.sst.values)
    return r2, np.percentile(rb, 2.5), np.percentile(rb, 97.5)

def fit_lasso(X, y, groups, n_folds=5):
    """Lasso on standardized features; penalty = the grouped-CV minimum on TRAIN matches.
    (Part 1's one-standard-error rule is too blunt here: the fold-to-fold spread of error, driven by a few volatile
    matches, is larger than the whole signal, so the 1-SE rule zeroes every feature.)"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.model_selection import GroupKFold
    scaler = StandardScaler().fit(X.values); Z = scaler.transform(X.values)
    cv = LassoCV(cv=list(GroupKFold(n_folds).split(Z, y, groups)), n_alphas=40, n_jobs=-1).fit(Z, y)
    model = Lasso(alpha=cv.alpha_).fit(Z, y)
    return {"kind": "lasso", "scaler": scaler, "model": model, "features": list(X.columns), "alpha": cv.alpha_}

def fit_xgb(X, y, groups, seed=0):
    """Shallow boosted trees (depth 2: pairwise interactions only, still explainable with SHAP), number of trees by
    grouped CV on train."""
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    best = None
    for n in (100, 300, 600):
        sse = 0.0
        for a, b in GroupKFold(4).split(X, y, groups):
            m = xgb.XGBRegressor(n_estimators=n, max_depth=2, learning_rate=0.03, subsample=0.7, colsample_bytree=0.8,
                                 min_child_weight=200, reg_lambda=10, n_jobs=8, random_state=seed, tree_method="hist")
            m.fit(X.iloc[a], y[a]); sse += ((y[b] - m.predict(X.iloc[b])) ** 2).sum()
        if best is None or sse < best[1]: best = (n, sse)
    m = xgb.XGBRegressor(n_estimators=best[0], max_depth=2, learning_rate=0.03, subsample=0.7, colsample_bytree=0.8,
                         min_child_weight=200, reg_lambda=10, n_jobs=8, random_state=seed, tree_method="hist").fit(X, y)
    return {"kind": "xgb", "model": m, "features": list(X.columns), "n_trees": best[0]}

def predict(M, X):
    X = X[M["features"]]
    if M["kind"] == "lasso": return M["model"].predict(M["scaler"].transform(X.values))
    return M["model"].predict(X)

def permutation_importance(M, X, y, ybar, units, seed=0, reps=3):
    """Drop in test R^2 when a UNIT of features is shuffled (rows permuted together across the test set, averaged
    over `reps` shuffles). Correlated features that only work together (a gap and its interaction) are one unit,
    otherwise shuffling one of them alone overstates its importance. units: {name: [columns]}."""
    rng = np.random.default_rng(seed)
    base = 1 - ((y - predict(M, X)) ** 2).sum() / ((y - ybar) ** 2).sum()
    out = {}
    for name, cols in units.items():
        cols = [c for c in cols if c in M["features"]]
        if not cols: continue
        drops = []
        for _ in range(reps):
            Xp = X.copy(); perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].values[perm]
            drops.append(base - (1 - ((y - predict(M, Xp)) ** 2).sum() / ((y - ybar) ** 2).sum()))
        out[name] = np.mean(drops)
    return pd.Series(out).sort_values(ascending=False)

def feature_units(features):
    """Each feature is its own unit, except pairs that only make sense together."""
    pairs = {"feed_gap_x_stale": "feed_gap_c", "imb_10s_x_prints": "imb_10s"}
    units = {}
    for f in features:
        key = pairs.get(f, f); units.setdefault(key, []).append(f)
    return {(k + " (+ interaction)" if len(v) > 1 else k): v for k, v in units.items()}

def lasso_coefs(M):
    """Cents of expected move per one standard deviation of each feature (zero = dropped by the Lasso)."""
    return pd.Series(M["model"].coef_, index=M["features"]).sort_values(key=np.abs, ascending=False)


FEATURE_GROUPS = {
    "tape":  ["imb_2s", "imb_10s", "imb_30s", "imb_120s", "log_prints_30s", "tape_mom_10s_c", "tape_mom_30s_c", "imb_10s_x_prints"],
    "touch": ["spread_c", "queue_imb", "log_touch_age"],
    "context": ["dist_edge_c", "minutes_in_play"],
    "feed_CS2": ["feed_gap_c", "feed_open", "feed_gap_x_stale", "beter_jump_10s_c", "log_beter_jump_age", "round_won_10s", "log_round_age"],
    "feed_MLB": ["wp_gap_c", "d_wp_10s_c", "d_wp_30s_c", "run_10s", "betstop_x_bat", "bip_10s", "flow_since_play", "flow_since_bip"],
    "state": ["map_no", "map_diff", "round_diff", "rounds_played", "inning", "bottom", "outs", "run_diff", "late", "close", "near_50"],
}
def group_of(f):
    if "_x_near_50" in f or "_x_late" in f or "_x_close" in f: return "state"
    return next(g.replace("_CS2", "").replace("_MLB", "") for g, fs in FEATURE_GROUPS.items() if f in fs)


# =================================================================================================
# game state (as received by t - LAT) and the "is the game tense?" scalers
# =================================================================================================
def cs2_state_features(G, t, k=0.0):
    """BETER real-time incidents: map number, series score, rounds in the current map. +x = team1 ahead."""
    inc = pd.read_parquet(D + "beter_esports_incident.parquet")
    inc = inc[(inc.sport_id == 3) & (inc.msg_type == 1) & inc.map_score.notna() & inc.round_score.notna()
              & inc.match_id.isin(set(G.match_id))].copy()
    ms = inc.map_score.str.split(":", expand=True).apply(pd.to_numeric, errors="coerce")
    rs = inc.round_score.str.split(":", expand=True).apply(pd.to_numeric, errors="coerce")
    inc["map_no"], inc["map_diff"] = ms[0] + ms[1] + 1, ms[0] - ms[1]
    inc["r1"], inc["r2"] = rs[0], rs[1]
    q = G[["match_id"]].assign(q=((t - LAT + k) * 1e9).astype("int64"))
    r = asof(q, inc, "q", "t_recv_ns", cols=["map_no", "map_diff", "r1", "r2"]).fillna({"map_no": 1, "map_diff": 0, "r1": 0, "r2": 0})
    rounds = r.r1 + r.r2; lead = r.r1 - r.r2
    return pd.DataFrame({"map_no": r.map_no.values, "map_diff": r.map_diff.values, "round_diff": lead.values,
                         "rounds_played": rounds.values,
                         "late": ((np.maximum(r.r1, r.r2) >= 10) | (r.map_no >= 3)).astype(float).values,   # a map near its end, or a deciding map
                         "close": (lead.abs() <= 3).astype(float).values}, index=G.index)

def mlb_state_features(G, t, S, k=0.0):
    """Sportradar game state as received: inning, half, outs, score difference (home - away)."""
    q = G[["match_id"]].assign(q=((t - LAT + k) * 1e9).astype("int64"))
    r = asof(q, S, "q", "t_recv_ns", cols=["inning", "half", "outs", "diff"])
    inning = r.inning.fillna(1).values; diff = r["diff"].fillna(0).values
    return pd.DataFrame({"inning": inning, "bottom": (r.half.values == "B").astype(float), "outs": r.outs.fillna(0).values,
                         "run_diff": diff, "late": (inning >= 7).astype(float), "close": (np.abs(diff) <= 1).astype(float)}, index=G.index)

def add_state(X, G, t, sport, S=None, k=0.0):
    """Game state (from the feed, so shifted by the same k) + 'tension' scalers + signal x state interactions."""
    st = cs2_state_features(G, t, k) if sport == "CS2" else mlb_state_features(G, t, S, k)
    X = pd.concat([X, st], axis=1)
    X["near_50"] = 1 - np.abs(G.mtape.values - 0.5) * 2                              # 1 at 50%, 0 at 0 or 100%
    signals = ["queue_imb", "imb_2s"] + (["feed_gap_c"] if sport == "CS2" else ["flow_since_bip"])
    for f in signals:
        for s in ("near_50", "late", "close"):
            X[f"{f}_x_{s}"] = X[f] * X[s]
    return X


# =================================================================================================
# PART 5b: the CANCEL and TAKE rules
# =================================================================================================
QUOTE_MAX_AGE_S = 2.0          # quote only if our forecast row is at most this old (else our data is stale: no quotes)
Q_MAX = 100.0                  # post size cap per fill (mm_core default)

def settle_values(sport):
    if sport == "CS2":
        r = pd.read_parquet(D + "map_esports.parquet").set_index("beter_match_id").result_team1
    else:
        r = pd.read_parquet(D + "map_mlb.parquet").set_index("sr_match_id").result_home
    return r.map({"yes": 1.0, "no": 0.0})                       # void / scalar -> NaN (no settlement mark)

def maker_fills(ctx, F, h_c, strict, sport, lat=LAT):
    """Always-on maker: bid/ask = fair -+ h (1c ticks), fair = Part 1 tape mid from the latest forecast row
    stamped <= print time - lat (and <= QUOTE_MAX_AGE_S old). Fills from mm_core.simulate_maker (print-driven;
    strict=True = only prints THROUGH our price). Each fill carries the forecasts the rule would have seen, and marks
    to the touch mid at +30 s / +120 s and to settlement. F: one row per grid second with match_id, t, mtape and
    forecast columns."""
    from mm_core import simulate_maker
    from p3_leadlag import touch_mid_at
    settle = settle_values(sport)
    fc = [c for c in F.columns if c.startswith(("size", "dir"))]
    by_tr = ctx["tr"].groupby("match_id").indices
    out = []
    for m, f in F.groupby("match_id"):
        if m not in by_tr: continue
        g = ctx["tr"].iloc[by_tr[m]]
        tp = g.t_ns.values / 1e9; ft = f.t.values
        j = np.searchsorted(ft, tp - lat, side="right") - 1
        ok = (j >= 0) & ((tp - lat - ft[np.clip(j, 0, None)]) <= QUOTE_MAX_AGE_S)
        jj = np.clip(j, 0, None)
        fair = np.where(ok, f.mtape.values[jj], np.nan)
        fair = np.where((fair > 0.03) & (fair < 0.97), fair, np.nan)
        bid = np.floor((fair - h_c / 100) * 100 + 1e-9) / 100; ask = np.ceil((fair + h_c / 100) * 100 - 1e-9) / 100
        fl = simulate_maker(g, bid, ask, settle.get(m, np.nan), q_max=Q_MAX, horizon_s=30.0, strict=strict)
        if not len(fl): continue
        pos = np.searchsorted(g.t_ns.values, fl.t_ns.values)           # the print that filled us
        for c in fc: fl[c] = f[c].values[jj[pos]]
        fl["match_id"] = m
        out.append(fl)
    fills = pd.concat(out, ignore_index=True)
    sgn = np.where(fills.side == "buy", 1.0, -1.0); t = fills.t_ns.values / 1e9
    for H in (30, 120):
        fills[f"pnl_book{H}"] = sgn * (touch_mid_at(ctx["bb"], fills.match_id.values, t + H) - fills.price.values)
    fills["test"] = fills.match_id.isin(ctx["test"])
    return fills

def wmean_t(pnl, w, groups):
    """Size-weighted mean P&L per contract (cents) and its t-stat clustered by match (ratio estimator)."""
    d = pd.DataFrame({"x": pnl * w, "w": w, "g": groups}).dropna().groupby("g").sum()
    if d.w.sum() == 0: return np.nan, np.nan
    r = d.x.sum() / d.w.sum(); se = np.sqrt(((d.x - r * d.w) ** 2).sum()) / d.w.sum()
    return 100 * r, (r / se if se > 1e-12 else np.nan)

def cancel_mask(fl, x, variant, size_col="size10", dir_col="dir10"):
    """Which fills the CANCEL rule would have avoided. 'both': pull both quotes when the size forecast > x.
    'one-sided': pull only the side the direction forecast says is about to be run over."""
    big = fl[size_col].values > x
    if variant == "both": return big
    at_risk = ((fl.side.values == "sell") & (fl[dir_col].values > 0)) | ((fl.side.values == "buy") & (fl[dir_col].values < 0))
    return big & at_risk

def take_trades(G, F, ctx, h, lat=LAT):
    """TAKE rule on the 1 s grid: cross when |direction forecast| > half the spread + taker fee 7 p (1-p) cents.
    Entry at the touch in force at t + lat (buy at the ask / sell at the bid); exit mark = touch mid at t + h; one
    position at a time (no new trade until the last one's horizon has passed). P&L in cents per contract, after fee."""
    from starter import asof
    d = F[["match_id", "t", f"dir{h}"]].copy()
    q = d[["match_id"]].assign(q=((d.t + lat) * 1e9).astype("int64"))
    r = asof(q, ctx["bb"].assign(bt=ctx["bb"].t_ns), "q", "t_ns", cols=["bid", "ask", "bt"])
    d["bid"], d["ask"] = r.bid.values, r.ask.values
    fresh = ((q.q.values - r.bt.values) / 1e9 <= 5) & (d.bid < d.ask)
    p = (d.bid + d.ask) / 2
    d["cost_c"] = 100 * (d.ask - d.bid) / 2 + 7 * p * (1 - p)
    d["exit"] = _touch_target(d.assign(t=d.t + h), ctx["bb"])
    go = fresh & np.isfinite(d.exit) & (d[f"dir{h}"].abs() > d.cost_c)
    rows = []
    for m, x in d[go].groupby("match_id"):
        last = -np.inf
        for t, f, b, a, e, pp in zip(x.t.values, x[f"dir{h}"].values, x.bid.values, x.ask.values, x.exit.values, ((x.bid + x.ask) / 2).values):
            if t < last + h: continue
            side = 1.0 if f > 0 else -1.0; entry = a if side > 0 else b
            rows.append({"match_id": m, "t": t, "side": side, "forecast_c": f, "pnl_c": 100 * side * (e - entry) - 7 * pp * (1 - pp)})
            last = t
    T = pd.DataFrame(rows, columns=["match_id", "t", "side", "forecast_c", "pnl_c"])
    T["test"] = T.match_id.isin(ctx["test"])
    return T


# =================================================================================================
# The final Part 5 models (moved here from part5.ipynb)
# =================================================================================================
def size_features(sport, Xb, XS):
    """Features for the SIZE model (|move|): unsigned activity (absolute values of the signed signals, spread, prints,
    touch age, event recency) + game state, with score differences as |diff| and two tension interactions."""
    act = pd.DataFrame({"log_prints_30s": Xb.log_prints_30s, "spread_c": Xb.spread_c, "log_touch_age": Xb.log_touch_age,
                        "abs_imb_2s": Xb.imb_2s.abs(), "abs_queue_imb": Xb.queue_imb.abs(), "abs_tape_mom_10s_c": Xb.tape_mom_10s_c.abs()})
    if sport == "CS2":
        act = act.assign(abs_feed_gap_c=Xb.feed_gap_c.abs(), abs_beter_jump_10s=Xb.beter_jump_10s_c.abs(),
                         abs_round_won_10s=Xb.round_won_10s.abs(), log_round_age=Xb.log_round_age)
    else:
        act = act.assign(bip_10s=Xb.bip_10s, betstop=Xb.betstop_x_bat.abs(), abs_run_10s=Xb.run_10s.abs(), abs_d_wp_30s=Xb.d_wp_30s_c.abs())
    st = XS[[c for c in FEATURE_GROUPS["state"] if c in XS.columns]].copy()
    st = st.assign(**{c: st[c].abs() for c in ("map_diff", "round_diff", "run_diff") if c in st})
    st["late_x_close"] = st.late * st.close; st["near_50_x_late"] = st.near_50 * st.late
    return act, st

def base_cols(sport):
    return FEATURE_GROUPS["tape"] + FEATURE_GROUPS["touch"] + FEATURE_GROUPS["context"] + FEATURE_GROUPS[f"feed_{sport}"]

def train_cv_r2(F, y, g, alpha, n_folds=5):
    """Grouped (by match) k-fold CV R² on TRAIN rows at a fixed Lasso penalty: the model-choice criterion."""
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    sse = sst = 0.0
    for a, b in GroupKFold(n_folds).split(F, y, g):
        sc = StandardScaler().fit(F.iloc[a].values)
        m = Lasso(alpha=alpha).fit(sc.transform(F.iloc[a].values), y[a])
        sse += ((y[b] - m.predict(sc.transform(F.iloc[b].values))) ** 2).sum(); sst += ((y[b] - y[a].mean()) ** 2).sum()
    return 1 - sse / sst

def design_matrices(sport, G, Xk, S, k):
    """(direction candidates, size candidates) for one sport and one feed shift k."""
    XSk = add_state(Xk, G, G.t.values, sport, S=S, k=k)
    act, st = size_features(sport, Xk, XSk)
    return ({"no game state": XSk[base_cols(sport)], "+ game state + signal × state": XSk},
            {"activity only": act, "activity + game state": pd.concat([act, st], axis=1)})


# ---- the CANCEL rule: choose x on TRAIN (most volume at break-even), evaluate on TEST ---------------------------------
CANCEL_QS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]

def choose_cancel_x(fl, grid_train_size, variant, sfx=""):
    """x = the least aggressive threshold (most volume kept) whose TRAIN P&L at +30 s is >= 0; if none breaks even,
    the threshold with the best train P&L per contract (flagged)."""
    thresholds = [np.inf] + list(grid_train_size.quantile(CANCEL_QS).values)
    tr_ = fl[~fl.test]; stats = []
    for x in thresholds:
        k_ = tr_[~cancel_mask(tr_, x, variant, "size10" + sfx, "dir10" + sfx)]
        stats.append((x, k_["size"].sum(), wmean_t(k_.pnl_book30.values, k_["size"].values, k_.match_id.values)[0] if len(k_) else np.nan))
    be = [z for z in stats if np.isfinite(z[2]) and z[2] >= 0]
    if be: return max(be, key=lambda z: z[1])[0], True
    return max(stats, key=lambda z: -np.inf if not np.isfinite(z[2]) else z[2])[0], False

def cancel_row(fl, x, variant, sfx=""):
    te = fl[fl.test]; cut = cancel_mask(te, x, variant, "size10" + sfx, "dir10" + sfx); kept, av = te[~cut], te[cut]
    f = lambda d, c: "—" if not len(d) else "{:+.2f} (t {:+.1f})".format(*wmean_t(d[c].values, d["size"].values, d.match_id.values))
    return {"volume kept": kept["size"].sum() / te["size"].sum(), "always-on +30s": f(te, "pnl_book30"), "with rule +30s": f(kept, "pnl_book30"),
            "avoided fills +30s": f(av, "pnl_book30"), "with rule +120s": f(kept, "pnl_book120"), "with rule settle": f(kept, "pnl_settle")}

def cancel_frontier_figure(FILLS, FALL, CHOSEN, h_c, path):
    import matplotlib.pyplot as plt
    from p3_leadlag import INK, INK2, GRID, C_TEAM1, C_TEAM2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, s in zip(axes, ("CS2", "MLB")):
        th = [np.inf] + sorted(FALL[s][~FALL[s].test].size10.quantile(CANCEL_QS).values, reverse=True)
        for strict, ls in ((False, "-"), (True, "--")):
            te = FILLS[(s, h_c, strict)]; te = te[te.test]
            for variant, col in (("both", C_TEAM2), ("one-sided", C_TEAM1)):
                pts = []
                for x in th:
                    kept = te[~cancel_mask(te, x, variant)]
                    pts.append((100 * kept["size"].sum() / te["size"].sum(), wmean_t(kept.pnl_book30.values, kept["size"].values, kept.match_id.values)[0]))
                pts = np.array(pts); pts = pts[pts[:, 0] >= 3]
                ax.plot(pts[:, 0], pts[:, 1], ls, marker="o", ms=3.5, color=col, label=f"{variant} · {'strict' if strict else 'optimistic'} fills")
                kept = te[~cancel_mask(te, CHOSEN[(s, h_c, strict, variant)], variant)]
                if kept["size"].sum() >= 0.03 * te["size"].sum():
                    ax.plot(100 * kept["size"].sum() / te["size"].sum(), wmean_t(kept.pnl_book30.values, kept["size"].values, kept.match_id.values)[0],
                            "*", ms=13, color=col, mec=INK, mew=0.8, zorder=5)
        ax.axhline(0, color=INK, lw=0.8); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.invert_xaxis()
        ax.set_title(s, loc="left", fontweight="bold"); ax.set_xlabel("volume kept (% of always-on contracts)")
    axes[0].set_ylabel("P&L per contract, marked to touch mid +30 s (cents)"); axes[1].legend(fontsize=8.5, frameon=False)
    fig.text(0.01, 0.985, f"CANCEL rule on TEST ({h_c}c half-spread): each point = one size-forecast threshold (right = cancel more)", fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.01, 0.935, "★ = threshold chosen on TRAIN · solid = optimistic fills, dashed = strict fills · points with < 3% of volume not shown", fontsize=9, color=INK2, va="top")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.11, wspace=0.18); fig.savefig(path, dpi=130); plt.show()
