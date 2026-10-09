"""Part 1 (final): tape-only estimators of the current Kalshi mid. `fit_part1` tunes and fits the two winners on TRAIN.

Pipeline for any sport's in-play prints `tr` (from starter.trades_on_axis, index reset):
    params = load_params()[sport]                      # tuned on TRAIN in part1.ipynb
    add_estimators(tr, params["H"], params["W"])        # per-print ewma / vwap columns
    Q = query_prints(tr, times)                         # one row per (match_id, t) query time
    Q["mid_est"] = predict_lasso(Q, tr, params)         # the Part 1 winner at each query time
Everything at query time t uses prints stamped <= t - LAT only.
"""
import os
import numpy as np, pandas as pd
import joblib
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from starter import D
from mm_core import tape_mid, HALF_SPREAD

LAT = 0.10                                   # seconds, same as mm_core.visible_index
BASE_WIN, MIN_BASE = 1800.0, 60.0            # 30-min trailing baseline for relvol; floor early in a match
FEATURES = ["dir", "ewma_gap", "vwap_gap", "accel", "relvol5", "orders3", "stale", "imb5",
            "dir_x_stale", "dir_x_accel", "imb_x_relvol"]
FEATURES_3B = FEATURES + ["dir_x_spread"]     # Model 3b (CS2): Model 3 + dir x Model 5 spread
FEATURES_6 = ["dir", "ewma_raw_gap", "spread_hat", "dir_x_spread", "accel", "relvol5", "orders3", "stale", "imb5",
              "dir_x_stale", "dir_x_accel", "imb_x_relvol"]   # Model 6 (MLB winner)
PARAMS_PATH = "output/part1_params.joblib"


# ---- grids ------------------------------------------------------------------------------------
def _print_state(tr, t, lat=LAT):
    """For one match's prints `tr` (time-sorted) and query times t (s): the last print visible at each t."""
    tp = tr.t_ns.values / 1e9      # print times, seconds (Kalshi exchange clock)
    px = tr.px.values              # print prices on the match axis: P(team1 / home wins)
    d  = tr["dir"].values          # aggressor side of each print: +1 taker bought team1/home, -1 sold
    j = np.searchsorted(tp, t - lat, side="right") - 1   # per query time: index of the last print VISIBLE at t
                                                         #   (stamped <= t - lat); -1 = none yet
    return tp, px, d, j

def grid_match(tr, bb, step=1.0, lat=LAT):
    """tr = one match's in-play prints (time-sorted), bb = the same match's touch rows (time-sorted)."""
    tb = bb.t_ns.values / 1e9      # touch-change times, seconds (collector clock, ~6 ms late; negligible at 1 s)
    t0, t1 = tr.t_ns.values[0] / 1e9, tr.t_ns.values[-1] / 1e9
    t  = np.arange(t0, t1, step)   # the evaluation grid: every `step` s from first to last in-play print
    tp, px, d, j = _print_state(tr, t, lat)
    k = np.searchsorted(tb, t, side="right") - 1         # per grid time: index of the touch row in force at t
                                                         #   (the target); -1 = no touch yet
    ok = (j >= 0) & (k >= 0)
    j, k, t = j[ok], k[ok], t[ok]

    bid, ask = bb.bid.values[k], bb.ask.values[k]
    return pd.DataFrame({
        "t": t, "mid": (bid + ask) / 2, "stale_s": t - tb[k], "crossed": bid >= ask,
        "j": j,                    # position of the last visible print WITHIN this match
        "row": tr.index.values[j], # same print as a row label of tr_cs2 / tr_mlb: est.values[G.row]
        "since_print_s": t - tp[j],
        "last_px": px[j],
        "tape_mid": tape_mid(px, d)[j],
    })

def build_grid(tr, bb, ids, max_stale=5.0):
    """One grid for a set of match ids. Drops stale (> max_stale s) and locked/crossed touches."""
    bb_by = dict(tuple(bb[bb.match_id.isin(ids)].groupby("match_id")))
    grids = [grid_match(g, bb_by[m]).assign(match_id=m)
             for m, g in tr[tr.match_id.isin(ids)].groupby("match_id") if m in bb_by]
    G = pd.concat(grids, ignore_index=True)
    return G[(G.stale_s <= max_stale) & ~G.crossed].reset_index(drop=True)

def query_prints(tr, times, lat=LAT):
    """Like grid_match without a touch: `times` has columns match_id, t (seconds). Returns the rows that
    have a visible print, with j, row, since_print_s, last_px, tape_mid (+ any other columns of `times`)."""
    out = []
    pos = tr.groupby("match_id").indices
    for m, q in times.groupby("match_id"):
        if m not in pos: continue
        g = tr.iloc[pos[m]]
        tp, px, d, j = _print_state(g, q.t.values, lat)
        ok = j >= 0; q = q[ok].copy(); j = j[ok]
        q["j"], q["row"] = j, g.index.values[j]
        q["since_print_s"], q["last_px"], q["tape_mid"] = q.t.values - tp[j], px[j], tape_mid(px, d)[j]
        out.append(q)
    return pd.concat(out, ignore_index=True)


# ---- scoring ----------------------------------------------------------------------------------
_RESULT = None
def match_results():
    """'yes' / 'no' / 'scalar' per match id: team1 (esports) or home (MLB) won."""
    global _RESULT
    if _RESULT is None:
        es = pd.read_parquet(D + "map_esports.parquet").set_index("beter_match_id").result_team1
        mlb = pd.read_parquet(D + "map_mlb.parquet").set_index("sr_match_id").result_home
        _RESULT = pd.concat([es, mlb])
    return _RESULT

def clustered_mean_se(x, groups):
    """Mean of x and its standard error clustered by group."""
    x = np.asarray(x, float); n = len(x); mu = x.mean()
    s = pd.Series(x - mu).groupby(np.asarray(groups)).sum().values
    return mu, np.sqrt((s ** 2).sum()) / n

def score(G, est, target="mid"):
    """RMSE / bias of G[est] against G[target], clustered by match; Brier against the match result."""
    e = G[est] - G[target]
    bias, bias_se = clustered_mean_se(e, G.match_id)
    mse, mse_se = clustered_mean_se(e ** 2, G.match_id)
    rmse = np.sqrt(mse); rmse_se = mse_se / (2 * rmse) if rmse > 0 else 0.0   # delta method
    y = G.match_id.map(match_results()); keep = y.isin(["yes", "no"])
    brier = ((G.loc[keep, est] - (y[keep] == "yes")) ** 2).mean()
    return {"RMSE_c": 100 * rmse, "RMSE_se_c": 100 * rmse_se, "bias_c": 100 * bias, "bias_se_c": 100 * bias_se,
            "Brier": brier, "matches": G.match_id.nunique(), "points": len(G)}


# ---- Model 1 and 2: per-print smoothers ---------------------------------------------------------
def ewma_by_match(tr, H, h=HALF_SPREAD):
    """Per-print EWMA of x = px - dir*h with a time half-life of H seconds; restarts every match."""
    x = pd.Series(tr.px.values - tr["dir"].values * h, index=tr.index)
    times = pd.to_datetime(tr.t_ns)
    return x.groupby(tr.match_id).transform(
        lambda s: s.ewm(halflife=pd.Timedelta(H, "s"), times=times[s.index]).mean())

def vwap_by_match(tr, W, h=HALF_SPREAD):
    """Per-print size-weighted mean of x = px - dir*h over prints with t_k >= t_i - W (k <= i); restarts every match."""
    out = np.empty(len(tr))
    for _, idx in tr.groupby("match_id").indices.items():      # idx: positions of this match's prints, in time order
        t = tr.t_ns.values[idx] / 1e9; q = tr["count"].values[idx]
        x = tr.px.values[idx] - tr["dir"].values[idx] * h
        cqx = np.concatenate([[0.0], np.cumsum(q * x)]); cq = np.concatenate([[0.0], np.cumsum(q)])
        s = np.searchsorted(t, t - W, side="left")              # first print inside each window
        i = np.arange(len(t)) + 1
        out[idx] = (cqx[i] - cqx[s]) / (cq[i] - cq[s])
    return out

def spread_by_match(tr, N, s0=2 * HALF_SPREAD):
    """Model 4, per print: tape-implied spread = mean price of BUY prints (they trade at the ask) minus mean price
    of SELL prints (they trade at the bid), over prints k <= i with t_k in (t_i - N, t_i]; restarts every match.
    Carried forward when the window lacks a buy or a sell; s0 (2c) before the first usable window; floored at 0."""
    out = np.empty(len(tr))
    for _, idx in tr.groupby("match_id").indices.items():
        t = tr.t_ns.values[idx] / 1e9; p = tr.px.values[idx]; buy = tr["dir"].values[idx] > 0
        csum = lambda v: np.r_[0.0, np.cumsum(v)]
        sum_b, n_b, sum_s, n_s = csum(p * buy), csum(buy), csum(p * ~buy), csum(~buy)
        s = np.searchsorted(t, t - N, side="right"); i = np.arange(len(t)) + 1     # window = prints s..i-1
        nb, ns = n_b[i] - n_b[s], n_s[i] - n_s[s]
        spread = np.where((nb > 0) & (ns > 0),
                          (sum_b[i] - sum_b[s]) / np.maximum(nb, 1) - (sum_s[i] - sum_s[s]) / np.maximum(ns, 1), np.nan)
        out[idx] = pd.Series(spread).ffill().fillna(s0).clip(lower=0).values
    return out

def spread_mid_by_match(tr, N):
    """Model 4 estimate per print: px - dir * spread / 2 (buy at the ask -> subtract, sell at the bid -> add)."""
    return tr.px.values - tr["dir"].values * spread_by_match(tr, N) / 2

def spread_lastn_by_match(tr, n, s0=2 * HALF_SPREAD):
    """Model 5, per print: tape-implied spread = mean price of the last n BUY prints (asks) minus mean price of the
    last n SELL prints (bids), among prints k <= i, with no time limit; restarts every match. n = 1: last ask - last bid.
    s0 (2c) until the match has at least one buy and one sell; floored at 0."""
    out = np.empty(len(tr))
    for _, idx in tr.groupby("match_id").indices.items():
        p = tr.px.values[idx]; buy = tr["dir"].values[idx] > 0
        cum_b, cum_s = np.r_[0.0, np.cumsum(p[buy])], np.r_[0.0, np.cumsum(p[~buy])]
        k_b, k_s = np.cumsum(buy), np.cumsum(~buy)              # buys / sells seen up to and including print i
        avg_ask = (cum_b[k_b] - cum_b[np.maximum(k_b - n, 0)]) / np.maximum(np.minimum(k_b, n), 1)
        avg_bid = (cum_s[k_s] - cum_s[np.maximum(k_s - n, 0)]) / np.maximum(np.minimum(k_s, n), 1)
        out[idx] = np.where((k_b > 0) & (k_s > 0), np.clip(avg_ask - avg_bid, 0, None), s0)
    return out

def spread_lastn_mid_by_match(tr, n):
    """Model 5 estimate per print: px - dir * spread / 2."""
    return tr.px.values - tr["dir"].values * spread_lastn_by_match(tr, n) / 2

def add_estimators(tr, H, W):
    """Add the tuned per-print `ewma` and `vwap` columns to tr (in place)."""
    tr["ewma"] = ewma_by_match(tr, H).values
    tr["vwap"] = vwap_by_match(tr, W)


# ---- Model 3: Lasso correction to tape_mid -----------------------------------------------------
def add_features(G, tr, c, lat=LAT):
    """Add the Model 3 features to grid G (in place). Windows end at a = t - lat and use prints <= a only.
    Needs G columns match_id, t, row, since_print_s, tape_mid and tr columns ewma, vwap."""
    pos = tr.groupby("match_id").indices                      # match -> positions of its prints (time order)
    for m, gi in G.groupby("match_id").indices.items():
        idx = pos[m]
        tp = tr.t_ns.values[idx] / 1e9; q = tr["count"].values[idx]; dq = tr["dir"].values[idx] * q
        cq = np.concatenate([[0.0], np.cumsum(q)]); cdq = np.concatenate([[0.0], np.cumsum(dq)])
        ut = np.unique(tp)                                    # distinct timestamps = taker orders
        a = G.t.values[gi] - lat
        n = lambda x: np.searchsorted(tp, x, side="right")    # prints stamped <= x
        V = lambda tau: cq[n(a)] - cq[n(a - tau)]             # contracts in (a - tau, a]
        V1, V5, V30 = V(1.0), V(5.0), V(30.0)
        base_len = np.clip(a - tp[0], MIN_BASE, BASE_WIN)
        rate = (cq[n(a)] - cq[n(a - base_len)]) / base_len    # contracts per second, trailing
        G.loc[G.index[gi], "accel"]   = np.log((V1 + c) / (V30 / 30 + c))
        G.loc[G.index[gi], "relvol5"] = np.log1p(V5 / (5 * np.maximum(rate, 1e-9)))
        G.loc[G.index[gi], "orders3"] = np.log1p(np.searchsorted(ut, a, "right") - np.searchsorted(ut, a - 3, "right"))
        G.loc[G.index[gi], "imb5"]    = np.where(V5 > 0, (cdq[n(a)] - cdq[n(a - 5)]) / np.where(V5 > 0, V5, 1), 0.0)
    G["dir"]      = tr["dir"].values[G.row]
    G["ewma_gap"] = 100 * (tr["ewma"].values[G.row] - G.tape_mid)   # in cents
    G["vwap_gap"] = 100 * (tr["vwap"].values[G.row] - G.tape_mid)   # in cents
    G["stale"]    = np.log1p(G.since_print_s)
    G["dir_x_stale"], G["dir_x_accel"] = G.dir * G.stale, G.dir * G.accel
    G["imb_x_relvol"] = G.imb5 * G.relvol5

def fit_lasso_1se(X, y, groups, n_folds=5):
    """Lasso on standardized X; alpha by match-grouped CV with the one-standard-error rule."""
    scaler = StandardScaler().fit(X); Z = scaler.transform(X)
    cv = LassoCV(cv=list(GroupKFold(n_folds).split(Z, y, groups)), n_alphas=60, n_jobs=-1).fit(Z, y)
    mse = cv.mse_path_.mean(axis=1); se = cv.mse_path_.std(axis=1) / np.sqrt(n_folds)
    i_min = mse.argmin()
    alpha_1se = cv.alphas_[mse <= mse[i_min] + se[i_min]].max()      # largest alpha within 1 SE of the best
    model = Lasso(alpha=alpha_1se).fit(Z, y)
    return scaler, model, cv.alpha_

def predict_lasso(G, tr, params):
    """The Part 1 winner at each row of G: tape_mid + Lasso correction, clipped to [0, 1]. Adds features to G."""
    add_features(G, tr, params["c"])
    corr_c = params["model"].predict(params["scaler"].transform(G[FEATURES].values))
    return np.clip(G.tape_mid.values + corr_c / 100, 0, 1)


def add_features_3b(G, tr, c, n_last, lat=LAT):
    """Model 3b features on G: Model 3's features plus dir_x_spread = dir * (Model 5 tape-implied spread, cents)."""
    add_features(G, tr, c, lat)
    spread = spread_lastn_by_match(tr, n_last)
    G["dir_x_spread"] = G.dir * 100 * spread[G.row]

def predict_lasso3b(G, tr, params):
    """Model 3b at each row of G: tape_mid + Lasso correction (FEATURES_3B), clipped to [0, 1]. Adds features to G.
    params: the sport's entry from load_params(); needs c, n_last and params["lasso3b"] = {"scaler", "model"}.
    tr needs the ewma / vwap columns (add_estimators with the saved H, W)."""
    add_features_3b(G, tr, params["c"], int(params["n_last"]))
    m = params["lasso3b"]
    corr_c = m["model"].predict(m["scaler"].transform(G[FEATURES_3B].values))
    return np.clip(G.tape_mid.values + corr_c / 100, 0, 1)


def ewma_raw_by_match(tr, alpha):
    """Per-print EWMA of the RAW print price, decayed per trade: E_k = alpha*p_k + (1-alpha)*E_{k-1}; restarts every match."""
    return tr.groupby("match_id").px.transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean()).values

def predict_lasso6(G, tr, params):
    """Model 6 (MLB winner) at each row of G: Model 5 baseline + Lasso correction (FEATURES_6), clipped to [0, 1].
    params: the sport's entry from load_params(); needs c, n_last and params["lasso6"] = {"alpha", "scaler", "model"}.
    tr needs the ewma / vwap columns (add_estimators with the saved H, W). Adds features to G."""
    add_features(G, tr, params["c"])
    n, m = int(params["n_last"]), params["lasso6"]
    m5 = spread_lastn_mid_by_match(tr, n)
    G["m5"] = m5[G.row]
    G["ewma_raw_gap"] = 100 * (ewma_raw_by_match(tr, m["alpha"])[G.row] - G.m5)      # cents
    G["spread_hat"] = 100 * spread_lastn_by_match(tr, n)[G.row]                         # cents
    G["dir_x_spread"] = G.dir * G.spread_hat
    corr_c = m["model"].predict(m["scaler"].transform(G[FEATURES_6].values))
    return np.clip(G.m5.values + corr_c / 100, 0, 1)


# ---- persistence --------------------------------------------------------------------------------
def save_params(params, path=PARAMS_PATH):
    """params: {sport: {"H", "W", "N", "n_last", "c", "scaler", "model", optional "lasso3b"}}."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(params, path)

def load_params(path=PARAMS_PATH):
    return joblib.load(path)


# ---- the final Part 1 pipeline: tune on TRAIN, fit the winners, score on TEST -------------------------------------------
HALF_LIVES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 5, 10, 30, 120]     # Model 1 EWMA half-life candidates (s)
WINDOWS = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60]                     # Model 2 VWAP window candidates (s)
LAST_N = [1, 2, 3, 5, 10, 20, 50]                                     # Model 5 last-n-prints candidates
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]                         # Model 6 per-trade EWMA weight candidates

def part1_data(sport):
    """In-play prints and touch for the matches with a book_ok recording (the Part 1 scoring sample)."""
    from starter import trades_on_axis, test_match_ids
    bbo = pd.read_parquet(D + "kalshi_bbo.parquet"); bq = pd.read_parquet(D + "book_quality.parquet")
    ok_ids = set(bbo.loc[bbo.event_ticker.isin(set(bq.loc[bq.book_ok, "event_ticker"])), "match_id"])
    if sport == "CS2":
        mp = pd.read_parquet(D + "map_esports.parquet"); ids = set(mp.loc[mp.sport_id == 3, "beter_match_id"])
        tr = trades_on_axis("esports"); tr = tr[tr.match_id.isin(ids)]; test = test_match_ids("esports") & ids
    else:
        tr = trades_on_axis("mlb"); test = test_match_ids("mlb")
    tr = tr[tr.match_id.isin(ok_ids) & tr.inplay].reset_index(drop=True)
    bb = bbo[(bbo.sport == sport) & bbo.match_id.isin(ok_ids)].sort_values("t_ns").reset_index(drop=True)
    return tr, bb, test

def fit_part1(sport, tr, bb, test_ids):
    """Tune every hyperparameter on TRAIN grid seconds (RMSE vs the touch mid), fit the sport's winner
    (CS2: Model 3b, MLB: Model 6) with the Lasso 1-SE rule, and return (params, train grid, test grid, tuning table)."""
    ids = set(tr.match_id)
    Gtr, Gte = build_grid(tr, bb, ids - test_ids), build_grid(tr, bb, ids & test_ids)
    rmse = lambda p: 100 * np.sqrt(((p - Gtr.mid.values) ** 2).mean())
    tune = []
    for H in HALF_LIVES: tune.append(("H (EWMA half-life, s)", H, rmse(ewma_by_match(tr, H).values[Gtr.row])))
    for W in WINDOWS: tune.append(("W (VWAP window, s)", W, rmse(vwap_by_match(tr, W)[Gtr.row])))
    for n in LAST_N: tune.append(("n (last-n spread)", n, rmse(spread_lastn_mid_by_match(tr, n)[Gtr.row])))
    if sport == "MLB":
        for a in ALPHAS: tune.append(("alpha (raw EWMA)", a, rmse(ewma_raw_by_match(tr, a)[Gtr.row])))
    T = pd.DataFrame(tune, columns=["hyperparameter", "value", "train_RMSE_c"])
    best = T.loc[T.groupby("hyperparameter").train_RMSE_c.idxmin()].set_index("hyperparameter").value
    P = {"H": float(best["H (EWMA half-life, s)"]), "W": float(best["W (VWAP window, s)"]), "n_last": int(best["n (last-n spread)"])}
    add_estimators(tr, P["H"], P["W"])
    P["c"] = tr.loc[~tr.match_id.isin(test_ids), "count"].median()             # median TRAIN print size
    if sport == "CS2":
        for G in (Gtr, Gte): add_features_3b(G, tr, P["c"], P["n_last"])
        y = 100 * (Gtr.mid - Gtr.tape_mid).values
        scaler, model, _ = fit_lasso_1se(Gtr[FEATURES_3B].values, y, Gtr.match_id.values)
        P["lasso3b"] = {"scaler": scaler, "model": model}
        for G in (Gtr, Gte): G["winner"] = predict_lasso3b(G, tr, P)
    else:
        a = float(best["alpha (raw EWMA)"])
        m5, raw, sp = spread_lastn_mid_by_match(tr, P["n_last"]), ewma_raw_by_match(tr, a), spread_lastn_by_match(tr, P["n_last"])
        for G in (Gtr, Gte):
            add_features(G, tr, P["c"])
            G["m5"] = m5[G.row]; G["ewma_raw_gap"] = 100 * (raw[G.row] - G.m5); G["spread_hat"] = 100 * sp[G.row]
            G["dir_x_spread"] = G.dir * G.spread_hat
        y = 100 * (Gtr.mid - Gtr.m5).values
        scaler, model, _ = fit_lasso_1se(Gtr[FEATURES_6].values, y, Gtr.match_id.values)
        P["lasso6"] = {"alpha": a, "scaler": scaler, "model": model}
        for G in (Gtr, Gte): G["winner"] = predict_lasso6(G, tr, P)
    for G in (Gtr, Gte):                                                        # simpler estimators for the scoreboard
        G["ewma"] = tr["ewma"].values[G.row]; G["vwap"] = tr["vwap"].values[G.row]
        G["lastn_mid"] = spread_lastn_mid_by_match(tr, P["n_last"])[G.row]
    return P, Gtr, Gte, T

def residual_figure(panels, path):
    """Box plots of TEST residuals (estimate - touch mid, cents). panels: [(sport, grid, [(label, column, color), ...])]."""
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 5), facecolor=SURFACE)
    for ax, (sport, G, ests) in zip(np.atleast_1d(axes), panels):
        ax.set_facecolor(SURFACE)
        res = [100 * (G[col] - G.mid).values for _, col, _ in ests]
        bp = ax.boxplot(res, widths=0.5, showfliers=False, patch_artist=True, whis=1.5, medianprops={"color": INK, "linewidth": 2})
        for i, (_, _, c) in enumerate(ests):
            bp["boxes"][i].set(facecolor=c + "33", edgecolor=c, linewidth=1.5)
            for part in ("whiskers", "caps"):
                for line in bp[part][2 * i: 2 * i + 2]: line.set(color=c, linewidth=1.5)
            ax.plot(i + 1, res[i].mean(), "D", ms=7, color=c, mec=SURFACE, mew=1.2, zorder=3)
        ax.axhline(0, color=INK2, linewidth=1, linestyle="--", zorder=0)
        ax.set_xticks(range(1, len(ests) + 1))
        ax.set_xticklabels([f"{n}\nRMSE {np.sqrt((r ** 2).mean()):.2f}c · IQR {np.subtract(*np.percentile(r, [75, 25])):.2f}c"
                            for (n, _, _), r in zip(ests, res)], fontsize=9)
        ax.set_title(f"{sport} · {G.match_id.nunique()} test matches · {len(G):,} seconds", fontsize=10, color=INK2, loc="left")
        ax.set_ylabel("residual: estimate − touch mid (cents)", color=INK2)
        ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.suptitle("Part 1 winners vs the tape-mid baseline: residuals on TEST", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.905, "box = IQR, line = median, whiskers = 1.5×IQR (outliers hidden), diamond = mean", color=INK2, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(path, dpi=150); plt.show()
