# final_code: the functions behind `final_report.ipynb`

Run from the `homework2` directory (data paths are relative) and put this folder first on the path:

```python
import sys; sys.path.insert(0, "final_code")
```

| Module | Part | What it holds |
|---|---|---|
| `starter.py`, `mm_core.py` | all | the provided helpers, **unchanged** (one axis per match, the train/test split, leak-proof `asof`, `visible_index`, `simulate_maker`) |
| `p1_mid.py` | 1 | tape-only mid estimators (EWMA, VWAP, last-n implied spread, Lasso corrections), the 1 s scoring grid, `score` (match-clustered SEs), and `part1_data` + `fit_part1`: tune every hyperparameter on TRAIN and fit the winners (CS2 Model 3b, MLB Model 6) |
| `p2_feed.py` | 2 | the MLB win-probability model (`fit_wp_model`: structural win expectancy + CV-chosen XGBoost), the feed-vs-mid comparison grids (`cs2_comparison_grid`, `mlb_comparison_grid`), bias / slope / RMSE, the train-fitted blend (`blend_table`) and the "when to stop trusting the feed" forest plot |
| `p3_leadlag.py` | 3 | `sport_context` (prints + Part 1 estimators + book_ok touch, used by Parts 3–6), feed events with feed-side directions, the event study (`study`) on both clocks, `lead_table`, charts |
| `p4_latency.py` | 4 | the feed delivered k s earlier: lead vs k (`lead_at_k`, `k_needed`) and blend weight vs k with a paired bootstrap of its peak (`blend_curve`) |
| `p5_forecast.py` | 5 | causal features (no touch mid in any feature), Lasso direction / size models, train-CV model choice, test R² with match-bootstrap CIs, the CANCEL rule (`maker_fills`, `choose_cancel_x`) and the TAKE rule |
| `p6_market.py` | 6 | the queue simulator against the real touch (`Sim`), quoting policy (`policy_quotes`, `version_quotes`), greedy train selection, frontier and paired-bootstrap break-even volume |

Rules applied everywhere (DATA.md): split by match on `kalshi_start`; feeds timed by `t_recv_ns` only; BETER incidents `msg_type == 1`,
Sportradar `feedtype == "delta"`; decisions use prints ≥ 0.1 s old; the touch is known only if its last change is ≤ 5 s old and uncrossed
(event windows and P&L marks accept ≤ 30 s, a stated compromise with a robustness check); void matches dropped before settlement marks;
standard errors clustered by match.
