---
output:
  pdf_document: default
  html_document: default
---
# BETER Internal Microstructure
## How quickly and how far does the bookmaker reprice?

**Assignment variant:** Option C, BETER-internal microstructure  
**Capture:** 2026-08-28 to 2026-08-29  
**Primary comparison:** Table Tennis versus Fifa/eFootball

## Executive answer

BETER's machine-fed Fifa stream reprices almost immediately after an incident, while its human-keyed table-tennis stream is much slower and far more dispersed.

| Result | Estimate |
|---|---:|
| Synthetic lag-estimation error floor | **1.0 s maximum** |
| Fifa median incident-to-probability-change latency | **0.25 s** per-match median |
| Table-tennis median incident-to-probability-change latency | **76.40 s** per-match median |
| Table-tennis median absolute probability jump | **1.31 percentage points** |
| Table-tennis 90th-percentile absolute jump | **8.4 percentage points** |
| Table-tennis calibration sample | **374 settled matches** |
| Table-tennis Brier score | **0.1451** |
| Table-tennis log loss | **0.4432** |

The evidence strongly supports an operational difference between the two feeds. It is consistent with the hypothesis that human scouting is slower than machine event generation. It is not, by itself, proof that human entry explains the entire difference: BETER's market-update cadence, event definitions, and recovery traffic can also contribute.

The bookmaker does not move by a fixed amount after every event. In table tennis, the typical probability move is modest, but the upper tail is large, especially when the score is tied or close. The probabilities are useful but not perfectly calibrated: the reliability curve departs from the 45-degree line in several buckets.

The central conclusion is therefore precise: **BETER's internal feed contains a measurable event-to-market delay, especially for table tennis, and that delay is economically meaningful enough to motivate further study of whether stale odds could be traded.** This Option C analysis does not claim a Kalshi edge or a PnL result; those belong to the separate cross-venue assignment variants.

---

## 1. Data and research design

The manifest contains 2,818 fixtures:

- 755 TableTennis matches
- 2,060 Fifa matches
- 3 CounterStrike matches

The parsed BETER capture contains 135,723 incident records, 59,650 scoreboard states, and 11.8 million trading rows containing probabilities or settlement information.

For every match, the notebook joins:

1. **`incident`**: the event stream, ordered by `incident.index` within a match;
2. **`trading`**: BETER market states and the margin-free `probability` field;
3. **`time_table` manifest**: match identity, participants, sport, and league.

The primary response observation is:

$$
\text{response latency}
= t_{\text{first changed probability after incident}}
- t_{\text{incident arrival}}.
$$

The probability immediately before the incident is carried forward, then the first later state with a changed probability is selected. Unchanged snapshots are not counted as repricings.

### Clock discipline

The analysis uses the capture clocks, not the provider's state timestamp:

- `recv_wall_ns` is the cross-channel arrival clock;
- `recv_mono_ns` is reserved for duration calculations within the capture;
- `incident.index` orders incidents within a match;
- trading states are deduplicated using the full state key `(match_id, offset, market_id, outcome_id)`.

The provider `timestamp` is deliberately not used as an event clock because it represents source state and can move backward across message types.

---

## 2. Part 0: calibration of the timing method

The supplied synthetic generator creates three matches with known delays of 3.0, 4.5, and 6.0 seconds. The notebook runs the synthetic data through the same parser and 1-second-grid cross-correlation logic used for timing work.

| Synthetic match | True lag | Estimated lag | Error |
|---|---:|---:|---:|
| Demo TT 1 | 3.0 s | 3 s | 0.0 s |
| Demo TT 2 | 4.5 s | 5 s | +0.5 s |
| Demo TT 3 | 6.0 s | 5 s | -1.0 s |

The maximum absolute error is **1.0 second**. This is the method's practical resolution floor. Real-feed results should not be interpreted more precisely than that without a higher-resolution estimator.

---

## 3. How fast does BETER reprice?

The comparison uses the focal home-side probability for each sport's main result market. For table tennis, this is the binary match-winner market. For Fifa, the selected series is the home-win outcome in the captured three-way match-result market.

The response sample contains 12,280 Fifa observations across 660 matches and 66,566 table-tennis observations across 368 matches.

![Incident-to-repricing latency comparison](analysis/option_c_results/figures/latency_comparison.png)

**What the plot shows.** The left panel compares the central part of each latency distribution, omitting observations above 300 seconds so the boxes remain readable. The right panel is an empirical cumulative distribution: at each time $t$, it shows the share of incidents followed by a changed probability by time $t$.

The separation is substantial:

| Sport | Median event latency | Median of per-match medians | P90 event latency |
|---|---:|---:|---:|
| Fifa/eFootball | 0.20 s | 0.25 s | 16.87 s |
| Table Tennis | 70.89 s | 76.40 s | 205.77 s |

The sport comparison is not a small timing difference. The typical table-tennis response is roughly five minutes slower than the typical Fifa response. The long TT tail also indicates that a market participant cannot treat every incident as immediately incorporated into the odds.

A match-level bootstrap was used for uncertainty because incidents within a match are correlated. The 95% bootstrap intervals for the median of per-match medians were approximately:

| Sport | Median | 95% match-level interval |
|---|---:|---:|
| Fifa/eFootball | 0.25 s | 0.23 to 0.39 s |
| Table Tennis | 76.40 s | 73.0 to 80.5 s |

These intervals are narrow relative to the difference between sports.

### Interpretation

The result is consistent with the expected production pipeline:

- Fifa events are machine-fed and can trigger an almost immediate odds response.
- TT events are keyed by a human scout, so the event itself may arrive later and less regularly.
- The observed TT delay may also include batching, market suspension, recovery messages, or a slower odds-update schedule.

The data identify the end-to-end incident-to-trading delay. They do not isolate the exact contribution of the scout, the odds engine, or downstream transport.

---

## 4. How far does the probability move?

Across the response observations, the median absolute probability jump was:

- **1.31 percentage points for Table Tennis**;
- **0.89 percentage points for Fifa**.

The upper tails are more informative than the medians. For TT, the 90th-percentile absolute jump was **8.4 percentage points**. This means a normal point often produces a small movement, but a sufficiently informative event can move the match-winner probability by a large amount.

![Table-tennis repricing by score state](analysis/option_c_results/figures/score_state_repricing.png)

**What the plot shows.** Bars are median absolute probability jumps. Orange points are the 90th percentile. The score state is measured using the absolute point margin observed in the incident parameters.

| Score state | Median jump | P90 jump | Median response latency |
|---|---:|---:|---:|
| Tied | 1.31 pp | 9.18 pp | 100.2 s |
| 1-2 points apart | 1.29 pp | 8.86 pp | 84.6 s |
| 3-5 points apart | 1.31 pp | 6.03 pp | 41.0 s |
| 6+ points apart | 1.29 pp | 3.51 pp | 22.8 s |

The median jump is similar across states because many ordinary point events are small. The upper tail is clearly state-dependent: when a match is close, a point can change the winner's prospects much more than when one player already leads comfortably. The latency pattern is also informative: large existing leads are followed by faster observed changes, while close states have a longer response time in this capture.

This is economically sensible. Information value depends on the state of the match. A point at 0-0 and a point that creates a decisive lead are not equivalent shocks, even if they share the same incident label.

---

## 5. Are the probabilities calibrated?

For the table-tennis match-winner market, the notebook takes the last active focal probability before settlement and compares it with the realized result. Settlement codes `2` and `3` are treated as win and loss for the focal outcome; unresolved or nonstandard result codes are excluded.

The calibration sample contains **374 settled matches**.

![Table-tennis probability calibration](analysis/option_c_results/figures/tt_calibration.png)

**What the plot shows.** The left panel is a reliability curve. Perfect calibration lies on the dashed 45-degree line: a group of matches forecast at 70% should win approximately 70% of the time. The right panel shows the distribution of final active probabilities.

The summary metrics are:

- **Brier score: 0.1451**
- **Log loss: 0.4432**

The reliability table shows the main pattern:

| Probability bucket | Mean forecast | Realized win rate | Matches |
|---|---:|---:|---:|
| 0-10% | 3.8% | 0.0% | 41 |
| 10-20% | 14.8% | 0.0% | 11 |
| 20-30% | 25.0% | 4.3% | 23 |
| 30-40% | 36.8% | 0.0% | 20 |
| 40-50% | 46.8% | 32.2% | 118 |
| 50-60% | 54.9% | 76.1% | 71 |
| 60-70% | 64.2% | 91.7% | 24 |
| 70-80% | 74.7% | 95.7% | 23 |
| 80-90% | 85.9% | 100.0% | 10 |
| 90-100% | 97.0% | 100.0% | 33 |

The middle buckets are noisy, but the pattern is not perfectly diagonal. In this sample, BETER appears conservative in several higher-probability buckets: outcomes occurred more often than the stated probability suggested. Small bucket sizes, selection of the final active state, and the fact that probabilities can be close to 0 or 1 all limit how strongly this should be interpreted.

Calibration is therefore useful evidence about forecast quality, not a claim that BETER's displayed price is directly tradeable. The quoted odds can still contain a margin even when the underlying probability estimate is reasonably informative.

---

## 6. Data-quality and identification limits

Several limitations matter for interpretation:

1. **End-to-end latency is not scout-only latency.** The measurement includes incident capture, any event-processing delay, odds-engine scheduling, and trading-feed delivery.
2. **Incident types are sport-specific.** The TT stream is dominated by point and score events, while Fifa contains a different event vocabulary. Raw sport medians should therefore be supplemented by event-type tables rather than interpreted as a pure treatment effect.
3. **Recovery and repeated states require deduplication.** The analysis removes duplicate incident keys and full trading-state keys before response measurement.
4. **Incident-index gaps are not automatically lost points.** The incident stream contains more than one type of event, so a positive index jump can reflect omitted event categories rather than missing observations. The notebook reports these gaps instead of silently treating them as losses.
5. **Settlement calibration uses a selected terminal state.** The final active probability is the most natural forecast for settlement, but it is not an independent pre-event forecast and may be affected by late feed behavior.
6. **The Fifa comparison is operational, not causal.** The data strongly show different pipelines, but they do not identify which component of the pipeline creates the difference.

---

## Final answer

**How fast does the bookmaker reprice after an incident?**

For machine-fed Fifa, approximately 0.25 seconds at the per-match median. For human-keyed table tennis, approximately 76.4 seconds at the per-match median, with a much wider tail and a 90th-percentile event latency above 200 seconds.

**How far does it move?**

The typical TT probability movement is about 1.31 percentage points, but the 90th percentile is 8.4 points. Close score states have substantially larger upper-tail moves than already-decided states.

**Is the probability calibrated?**

Partly, but not perfectly. The TT sample produces a Brier score of 0.1451 and log loss of 0.4432. The reliability curve departs from the diagonal, especially in middle and higher-probability buckets, although some buckets are small.

**What is the market lesson?**

Option C reveals a measurable internal information-processing bottleneck. A human-keyed event feed can leave the bookmaker's published probability stale for long enough to matter, but the exploitable value depends on whether the odds remain available, whether the event is correctly classified, and whether the observed delay is truly predictable. The natural next experiment would be a market-making backtest using the BETER probability changes themselves, with explicit assumptions about when an order could have been placed and whether the odds were actually available during the delay.

---

## Reproducibility

The complete analysis is in [analysis/option_c_analysis.ipynb](analysis/option_c_analysis.ipynb). The exported event-level and summary data are in [analysis/option_c_results](analysis/option_c_results), including:

- `incident_responses.csv`
- `latency_bootstrap_summary.csv`
- `latency_match_summary.csv`
- `synthetic_calibration.csv`
- `tt_calibration_forecasts.csv`
- `figures/latency_comparison.png`
- `figures/score_state_repricing.png`
- `figures/tt_calibration.png`
