# Decisions

Every choice in the project, why it was made, and what was rejected. Numbers are **validation** results from the files in `results/` (Phase 2: mean over seeds, ± = sample standard deviation over seeds). Test is scored once, in notebook 10, and no decision below uses it.

## Words used here

| Word | Meaning |
| --- | --- |
| RUL | Remaining Useful Life: cycles left before the engine fails |
| Early / late point | True RUL > 125 (far from failure) / true RUL ≤ 125 (close to failure) |
| Window | The last W cycles of one engine × 14 sensors, read by a sequence model |
| Capped training | Targets above 125 are set to 125 while training. Scoring always uses the true RUL. |
| Seed | The random start of training; different seeds give slightly different models |
| Cap rule | Adopt capping only if overall RMSE is no worse **and** late RMSE is better, on validation (decided 2026-10-21, before any result) |

---

## Phase 1: data and evaluation

| Decision | Why | Rejected, and why |
| --- | --- | --- |
| Use subset **FD001** | The project brief starts there: one operating condition, one failure type, so the comparison is clean | FD002–FD004: extra conditions and failure types would hide the main comparison |
| RUL = last cycle − current cycle | Training engines run to failure, so the exact answer is known for every row | — |
| Remove **8 flat columns** (`setting_3`, sensors 1, 5, 6, 10, 16, 18, 19) | Their most common value fills ≥ 95% of rows, so they carry no wear signal. The next highest share is 26%, so the line isn't close. | Keeping all 24: noise only |
| No normalisation "per operating regime" | FD001 has one regime, so per regime = one global scaling | Following the brief word for word would change nothing |
| Split **by engine**: 80 train, 20 validation | Rows of one engine are near-copies; a row split puts the same engine on both sides and the score looks better than it is | Random row split: leaks |
| Score validation on **100 fixed points** (20 engines × 5 cycles, seed 42) | Copies the test set (one point per engine before failure); every model is scored on the same points | All validation rows: long-lived engines would dominate |
| Target = **true RUL**, no cap in the answer key | The answer key stays the real answer | Capping the answer key: makes scores look better without the model getting better |
| Report **early vs late** error at 125 | The brief asks for error by degradation stage; late errors are the costly ones. 125 is the common C-MAPSS convention. | Grouping by the *predicted* RUL: the model would choose its own group |
| **MAE and RMSE** as the main metrics | The brief asks for them. MAE = average miss; RMSE punishes big misses. | — |
| Test labels loaded in **one notebook only** | If test numbers influence any choice, the test score stops being honest | Tuning on test |

## Phase 1: models

| Decision | Why | Rejected, and why |
| --- | --- | --- |
| Rolling mean, std, slope over the **last 10 cycles** | Smooths noise and captures the trend; 10 is shorter than the shortest test engine (31 cycles). Past cycles only. | Centred windows: use the future |
| Linear baseline (OLS, Elastic Net), then Random Forest and gradient boosting | Degradation is "flat, then steep"; trees can bend, a line can't. Best: RF / GB 40.17. | — |
| No built-in `early_stopping` in gradient boosting | Its internal hold-out picks random rows, which leaks the same engine | — |
| Capped training tested but **not adopted** for trees | Failed the cap rule: overall RMSE worse by 1.7–5.3 | — |

---

## Phase 2: which models

| Decision | Why | Rejected, and why |
| --- | --- | --- |
| **LSTM** as the main sequence model | Reads the raw history and keeps a memory (3 gates + a memory cell), which is what the 10-cycle Phase 1 features miss. The brief's pre-read (Hochreiter and Schmidhuber, 1997). | — |
| **1D CNN** as the non-recurrent comparison | Fast, and common on FD001. Filters slide along **time only**; flatten at the end keeps *when* a pattern happened. | 2D "image" CNN: sensor column order means nothing. Global pooling: loses timing. |
| CNN **not** the main model | Loses to the LSTM on both jobs: uncapped 40.14 ± 1.69 vs 36.28 ± 3.14; capped late 15.02 vs 12.72. It has 5–26× more weights, almost all in the flatten → dense layer. | — |
| **No Transformer** | 80 training engines (~16k windows) is small data; Transformers need more data and tuning, and published FD001 gains over a good LSTM are small | — |
| **CNN-LSTM** (arXiv 2412.15998) not adopted | Uncapped: best version 36.59 vs plain LSTM 34.28 (3 seeds). Capped: late 12.34–14.49 vs about 12.5 for the plain LSTM, so it ties at best. It adds a conv layer and weights without a clear gain. | Quoting the paper's numbers: its own results table contradicts itself |
| **RNN** not adopted; **GRU** ties but isn't swapped in | RNN (no gates, early cycles fade): worst everywhere, uncapped 39.40, capped late 18.29. GRU (2 gates): with 5 seeds it ties the LSTM on the uncapped job (35.76 ± 0.53 vs 36.28 ± 3.14, steadier), but the LSTM is better near failure (capped late 12.72 vs 13.55), the job the hybrid depends on most. With 3 seeds the uncapped order was the other way round (LSTM 34.28 vs GRU 35.97), which is why final comparisons use 5 seeds. | Swapping in the GRU: settings were tuned for the LSTM, and the LSTM wins the near-failure job |

## Phase 2: inputs and training

| Decision | Why | Rejected, and why |
| --- | --- | --- |
| Same **14 sensors**; z-score fitted on the 80 training engines only | Fair comparison with Phase 1; fitting on validation or test would leak | Settings 1 and 2: they only wobble around one operating condition |
| Windows stay **inside one engine** and end exactly at the scored cycle | Crossing engines mixes a failing engine with a new one; ending later uses the future | — |
| Short histories padded with the engine's **first row** | The first (healthiest) reading means "no change yet" | Zero padding: after scaling, 0 means "average reading" |
| **Early stopping** on 16 of the 80 training engines | Training stops on engines it doesn't learn from; validation stays clean for choosing settings | Early stopping on validation: validation would be used twice |
| Adam, batch 128, LR halving on plateau, gradient clipping 1.0, patience 15 | Standard, stable recipe in C-MAPSS papers | RMSprop: worse in the tuning runs |
| **5 seeds** for every final model, 3 for experiments, mean ± std (ddof 1) | Results move a lot between seeds (uncapped LSTM ± 3.14). One seed can crown a lucky model; 3 seeds once got the GRU vs LSTM order wrong. | Single-seed results |
| Settings chosen **on validation only**, in stages | Fewer runs than a full grid; test stays untouched | Full grid; tuning on test |

## Phase 2: chosen settings

| Model | Window | Size | Optimizer | Why |
| --- | --- | --- | --- | --- |
| LSTM uncapped | 30 | 2 layers × 64 | Adam, lr 0.001 | Best 3-seed RMSE (34.28). W = 50 / 70 were worse (38.63 / 38.69 at 1 × 64); 32 units worse; 128 × 2 close (35.34) but slower. |
| LSTM capped | 70 | 1 layer × 64 | Adam, lr 0.003 | Longer history helps near failure: late RMSE 15.69 → 13.95 → 12.23 for W = 30 / 50 / 70. |

## Phase 2: capping and the two-stage hybrid

| Decision | Why | Rejected, and why |
| --- | --- | --- |
| A **single capped model is not the final model** | Cap rule: late RMSE much better (12.72 vs 23.78) but overall worse (39.79 vs 36.28) → **FAIL**. It can't say more than ~125 for healthy engines (early RMSE 81.35). | — |
| **Two-stage hybrid** is the final model | Capped LSTM answers; if it predicts **> 115**, the uncapped LSTM answers instead (larger of the two). RMSE 34.78 ± 2.27 (R² 0.67), late 15.15. Better overall than either LSTM alone and far better late than gradient boosting (26.54): it passes the spirit of the cap rule. | — |
| Gate threshold **115** | Gates 100 / 110 give 33.4 overall but miss 9–21 near-failure points per seed and raise late RMSE to 16.4–17.7. Gate 120 barely fires (39.23). At 115: 4.6 missed per seed, all with true RUL 89–121. | Gate 125: the capped model almost never predicts above ~123, so it would never fire |
| Keep the **larger** of the two answers | The gate already judged the engine healthy; a lower number would contradict it | — |
| Gate = the capped model's own prediction | It already carries the "close to failure?" information | A separate early/late classifier: 77% accurate but calls 17 of 78 near-failure points "healthy"; late RMSE 18.36 |

## Phase 2: ML + DL combinations

| Decision | Why | Rejected, and why |
| --- | --- | --- |
| Report **capped LSTM + gradient-boosting fallback** as the ML + DL result | Same gate, GB answers for healthy engines. Ties the hybrid (35.22 vs 34.78, inside the seed spread), slightly better late (14.61 vs 15.15) and steadier (± 1.28 vs ± 2.27), because GB gives the same answer every run. | Making it the final model: it only ties, and the LSTM hybrid was chosen first |
| Weighted averages with gradient boosting not adopted | Best weights put 10–20% on GB and move RMSE by 0.3 or less. A 50/50 blend with the capped LSTM lowers overall RMSE (38.29) but worsens late RMSE (17.93), which defeats the point of the capped model. | — |

## Phase 2: metrics

| Decision | Why | Rejected, and why |
| --- | --- | --- |
| MAE and RMSE stay main; add **R²**, early / late RMSE, **late precision / recall** | R² is easy to read (share of variation explained) but ranks models like RMSE within a set. Precision / recall apply because early / late turns each RUL into a yes/no "close to failure?": recall = dangerous misses, precision = wasted maintenance. | Recall alone: a capped model gets ~1.00 just by almost never predicting above 125 |
| **NASA score** on test only (notebook 10) | The C-MAPSS standard; punishes predicting too much life exponentially. On validation a few engines with RUL up to 329 blow it up to millions. | NASA on validation |
| Adjusted R² for OLS only | It needs fewer inputs than the 100 scored points. An LSTM reads 420–980 inputs, so it's undefined; trees (57 inputs) get near zero. | Adjusted R² for every model |
