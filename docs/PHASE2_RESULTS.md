# Phase 2 Results: Sequence Models vs Engineered Features

Phase 2 asks one question: **does a model that reads the raw sensor history beat the Phase 1 models built on hand-made features, and where?**

Phase 1 rules are unchanged: same 80 training / 20 validation engines, same 100 validation points, same test rows, true RUL as the answer key, MAE and RMSE, early (true RUL > 125) vs late (true RUL ≤ 125).

> This document is for discussion. Please read it and leave your opinion, especially on the [questions at the end](#questions-for-the-team).

## Words used here

| Word | Meaning |
| --- | --- |
| Window | The last N cycles of one engine that a sequence model reads |
| Capped training | The model is trained with RUL limited to 125 ("125 or more" all look the same). It is still scored against the true RUL. |
| Seed | The random starting point of training. Different seeds give slightly different models, so we average 5. |
| Early / late point | True RUL > 125 (far from failure) / true RUL ≤ 125 (close to failure) |

---

## Final model: two-stage hybrid

```
last 70 cycles ──> capped LSTM ──> prediction p
                                   │
             p ≤ 115 ──────────────┴────────────── p > 115
             "close to failure"                    "far from failure"
             answer = p                            q = uncapped LSTM (reads the last 30 cycles)
                                                   answer = the larger of q and p
```

- The **capped LSTM** is the expert on engines near failure. It is the most accurate model on late points.
- It was trained never to say much above 125, so when it predicts **more than 115** the engine is probably healthy and the answer is handed to the **uncapped LSTM**, which can predict 200 or 300. We keep the larger of the two predictions: the gate has already judged the engine healthy, so a lower number from the uncapped model would contradict it.

**Why this one:** on validation it is the only model that is good at both jobs. It has the lowest overall RMSE (35.34) and close to the lowest late RMSE (14.76). It was chosen on validation **before** the test set was scored.

---

## Results

Phase 1 rows are single runs. Phase 2 rows are the mean of 5 seeds (± = spread between seeds). All errors are in cycles; lower is better.

### Validation (100 points: 22 early, 78 late)

| Phase | Model | MAE | RMSE | Early RMSE | Late RMSE |
| --- | --- | --- | --- | --- | --- |
| 1 | OLS | 32.86 | 42.92 | 64.87 | 34.27 |
| 1 | Elastic Net | 31.66 | 41.77 | 66.02 | 31.74 |
| 1 | Random Forest | 25.03 | 40.17 | 69.59 | 26.50 |
| 1 | Gradient boosting | 26.12 | 40.17 | 69.54 | 26.54 |
| 2 | LSTM uncapped | 23.27 | 36.47 ± 3.80 | **64.16** | 23.22 |
| 2 | LSTM capped | 20.84 | 39.62 ± 0.29 | 81.29 | **12.16** |
| 2 | 1D CNN uncapped | 25.67 | 41.12 ± 2.29 | 74.29 | 24.54 |
| 2 | 1D CNN capped | 23.22 | 41.76 ± 0.96 | 84.74 | 14.46 |
| 2 | **Two-stage hybrid (final)** | **19.94** | **35.34 ± 3.76** | 69.99 | 14.76 |

### Test (100 engines: 11 early, 89 late), scored once

| Phase | Model | MAE | RMSE | Early RMSE | Late RMSE |
| --- | --- | --- | --- | --- | --- |
| 1 | OLS | 24.91 | 31.17 | 23.00 | 32.04 |
| 1 | Elastic Net | 23.61 | 29.04 | 20.60 | 29.92 |
| 1 | Random Forest | 18.95 | 25.88 | 25.80 | 25.89 |
| 1 | Gradient boosting | 18.54 | 25.33 | 22.98 | 25.61 |
| 2 | LSTM uncapped | 19.55 | 28.16 ± 2.34 | 31.88 | 27.49 |
| 2 | LSTM capped | **11.65** | **15.55 ± 0.86** | 24.22 | **14.09** |
| 2 | 1D CNN uncapped | 18.79 | 24.96 ± 3.17 | **17.91** | 25.69 |
| 2 | 1D CNN capped | 12.88 | 16.66 ± 0.81 | 26.40 | 14.97 |
| 2 | **Two-stage hybrid (final)** | 15.48 | 23.11 ± 1.33 | 30.14 | 21.91 |

### What the results say

1. **Phase 2 fixes the Phase 1 gap on late points.** Every capped sequence model roughly halves the error near failure: 12–15 vs 26 for gradient boosting, on both validation and test.
2. **The final model beats every Phase 1 model** on both sets: 35.34 vs 40.17 on validation, 23.11 vs 25.33 on test.
3. **Validation and test disagree on the single best model.** On test, the capped LSTM alone is best (15.55, 39% below gradient boosting). The two sets contain different engines:
   - validation has 22 far-from-failure points with true RUL up to **329**;
   - test has only 11, and its largest true RUL is **145**.

   A capped model can't be badly wrong when nothing is far from failure, so this test set favours capping. We did **not** switch models after seeing the test numbers; doing so would make the test score meaningless.
4. **Far-from-failure points stay hard for every model** (validation early RMSE 64–85). A healthy engine looks the same with 200 or 300 cycles left. This is a limit of the data, not of the models.
5. **LSTM vs CNN:** the LSTM wins clearly on validation (late 12.16 vs 14.46). On test the two capped models are tied (14.09 vs 14.97, a gap smaller than the seed spread).
6. **Sanity check:** the capped LSTM's test RMSE (15.55) sits in the range published for FD001 LSTMs (about 15–16), so the pipeline behaves as expected.

---

## Decisions

Every choice below was made on **validation only**. The test set was scored once, at the end.

### Which models

| Decision | Why | Not chosen, and why |
| --- | --- | --- |
| **LSTM** as the main model | Built for sequences; it can remember what happened many cycles ago, which is exactly what the Phase 1 features miss. It's the brief's pre-read (Hochreiter and Schmidhuber, 1997). | — |
| **1D CNN** as the second model | Fast, and strong on FD001 in papers. A second architecture shows the result isn't one lucky model. | — |
| CNN slides over **time only** | Neighbouring cycles are related. Neighbouring sensor columns are not; their order is arbitrary. | 2D "image" CNN: would treat column order as meaningful. |
| CNN ends with **flatten** | Keeps *when* a pattern happened. A first version with global average pooling threw that away and was the weakest model. | Global pooling. |
| **No Transformer** | 80 training engines (~16k windows) is small data. Transformers need more data and tuning, and published FD001 gains over a good LSTM are small. | — |

### Inputs and training

| Decision | Why | Not chosen, and why |
| --- | --- | --- |
| Same **14 sensors** as Phase 1, z-score fitted on training engines only | Fair comparison. Fitting the scaler on validation or test leaks information. | — |
| Windows stay **inside one engine** and end exactly at the scored cycle | A window crossing two engines mixes a failing engine with a new one; ending later would use the future. | — |
| Short histories padded with the engine's **first row** | Repeating the first (healthiest) reading means "no change yet". | Zero padding: after scaling, 0 means "average reading", not "no data". |
| **Early stopping** on 16 of the 80 training engines | Training stops when it stops improving on engines it doesn't learn from, and validation stays clean for choosing settings. | Early stopping on validation: validation would be used twice. |
| Learning rate halves when progress stalls; gradient clipping | Standard, stable recipe in C-MAPSS papers. | — |
| **5 seeds**, mean ± spread | Results move noticeably between seeds (uncapped LSTM: ± 3.8). One seed can crown a lucky model. | Single-seed results. |

### Settings chosen on validation

| Model | Window | Size | Optimizer | Why |
| --- | --- | --- | --- | --- |
| LSTM uncapped | 30 | 2 layers × 64 | Adam, lr 0.001 | Best and most stable overall RMSE. Longer windows (50, 70) were worse and less stable; 128 units were worse and slower; 32 units were unstable. |
| LSTM capped | 70 | 1 layer × 64 | Adam, lr 0.003 | Longer history helps near failure (late RMSE 15.7 → 13.2 → 12.9 for windows 30 / 50 / 70). Larger models added no accuracy. |

Also tried and rejected on validation: RMSprop (worse for both), dropout 0.5 (worse, unstable), batch size 64 (no gain), AdamW (no gain).

### Capping and the two-stage design

| Decision | Why | Not chosen, and why |
| --- | --- | --- |
| A **single capped model is not the final model** | Our pre-registered cap rule (from Phase 1): adopt capping only if overall RMSE is no worse **and** late RMSE is better. Capped LSTM: late much better (12.16 vs 23.22) but overall worse (39.62 vs 36.47), so it fails. | — |
| **Two-stage** design | Combines the strengths of the capped model (late points) and the uncapped model (far-from-failure points). It passes the cap rule: better overall and better late. | — |
| Gate = the **capped LSTM's own prediction** | Missed only 3–11 late points per seed on validation, almost all with true RUL 99–121, close to the 125 line. | A separate early/late classifier: 77% accurate, but it called 17 of 78 late points "healthy" and raised late RMSE to 18.4. |
| Gate threshold **115** | Tried 100 / 110 / 115 / 120. 115 balanced few missed late points against how often the hand-off triggers. | 125: the capped model almost never predicts above ~123, so the gate would never fire. |

### Known weakness

The gate is less reliable on test. Per seed, 10–18 late test engines were handed to the uncapped model, true RUL 77–121, most of them above 100. That's why the hybrid's test late RMSE (21.91) is higher than the capped LSTM's (14.09).

---

## Questions for the team

1. **Final model:** keep the two-stage hybrid (chosen on validation), or present the capped LSTM as the main result (best on test, but it failed our own cap rule on validation)?
2. **"Honest" answer for healthy engines:** instead of giving a number when the gate fires, should the model say *"more than 115 cycles left"*? It's more honest for far-from-failure engines, but it can't be scored with a single RMSE.
3. **Report framing:** is the validation-vs-test difference (point 3 above) clear enough to defend in the viva?
4. **Next step:** who writes which Phase 2 notebook so these numbers are reproduced in the repo?
