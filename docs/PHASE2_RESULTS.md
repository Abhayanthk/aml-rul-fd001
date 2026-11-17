# Phase 2 Results: Sequence Models vs Engineered Features

Phase 2 asks one question: **does a model that reads the raw sensor history beat the Phase 1 models built on hand-made features, and where?**

Phase 1 rules are unchanged: same 80 training / 20 validation engines, same 100 validation points, same test rows, true RUL as the answer key, early (true RUL > 125) vs late (true RUL ≤ 125).

> This document is for discussion. Please read it and leave your opinion, especially on the [questions at the end](#questions-for-the-team).

## Contents

1. [Words used here](#words-used-here)
2. [Metrics](#metrics)
3. [Models we built](#models-we-built)
4. [Final model: two-stage hybrid](#final-model-two-stage-hybrid)
5. [Results: every model, every metric](#results-every-model-every-metric)
6. [What the results say](#what-the-results-say)
7. [Decisions](#decisions)
8. [Known weaknesses](#known-weaknesses)
9. [Where the numbers come from](#where-the-numbers-come-from)
10. [Questions for the team](#questions-for-the-team)

## Words used here

| Word | Meaning |
| --- | --- |
| Window | The last N cycles of one engine that a sequence model reads |
| Uncapped training | The model learns the true RUL (can predict 200 or 300) |
| Capped training | The model learns RUL limited to 125 ("125 or more" all look the same). It is still **scored against the true RUL**. |
| Seed | The random starting point of training. Different seeds give slightly different models, so we train 5 (seeds 0–4) and report the mean. |
| ± | Sample standard deviation of RMSE over the 5 seeds (ddof 1): how much the result moves between runs |
| Early / late point | True RUL > 125 (far from failure) / true RUL ≤ 125 (close to failure) |
| Gate | The rule that decides which model answers in the two-stage hybrid |

---

## Metrics

| Metric | Question it answers | Better |
| --- | --- | --- |
| MAE | Average miss in cycles | Lower |
| RMSE | Like MAE, but big misses count more. **Main metric.** | Lower |
| R² | Share of the variation in true RUL the model explains (1 = perfect, 0 = no better than always guessing the average) | Higher |
| Early / late RMSE | RMSE on far-from-failure and close-to-failure points only | Lower |
| Late precision | Of engines the model says are close to failure (prediction ≤ 125), how many really are? Low = healthy engines sent to maintenance early | Higher |
| Late recall | Of engines really close to failure, how many did the model flag? Low = dying engines called healthy (**dangerous**) | Higher |
| Late F1 | One number that balances precision and recall | Higher |
| NASA score | Sum of penalties that grow exponentially, harder for predicting **too much** life: `e^(d/10) − 1` if d > 0, `e^(−d/13) − 1` if d < 0, d = predicted − true. Standard in C-MAPSS papers. | Lower |

- Precision and recall are classification metrics. They apply here because the early/late split turns every RUL prediction into a yes/no answer: "close to failure?"
- The NASA score is shown on **test only**, as in papers. On validation a few engines with true RUL up to 329 blow it up to millions.
- **Adjusted R²** penalises R² for the number of inputs: `1 − (1 − R²)(n − 1)/(n − p − 1)`, n = 100 points. It only makes sense for OLS (p = 24 inputs): validation **0.338**, test **0.257**. For the trees (p = 57) it is near zero or negative, and for sequence models it is undefined: a window of 30 × 14 = 420 inputs is more than the 100 scored points. So it is not in the main tables.

---

## Models we built

All sequence models read the same 14 sensors, scaled with the training engines' mean and std, and end in the same small head (dense 32 + ReLU → 1 output).

| Model | What it is | Layers × units | Trainable weights |
| --- | --- | --- | --- |
| LSTM | Recurrent network with a separate memory cell and 3 gates (forget, input, output). It can keep or drop information from many cycles ago. | uncapped 2 × 64, capped 1 × 64 | 55,873 / 22,593 |
| GRU | Lighter recurrent network: 2 gates, memory merged into the hidden state | same as LSTM | 42,433 / 17,473 |
| RNN | Plain recurrent network, no gates: memory is overwritten every step | same as LSTM | 15,553 / 7,233 |
| 1D CNN | 3 convolution layers sliding over **time** (32, 64, 64 filters, kernels 5, 5, 3), then flatten → dense 128 → 1 | 3 conv + 2 dense | 270,945 (W 30) / 598,625 (W 70) |
| CNN-LSTM | One convolution (64 filters, kernel 5) + max-pooling, then a 2 × 64 LSTM | 1 conv + 2 × 64 | 73,217 |
| Two-stage hybrid | Capped LSTM + uncapped LSTM joined by a gate (next section) | — | both LSTMs |
| Capped LSTM + GB fallback | Same gate, but healthy engines go to Phase 1 gradient boosting | — | capped LSTM + GB |
| Classifier + late regressor | A separate LSTM classifier decides early/late first; late engines go to the capped LSTM | — | — |
| Blends | Weighted average of gradient boosting and one DL model, weight chosen on validation | — | — |

Training recipe (same for every model): Adam, batch 128, MSE loss on RUL / 100, gradient clipping 1.0, learning rate halves when progress stalls (5 epochs), early stopping after 15 epochs without improvement on 16 held-out training engines, at most 150 epochs.

| Setting | Uncapped models | Capped models |
| --- | --- | --- |
| Window | 30 cycles | 70 cycles |
| Learning rate | 0.001 | 0.003 (CNN: 0.001) |
| Target | true RUL | RUL capped at 125 |

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

**Why this one:** on validation it is the only design that is good at both jobs. It has the lowest overall RMSE (**34.78**, R² 0.67) and close to the lowest late RMSE (**15.15**). It was chosen on validation **before** the test set was scored.

---

## Results: every model, every metric

Phase 1 rows are single runs (they give the same answer every time). Phase 2 rows are the mean of 5 seeds, ± = sample std of RMSE between seeds. Errors are in cycles.

### Validation (100 points: 22 early, 78 late)

| Group | Model | MAE | RMSE | R² | Early RMSE | Late RMSE | Late precision | Late recall | Late F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 1 | OLS | 32.86 | 42.92 | 0.499 | 64.87 | 34.27 | 0.89 | 0.76 | 0.82 |
| Phase 1 | Elastic Net | 31.66 | 41.77 | 0.525 | 66.02 | 31.74 | 0.91 | 0.79 | 0.85 |
| Phase 1 | Random Forest | 25.03 | 40.17 | 0.561 | 69.59 | 26.50 | 0.88 | 0.86 | 0.87 |
| Phase 1 | Gradient boosting | 26.12 | 40.17 | 0.561 | 69.54 | 26.54 | 0.88 | 0.86 | 0.87 |
| Uncapped | LSTM | 23.48 | 36.28 ± 3.14 | 0.640 | 62.96 | 23.78 | 0.93 | 0.86 | 0.89 |
| Uncapped | GRU | 22.82 | 35.76 ± 0.53 | 0.652 | 62.57 | 23.07 | 0.94 | 0.88 | 0.91 |
| Uncapped | RNN | 27.05 | 39.40 ± 2.68 | 0.576 | 62.16 | 29.72 | 0.94 | 0.82 | 0.87 |
| Uncapped | 1D CNN | 25.43 | 40.14 ± 1.69 | 0.561 | 72.75 | 23.78 | 0.90 | 0.93 | 0.91 |
| Uncapped | CNN-LSTM | 24.02 | 37.33 ± 1.11 | 0.620 | 64.52 | 24.62 | 0.93 | 0.87 | 0.89 |
| Capped | LSTM | 21.25 | 39.79 ± 0.59 | 0.569 | 81.35 | 12.72 | 0.79 | 1.00 | 0.88 |
| Capped | GRU | 21.25 | 39.38 ± 0.68 | 0.578 | 79.95 | 13.55 | 0.79 | 0.99 | 0.88 |
| Capped | RNN | 24.43 | 41.96 ± 0.69 | 0.521 | 82.56 | 18.29 | 0.78 | 0.99 | 0.87 |
| Capped | 1D CNN | 23.66 | 42.03 ± 1.13 | 0.519 | 84.99 | 15.02 | 0.80 | 0.99 | 0.88 |
| Capped | CNN-LSTM | 21.32 | 39.76 ± 1.27 | 0.569 | 80.55 | 13.95 | 0.80 | 0.99 | 0.89 |
| Two-stage | **Two-stage hybrid (final)** | 20.32 | 34.78 ± 2.27 | 0.670 | 68.40 | 15.15 | 0.87 | 0.95 | 0.91 |
| Two-stage | Capped LSTM + GB fallback | 19.91 | 35.22 ± 1.28 | 0.662 | 69.84 | 14.61 | 0.86 | 0.96 | 0.90 |
| Two-stage | Classifier + late regressor | 23.22 | 40.33 ± 1.24 | 0.557 | 78.83 | 18.17 | 0.78 | 1.00 | 0.88 |
| Blend | 0.2·GB + 0.8·LSTM uncapped | 23.53 | 35.98 ± 2.43 | 0.646 | 63.43 | 22.85 | 0.91 | 0.87 | 0.89 |
| Blend | 0.1·GB + 0.9·Two-stage hybrid | 20.34 | 34.71 ± 1.95 | 0.671 | 68.03 | 15.39 | 0.87 | 0.95 | 0.91 |
| Blend | 0.5·GB + 0.5·LSTM capped | 22.44 | 38.29 ± 0.37 | 0.601 | 74.32 | 17.93 | 0.88 | 0.91 | 0.89 |

### Test (100 engines: 11 early, 89 late), scored once

| Group | Model | MAE | RMSE | R² | Early RMSE | Late RMSE | Late precision | Late recall | Late F1 | NASA score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 1 | OLS | 24.91 | 31.17 | 0.437 | 23.00 | 32.04 | 0.96 | 0.79 | 0.86 | 11,407 |
| Phase 1 | Elastic Net | 23.61 | 29.04 | 0.512 | 20.60 | 29.92 | 0.96 | 0.78 | 0.86 | 5,620 |
| Phase 1 | Random Forest | 18.95 | 25.88 | 0.612 | 25.80 | 25.89 | 0.92 | 0.80 | 0.86 | 6,048 |
| Phase 1 | Gradient boosting | 18.54 | 25.33 | 0.628 | 22.98 | 25.61 | 0.94 | 0.81 | 0.87 | 4,813 |
| Uncapped | LSTM | 19.70 | 28.52 ± 2.66 | 0.526 | 33.78 | 27.62 | 0.96 | 0.80 | 0.87 | 32,544 |
| Uncapped | GRU | 18.47 | 26.99 ± 2.69 | 0.575 | 21.68 | 27.55 | 0.98 | 0.78 | 0.87 | 16,470 |
| Uncapped | RNN | 23.94 | 32.71 ± 2.62 | 0.377 | 22.88 | 33.67 | 0.98 | 0.70 | 0.82 | 54,422 |
| Uncapped | 1D CNN | 18.13 | 24.21 ± 2.58 | 0.657 | 17.34 | 24.92 | 0.95 | 0.82 | 0.88 | 4,996 |
| Uncapped | CNN-LSTM | 18.86 | 25.94 ± 3.35 | 0.605 | 21.66 | 26.35 | 0.96 | 0.76 | 0.85 | 9,550 |
| Capped | LSTM | 11.13 | 14.76 ± 0.26 | 0.874 | 23.14 | 13.36 | 0.89 | 0.99 | 0.94 | 343 |
| Capped | GRU | 11.25 | 14.93 ± 1.08 | 0.870 | 22.00 | 13.78 | 0.89 | 0.98 | 0.93 | 354 |
| Capped | RNN | 13.33 | 17.99 ± 1.41 | 0.812 | 23.66 | 17.14 | 0.89 | 1.00 | 0.94 | 903 |
| Capped | 1D CNN | 13.13 | 16.83 ± 0.86 | 0.836 | 26.24 | 15.20 | 0.90 | 0.96 | 0.93 | 490 |
| Capped | CNN-LSTM | 11.97 | 15.74 ± 1.01 | 0.856 | 21.70 | 14.77 | 0.90 | 0.97 | 0.93 | 424 |
| Two-stage | **Two-stage hybrid (final)** | 14.70 | 22.19 ± 1.74 | 0.714 | 30.46 | 20.84 | 0.92 | 0.91 | 0.91 | 16,337 |
| Two-stage | Capped LSTM + GB fallback | 14.08 | 20.14 ± 0.76 | 0.765 | 23.48 | 19.68 | 0.92 | 0.89 | 0.91 | 3,030 |
| Two-stage | Classifier + late regressor | 12.71 | 18.19 ± 0.92 | 0.808 | 23.33 | 17.37 | 0.89 | 1.00 | 0.94 | 1,195 |
| Blend | 0.2·GB + 0.8·LSTM uncapped | 18.48 | 26.24 ± 2.13 | 0.599 | 28.20 | 25.87 | 0.96 | 0.80 | 0.87 | 12,735 |
| Blend | 0.1·GB + 0.9·Two-stage hybrid | 14.51 | 21.74 ± 1.53 | 0.725 | 28.74 | 20.64 | 0.92 | 0.90 | 0.91 | 10,011 |
| Blend | 0.5·GB + 0.5·LSTM capped | 13.12 | 17.76 ± 0.18 | 0.817 | 18.38 | 17.68 | 0.92 | 0.84 | 0.88 | 786 |

---

## What the results say

1. **Phase 2 fixes the Phase 1 gap on late points.** Every capped sequence model cuts the error near failure: late RMSE 12.72–18.29 on validation and 13.36–17.14 on test, vs 26.50 and 25.61 for the best Phase 1 model. The capped LSTM roughly halves it.
2. **The final model beats every Phase 1 model** on both sets: RMSE 34.78 vs 40.17 on validation, 22.19 vs 25.33 on test, R² 0.67 vs 0.56 on validation.
3. **Validation and test disagree on the single best model.** On test, the capped LSTM alone is best (14.76, 42% below gradient boosting). The two sets contain different engines:
   - validation has 22 far-from-failure points with true RUL up to **329**;
   - test has only 11, and its largest true RUL is **145**.

   A capped model can't be badly wrong when nothing is far from failure, so this test set favours capping. We did **not** switch models after seeing the test numbers; doing so would make the test score meaningless.
4. **Far-from-failure points stay hard for every model** (validation early RMSE 62–85). A healthy engine looks the same with 200 or 300 cycles left. This is a limit of the data, not of the models.
5. **Memory ladder (RNN → GRU → LSTM):** the RNN is the worst recurrent model everywhere (no gates, it forgets). The GRU ties the LSTM on the uncapped job (35.76 vs 36.28) and is steadier (± 0.53 vs ± 3.14), but the LSTM is the best near-failure expert (late 12.72 vs 13.55 GRU, 18.29 RNN). The LSTM stays in the final model.
6. **CNN vs LSTM:** the 1D CNN loses on validation in both versions (capped late 15.02 vs 12.72) with 5–26× more weights. The CNN-LSTM comes close to the capped LSTM (late 13.95) but doesn't beat it.
7. **Combining ML and DL:** averaging with gradient boosting moves RMSE by under 0.4 (uncapped LSTM 36.28 → 35.98, hybrid 34.78 → 34.71), inside the seed noise. With the capped LSTM a 50/50 blend lowers overall RMSE (39.79 → 38.29) but ruins late RMSE (12.72 → 17.93). Using GB as the **fallback** instead ties the hybrid (35.22 vs 34.78), is steadier (± 1.28 vs ± 2.27), and has a far lower test NASA score (3,030 vs 16,337).
8. **A separate classifier is worse than the gate:** it called 17 of 78 late points "healthy" (3 seeds) and its late RMSE is 18.17 vs 15.15 for the hybrid.
9. **What the extra metrics add:**
   - **R²** ranks models the same as RMSE within a set (both use squared errors) but is easier to read: the hybrid explains 67% of the variation on validation, gradient boosting 56%.
   - **Recall can be gamed:** capped models reach late recall 0.99–1.00 on validation only because they almost never predict above 125. Their precision (0.78–0.80) shows the cost. Always read both.
   - **The NASA score exposes the hybrid's weakness:** 16,337 on test vs 4,813 for gradient boosting and 343 for the capped LSTM (see [Known weaknesses](#known-weaknesses)).
10. **Sanity check:** the capped LSTM's test RMSE (14.76) is close to published FD001 LSTM results (about 15–16), so the pipeline behaves as expected.

---

## Decisions

Every choice below was made on **validation only**. The test set was scored once, at the end.

### Which models

| Decision | Why | Not chosen, and why |
| --- | --- | --- |
| **LSTM** as the main model | Built for sequences; its memory keeps what happened many cycles ago, which is exactly what the Phase 1 features miss. The brief's pre-read (Hochreiter and Schmidhuber, 1997). | — |
| **GRU and RNN** tested (team suggestion) | To show the LSTM's gates are worth it. | Not swapped in: the GRU ties uncapped but is worse near failure (13.55 vs 12.72); the RNN is worst everywhere. |
| **1D CNN** as the second architecture | Fast, strong on FD001 in papers. A second architecture shows the result isn't one lucky model. | Not the main model: loses to the LSTM on validation. |
| CNN slides over **time only** | Neighbouring cycles are related. Neighbouring sensor columns are not; their order is arbitrary. | 2D "image" CNN: would treat column order as meaningful. |
| CNN ends with **flatten** | Keeps *when* a pattern happened. | Global average pooling: threw that away and was the weakest model. |
| **CNN-LSTM** tested (arXiv 2412.15998) | Convolution summarises neighbouring cycles before the LSTM. | Not adopted: close to the capped LSTM, not better, more weights. |
| **No Transformer** | 80 training engines (~16k windows) is small data. Transformers need more data and tuning, and published FD001 gains over a good LSTM are small. | — |

### Inputs and training

| Decision | Why | Not chosen, and why |
| --- | --- | --- |
| Same **14 sensors** as Phase 1, z-score fitted on training engines only | Fair comparison. Fitting the scaler on validation or test leaks information. | — |
| Windows stay **inside one engine** and end exactly at the scored cycle | A window crossing two engines mixes a failing engine with a new one; ending later would use the future. | — |
| Short histories padded with the engine's **first row** | Repeating the first (healthiest) reading means "no change yet". | Zero padding: after scaling, 0 means "average reading", not "no data". |
| **Early stopping** on 16 of the 80 training engines | Training stops when it stops improving on engines it doesn't learn from, and validation stays clean for choosing settings. | Early stopping on validation: validation would be used twice. |
| Learning rate halves when progress stalls; gradient clipping | Standard, stable recipe in C-MAPSS papers. | — |
| **5 seeds**, mean ± sample std | Results move between seeds (uncapped LSTM ± 3.1). One seed can crown a lucky model: with 3 seeds the LSTM looked better than the GRU, with 5 they tie. | Single-seed results. |

### Settings chosen on validation (3 seeds per experiment)

| Model | Window | Size | Optimizer | Why |
| --- | --- | --- | --- | --- |
| LSTM uncapped | 30 | 2 layers × 64 | Adam, lr 0.001 | Best and most stable overall RMSE. Longer windows (50, 70) were worse and less stable; 128 units were worse and slower; 32 units were unstable. |
| LSTM capped | 70 | 1 layer × 64 | Adam, lr 0.003 | Longer history helps near failure (late RMSE 15.69 → 13.95 → 12.23 for windows 30 / 50 / 70). Larger models added no accuracy. |

Also tried and rejected: RMSprop (worse for both), dropout 0.5 (worse, unstable), batch size 64 (no gain), AdamW (no gain).

### Capping and the two-stage design

| Decision | Why | Not chosen, and why |
| --- | --- | --- |
| A **single capped model is not the final model** | Our pre-registered cap rule (from Phase 1): adopt capping only if overall RMSE is no worse **and** late RMSE is better. Capped LSTM: late much better (12.72 vs 23.78) but overall worse (39.79 vs 36.28), so it fails. | — |
| **Two-stage** design | Combines the strengths of the capped model (late points) and the uncapped model (far-from-failure points). Better overall **and** better late. | — |
| Gate = the **capped LSTM's own prediction** | Missed only 3–6 late points per seed on validation, all with true RUL 89–121, close to the 125 line. | A separate early/late classifier: called 17 of 78 late points "healthy", late RMSE 18.17. |
| Gate threshold **115** | Tried 100 / 110 / 115 / 120. Lower gates miss more late engines (21 per seed at 100, 9 at 110); 120 barely fires and overall RMSE climbs to 39.23. 115 balances the two. | 125: the capped model almost never predicts above ~123, so the gate would never fire. |
| Final = **LSTM hybrid**, GB-fallback reported alongside | The LSTM hybrid was chosen first and the GB version only ties it on validation. | — |

---

## Known weaknesses

- **The gate is less reliable on test.** Late engines handed to the uncapped model, per seed: validation 5, 4, 6, 3, 5 (true RUL 89–121); test 10, 8, 9, 14, 13 (true RUL 77–121, most above 100). That's why the hybrid's test late RMSE (20.84) is higher than the capped LSTM's (13.36).
- **The NASA score punishes it hard.** When the uncapped LSTM over-predicts, the penalty grows exponentially: 95.5–99.8% of the hybrid's NASA score comes from over-predictions, and it swings from 2,754 to 45,410 between seeds. In seed 4, one engine with 137 cycles left was predicted 241, which alone adds 33,606. Gradient boosting as the fallback over-predicts far less (3,030, steady at 2,653–3,534).
- **Far-from-failure engines** stay unpredictable for every model (validation early RMSE 62–85).

---

## Where the numbers come from

- `scripts/score_all_models.py` scores every prediction file in `results/` and writes `results/all_models_metrics.csv`, the source of both tables above. Run it from the repo root.
- Each Phase 2 prediction file has 5 seed columns: validation rows follow `data/validation_cutoffs.csv`, test rows follow engine id.
- LSTM, 1D CNN, hybrid and GB-fallback files come from notebooks 07, 08 and 09. GRU, RNN, CNN-LSTM and classifier files come from the Phase 2 exploration runs with the same settings and seeds; the GRU and RNN validation scores match notebook 08's 5-seed ladder exactly. Blends are computed from the GB and DL files with the weights in `results/experiments_blends.csv`.
- Experiments behind each setting: `results/experiments_*.csv`.

---

## Questions for the team

1. **Final model:** keep the two-stage hybrid (chosen on validation), or present the capped LSTM as the main result (best on test, but it failed our own cap rule on validation)?
2. **GB fallback:** it ties the hybrid on validation and has a much lower NASA score. Should we present it as the safer version?
3. **"Honest" answer for healthy engines:** instead of giving a number when the gate fires, should the model say *"more than 115 cycles left"*? It's more honest for far-from-failure engines, but it can't be scored with a single RMSE.
4. **Report framing:** is the validation-vs-test difference (point 3) clear enough to defend in the viva?
