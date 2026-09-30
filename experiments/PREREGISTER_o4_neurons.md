# Pre-registration: O4 — neuron-level frequency purity at 121 and 125

**Commit this file BEFORE computing the numbers it scores.**

**Date:** 2026-09-14 · **Script:** `analyze_o4.py`
**Data:** already on disk — `results/k03_grid_acts` (torch float32, 5 seeds per modulus)
and `results/n7_engine` (engine float64, 3 seeds). **No training. Zero compute budget.**

## Disclosure of what was already computed, before this file was written

The method was validated at **n=113** first, because a statistic that cannot reproduce a
published number is not worth pointing at an open question. Result: **94.9% of 512 neurons
above the 0.85 threshold** in discrete-log coordinates (k03 `B_thesis` n=113 s0), against
**96.9%** reported by 2606.17399 and **84.6%** by Nanda for `a+b`. The same neurons in raw
integer coordinates read **0.0%**, and a cell-shuffled control reads **0.00%**.

**n=121 and n=125 — the runs this pre-registration is about — have not been touched.**
That is the whole point of writing this now: O4 asks about the prime powers, and the
prime-power numbers do not exist yet.

## Why

**C6 is the thesis's headline claim and it rests entirely on embedding spectra.**
2606.17399's equivalent claim carries neuron-level support (their Fig 3 / Table 1). A
reviewer will ask whether our 4-key-frequency structure at 121 and 125 is a property of
the *circuit* or an artifact of looking only at `W_E`, and right now we cannot answer.

## Hypothesis

The discrete-log clock at the prime powers is implemented in the MLP, not just visible in
the embedding: neurons at 121 and 125 are single-frequency objects in multiplicative
coordinates, as they are at the prime 113.

## Prediction

**≥ 60% of the 512 MLP neurons tuned (frac > 0.85) in discrete-log coordinates, in ≥ 4 of
5 seeds, at each of 121 and 125.**

The threshold is set at 60%, well below 113's measured 94.9%, deliberately. A prime power
has a smaller unit group (φ(121)=110, φ(125)=100 against φ(113)=112) and — unlike a prime —
its non-units are a nontrivial subgroup structure the model must also handle, so demanding
parity with the prime would be demanding more than the hypothesis claims. 60% still puts
the result far above both controls, which sit near zero.

## Success criteria — exact, and implemented as code before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| N1 | neurons are single-frequency in **multiplicative** coordinates | tuned ≥ **60%** in ≥4/5 seeds, at 121 **and** 125 | `analyze_o4.py::n1` |
| N2 | it is the **coordinate**, not the neurons | tuned(dlog) > **10×** tuned(raw) per seed | `analyze_o4.py::n2` |
| N3 | it is not the grid size | tuned(dlog) > **10×** tuned(cell-shuffled dlog) per seed | `analyze_o4.py::n3` |
| N4 | the tuned neurons concentrate on **few** frequencies | ≤ **8** distinct best-k cover ≥50% of tuned neurons, at 121 and 125 | `analyze_o4.py::n4` |

N4 is the link back to C6: the embedding says 4–8 key frequencies carry the circuit, so the
neurons should agree. If N1 holds but N4 fails — many tuned neurons spread over many
frequencies — the neurons are periodic but they are not *this* clock, and C6 gains no
neuron-level support from them.

## What would falsify this

- **N1 < 60% at either prime power** while 113 reads 94.9%: the prime-power circuit is not
  neuron-level single-frequency, C6 stays an embedding-only claim, and the paper must say
  so. This is a publishable negative — it is the "circuits differ across algebraic
  conditions" branch of the Gate 2 thesis sentence.
- **N2 or N3 fails**: the statistic is reading grid size or a coordinate artifact, not
  tuning. Then the n=113 number is suspect too and nothing is claimed anywhere.
- **N4 fails with N1 holding**: report as "neurons are periodic but not concentrated" and
  do **not** promote it as support for C6.

## Required controls

| claim shape | control |
|---|---|
| "sparse in basis X" | the **additive/raw** basis on the same neurons (N2) — 2606.17399's own control |
| "structure, not size" | **cell-shuffled** activations per neuron: same grid side, same value distribution, no 2D structure (N3) |
| "specific to generalisation" | **ungrokked runs** — `n125_s1` (engine, test acc 0.6082, Gini_mult 0.218, n_key 0) and any k03 run below 0.99 — scored and reported alongside, **exploratory**, no criterion |

## Known confounds

- **Grid side differs across moduli** (112 / 110 / 100 in dlog; 113 / 121 / 125 raw). The
  chance level of the statistic is eight bins out of s² maximised over s/2 frequencies, so
  it moves with s. N3's shuffle control measures that chance level *at each grid size*,
  which is why the criterion is a ratio to the control and not a bare percentage. This is
  the defence LAB_PROTOCOL.md prescribes after size confounds killed C7b, C19 and C20.
- **Dead neurons.** A ReLU neuron that never fires has ~0 energy and `freq_fraction`
  returns ~0 for it, so dead neurons depress the percentage rather than inflate it. The
  count of neurons with negligible variance is reported alongside, and the criterion is
  scored on **all 512** — the denominator 2606.17399 uses.
- **n=119 has no discrete log** — (Z/119Z)* ≅ Z₆ × Z₁₆ is non-cyclic. It is **skipped
  loudly**, never silently. The product-character generalisation exists (C23) but extending
  this statistic to it is a separate piece of work and is not claimed here.
- **Two precision regimes.** k03 is float32, `results/n7_engine` float64 (C30). Primary
  scoring is on **k03, 5 seeds**; the engine runs are reported in a separate block and
  never pooled with them.

## Analysis plan — decided now

`PYTHONPATH=. .venv/bin/python analyze_o4.py`, scoring N1–N4 on `results/k03_grid_acts`
(`B_thesis`, n ∈ {113, 121, 125}, seeds 0–4), with `results/n7_engine` reported separately.
Statistic: `src.analysis.neurons.freq_fraction`, 8-bin family, DC removed, threshold 0.85 —
the same computation `src/viz/mechinterp.neuron_frequency_map` draws, imported not copied.
Coordinates via `mechinterp._dlog_order` / `_reorder_dlog`.

Recorded per run, **exploratory, no criterion attached**: the best-k histogram of the tuned
neurons and its overlap with the embedding's key frequencies, the dead-neuron count, the
mean max-fraction, and the ungrokked control runs.

## Outcome (filled in AFTER the run — never edit anything above)

**Date scored:** 2026-09-14 · `analyze_o4.py` on `results/k03_grid_acts`, 15 runs ·
full output `logs/o4_k03.log`

- **Result: all four criteria HELD.** The prime-power neurons are single-frequency objects
  in discrete-log coordinates, on exactly the frequencies the embedding already named.

| n | tuned (dlog), 5 seeds | raw | shuffled | mean max-fraction |
|---|---|---|---|---|
| 113 (reference prime) | 94.9 · 91.8 · 100.0 · 99.8 · 98.0 % | 0.0 % | 0.00 % | 0.93–0.95 |
| **121 = 11²** | **74.0 · 79.9 · 89.3 · 73.2 · 78.9 %** | 0.0 % | 0.00 % | 0.82–0.90 |
| **125 = 5³** | **76.0 · 72.9 · 71.7 · 85.7 %** (4 grokked seeds) | 0.0 % | 0.00 % | 0.86–0.91 |

- **Criteria met:**
  - **N1 HELD** — 5/5 at 121 and 4/4 at 125 clear the 60% threshold. n=113 reads 94.9% s0
    against 2606.17399's published **96.9%**, which is the method check.
  - **N2 HELD** — the same neurons in raw integer coordinates are **0.0%** tuned at every
    modulus and every seed. The ratio is division by zero in the favourable direction.
  - **N3 HELD** — cell-shuffled controls read **0.00%** at every grid size (112, 110, 100),
    so the result is not the chance level of a smaller unit group.
  - **N4 HELD** — the tuned neurons occupy only **3–5 distinct frequencies**, and those
    frequencies are **exactly** the embedding's key set: overlap 4/4, 5/5, 3/3 in all 14
    grokked runs. Note the *cover* half of N4 never bound (100% everywhere by construction
    once the distinct count is below 8); what carries the criterion is the distinct count.

- **The negative control arm, added while scoring (exploratory):** runs that actually
  failed read **6.8%** (engine `n125_s1`, acc 0.6082, mean fraction 0.295) and **0.0%**
  (k04 `n49` s0/s1, `n54` s1, acc 0.15–0.27), against 71.7–100% for the grokked runs.
  `n63` is skipped loudly — (Z/63Z)* = Z₆×Z₆, no discrete log.

- **Deviations from this pre-registration, and why:**
  1. **A third run state was added: `near-grok`.** The plan had grokked (>0.99) and
     "ungrokked". k03's `n125_s1` ends at **0.9779** — a model at 97.8% test accuracy that
     reads 74.8% tuned, i.e. like the grokked runs. Scoring it as the negative control
     would have put a working model on the control side and destroyed the contrast. It is
     now excluded from both the criteria and the control arm, and reported on its own.
     The failed runs listed above are the control instead. **This changes no criterion**:
     N1 counts grokked seeds and `n125_s1` was never among them.
  2. **The embedding-overlap column was off by one when first computed.**
     `key_freqs_5x_median(mult_amplitude(...))` indexes a **DC-dropped** array, so its
     index 0 is frequency 1, while `freq_fraction` returns the frequency itself. Fixed by
     planting a pure character of known frequency and reading both conventions off the
     result, never by reasoning about them. Overlap went from a spurious 0/4 to 4/4. The
     column is exploratory and no criterion depended on it.
  3. **n=119 was never scored**, as pre-registered: non-cyclic unit group, skipped loudly.

- **What this changes.** C6 was an embedding-only claim; it now has neuron-level support at
  both prime powers, with a matched-failure control, on data that was already on disk. It
  does **not** say the prime powers match the prime — 72–89% against 92–100% is a real gap,
  reported as measured and not explained here.
