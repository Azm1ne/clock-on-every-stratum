# CLAIM_MAP — the paper's evidence ledger

**Built 2026-09-15, Phase A2** (`MASTER_PLAN.md` §2). One row per claim in `STATE.md` §2,
all **42**. Status is **copied from the ledger**, not re-judged here; where the ledger's
status carries a qualifier, the qualifier is reproduced in §2 below and **must appear in the
sentence that makes the claim**.

**What this file is for.** `write-paper` Step 4 greps a drafted section against it. A number
in the paper that is not traceable through a row here is not a number — it is a recollection.
Three checks run against this file:

1. every number in a drafted section appears in a row;
2. no claim marked **RETRACTED** is asserted anywhere but Appendix B;
3. every claim marked **1 seed** carries `(1 seed)` in its own sentence.

**Section keys** are `MASTER_PLAN.md` §3's B-numbers. Every claim also appears in
**Appendix A** (the full ledger) by construction; the `section` column gives its **body**
home, or the appendix where it lives if it has none.

Rows marked **⟨assigned here⟩** were not named in `MASTER_PLAN.md` §3 and are placed by this
pass — those placements are the ones to check at A3.

---

## 1 · The 42 claims

| # | claim, one line | status — from the ledger | section | artifact | script / figure |
|---|---|---|---|---|---|
| C1 | Plan §1.3 algebra table is correct | **VERIFIED, self-checking** | B1 | — (self-checking) | `src/tasks/algebra.py::_selfcheck` |
| C2 | Non-regular 𝒥-class counts 0,0,1,2,8 for 113,119,121,125,120 | **VERIFIED** | B1 | — | `src/tasks/algebra.py::j_structure`; reproduces `2607.07066` Thm D.17 |
| C3 | All six moduli grok on `a·b mod n` | **VERIFIED, 18 moduli, PyTorch autograd** | B3 | `results/k02_grid/`, `results/k04_extended/` | `analyze_k02.py`, `analyze_k04.py` · **F8** (gap) |
| C4 | n=113 replicates `2606.17399` — spectral statistics yes, key-frequency **count** no | **VERIFIED, 5 SEEDS (k07) — except its key-frequency-count clause, which fails** | B7 | `results/k07_arma_seeds/` | `analyze_k07.py` · `figures/k07_arma_seeds/` |
| C5 | Our analysis pipeline is correct | **VERIFIED against third-party ground truth** | B7 | `reference/interpreting-monoids/experiments/P165_d128_h4_mlp512_s1.pt` | `refcheck.py` |
| C6 | The clock survives prime powers | **VERIFIED — the clock survives at FIVE prime powers: 7², 11², 5³, 3⁴, 13²** | B3 | `results/k02_grid/`, `k04_extended/`, `k05_lowdata/` | `analyze_k02.py`, `analyze_k04.py`, `analyze_k05.py` · `figures/basis_comparison.pdf` |
| C7 | Non-regular 𝒥-classes still carry character structure | **VERIFIED, 1 seed** | B5 ⟨assigned here⟩ | `results/k03_grid_acts/` | `jclass_spectra.py` · `figures/jclass_blocks_n*.png` |
| C8 | Key additive freqs = multiples of `n/q`, q ‖ n maximal prime powers | **VERIFIED, 12 moduli, 3–5 seeds each — the best-supported claim in the project** | B6 (result) + B2 (protocol) | `results/k02_grid/`, `k04_extended/`, their n=165 checkpoint | `test_crt_law.py` |
| C9 | Gini_add tracks zero-divisor density | **VERIFIED, 18 moduli** | B6 | `results/k04_extended/` | `analyze_k04.py` |
| C9b | Engine has no memory leak | **VERIFIED** | App E ⟨assigned here⟩ | — | `test_memory.py`, `test_nograd_leak.py` |
| C10 | From-scratch engine is correct | **VERIFIED to machine precision** | B8 | — | `test_autograd.py`, `test_model.py` |
| C11 | From-scratch engine groks | **VERIFIED** | B8 ⟨assigned here⟩ | — | `test_grok.py` ⚠️ *had no assert until 2026-09-15; see §4* |
| C12 | H1.4 Softmax Collapse occurs on our engine | **VERIFIED** | B7 | ⚠️ **none saved** — `FINDINGS.md` §8.1 printed table | ⚠️ **no script in repo**; see §4 |
| C13 | StableMax mitigates C12 | **VERIFIED (mechanism). Neither grokked at n=17/40% — that was C14, not a StableMax failure** | B7 ⟨assigned here⟩ | — | `refcheck.py` (`src/autograd/nn.py::stablemax_cross_entropy`) |
| C14 | Critical dataset size is real here | **VERIFIED, 3 seeds × 3 moduli** | B3 + B7 | `results/k05_lowdata/` | `analyze_k05.py` |
| C15 | **GATE 1**: from-scratch engine reproduces Nanda `2301.05217` | **PASS, all 8 criteria** — with the honesty note that C6/C7 of that gate were edited after failing | B7 | `results/gate1/add113_final.npz` | `run_gate1.py` → `test_gate1.py` · `figures/training_curves.png` |
| C16 | Engine is 4.8× faster after two fixes | **VERIFIED** | App E ⟨assigned here⟩ | — | `test_model.py` verifies the **numerics** (2.68e-15); the **timing** is unstamped — see §4 |
| C17 | A frequency can be in `W_E` yet vestigial in the circuit | **VERIFIED, 1 seed** | B4 ⟨assigned here⟩ | `results/k03_grid_acts/` | `src/viz/mechinterp.py::logit_top_components` |
| C18 | Gate 1 circuit is a clean Clock: logit spectrum 100% `a+b` on {33,45,5,1} | **VERIFIED, 1 seed** | B7 ⟨assigned here⟩ | `results/gate1/add113_final.npz` | `test_gate1.py`, `src/viz/mechinterp.py` · `figures/logit_spectrum_n113.png` |
| ~~C19~~ | ~~φ(n)-normalised grokking time separates cyclic from non-cyclic~~ | **RETRACTED (k04 P1).** Third claim lost to a size confound | **App B** | `results/k04_extended/` | `analyze_k04.py` |
| ~~C20~~ | ~~Activation variance is graded by 𝒥-class depth~~ | **RETRACTED (k03 H1).** Confounded by block cell count | **App B** | `results/k03_grid_acts/` | `analyze_k03.py`, `src/analysis/jblocks.py` |
| C21 | The Discrete-Log Clock circuit uses exactly the embedding's key frequencies | **VERIFIED, 5 SEEDS (k07 S3)** | B3 ⟨assigned here⟩ | `results/k07_arma_seeds/` | `analyze_k07.py` · `figures/logit_spectrum_n113.png` |
| C22 | The clock frequencies at prime powers are **causally** responsible | **VERIFIED, 5 SEEDS** — permutation p < 0.01 in 5/5 everywhere | B4 | `results/k03_grid_acts/` (`logits_all`) | `analyze_n4.py` · **F4** (gap) |
| C23 | The causal test works where there is **no discrete log** — product-character circuit is causal | **VERIFIED, 5 seeds (6 moduli) + 3 seeds (12 moduli) + N7 paper data** | B4 | `results/k03_grid_acts/`, `k04_extended/`, `n7_engine/` | `analyze_n4.py` · **F4** (gap) |
| C24 | 40,000 steps is **not** a converged measurement point at four moduli | **VERIFIED, 5 seeds — and now SUPERSEDED in part by k06** (PR retired at composite moduli) | B2 ⟨assigned here⟩ | `results/k03_grid_acts/`, `k06_horizon/` | `analyze_k03.py`, `analyze_k06.py` |
| C25 | The multiplicative readout is exactly invariant to the choice of primitive root | **VERIFIED** | B2 | — | `test_generator_equivariance.py` |
| C26 | Gini read at a single checkpoint is a noisy estimator, sd ≈ 0.02–0.044 | **VERIFIED, 3 moduli × 3 seeds** | B2 | `results/k03_grid_acts/` | `analyze_k03.py` |
| ~~C27~~ | ~~A network can de-grok~~ | **NOT SUPPORTED / RETRACTED.** A censoring artifact | **App B** | `results/k06_horizon/` ⚠️ `WE_B_thesis_n119_s0.npz` holds a **transient** | `analyze_excursions.py` |
| C28 | The early CRT signal is CRT-specific — but only the **magnitude** discriminates | **VERIFIED for the specificity half, 3 seeds, n=119 only** (engine arm) | B6 | `results/n7_engine/`, `logs/c28_null.log` | `test_crt_null.py` · prereg `PREREGISTER_c28_crt_null.md` (`832635c`) |
| C29 | All four moduli grok on the **from-scratch engine** — the first paper data | **VERIFIED, 3 seeds, PAPER DATA (from-scratch engine, float64)** | B8 | `results/n7_engine/` | `analyze_n7.py` · prereg `PREREGISTER_n7_engine.md` |
| ~~C30~~ | ~~Training precision changes measured sparsity~~ (as a **main effect**) | **NOT SUPPORTED / RETRACTED.** Superseded by C34 as an **interaction**; never revive C30's sentence | **App B** | `results/n9_f32/` | `analyze_n9.py` · prereg `PREREGISTER_n9_precision.md` |
| C31 | The clock is in the **neurons** at the prime powers, not just the embedding | **VERIFIED, 5 seeds (torch float32)** | B3 | `results/k03_grid_acts/` (`mlp_acts`) | `analyze_o4.py` · prereg `PREREGISTER_o4_neurons.md` · `figures/neuron_freqs_n*_dlog.png` |
| C32 | C22/C23 causality survives all three of k08's initialisation conditions | **VERIFIED, 8 seeds × 3 conditions** | B4 | `results/k08_init/` | `analyze_n4.py results/k08_init` · prereg `PREREGISTER_k08_init.md` |
| C33 | **GATE 2**: the local mechanism is invariant; the **stratification** is not | **VERIFIED, 5 seeds (4 at n=125), k03 torch float32; replicated on the engine at 3 seeds** | **B5 (headline)** | `results/gate2/*.json` | `analyze_gate2.py`, `scripts/gate2_discriminate.py`, `scripts/gate2_table.py` · `figures/gate2_strata.pdf` · prereg `PREREGISTER_gate2_strata.md` (`bca6236`) |
| C34 | **O18 solved**: training precision changes measured sparsity — **in PyTorch**. An interaction | **VERIFIED, 3 seeds, torch float64** | B8 | `results/o18_f64/`, `results/o18_cpu/`, `results/o18_transplant/` | `run_o18_cpu.py` · prereg `PREREGISTER_o18_f64.md` (`3d19067`) · **F7** (gap) |
| C35 | n = 113 is not special: the clock holds at **three** primes, embedding and neurons | **VERIFIED, 3 seeds × 2 primes, torch float32** | B3 | `results/k09_primes_zdd/` | `analyze_k09.py` · prereg `PREREGISTER_k09_primes_zdd.md` (`ed9a1f2`) |
| C36 | The clock is **causal at 2⁷** — the prime-power set reaches p = 2, depth 7 | **VERIFIED, 2 seeds at n=128** (it grokked 2/3), torch float32 | B4 | `results/k09_primes_zdd/` | `analyze_n4.py` · **F4** (gap) |
| C37 | ω(n) separates additive sparsity at **matched** zero-divisor density | **CONFIRMATORY, NOT A DISCOVERY — 23 moduli, torch float32** | B6 | `results/k09_primes_zdd/` + all 23 moduli | `analyze_omega.py` · `figures/omega_bands.pdf` · prereg `PREREGISTER_omega.md` (`60ee31b`) |
| C38 | It is the **same** clock across strata, not one clock per stratum | **VERIFIED, 5 seeds × 3 moduli, torch float32 — WITH ONE STATED HOLE** | B5 | `results/gate2/r2_sameness_k03_grid_acts.json` | `analyze_r2_sameness.py` · prereg `PREREGISTER_r2_cross_stratum.md` (`373a085`) |
| C39 | The key characters are what the network **computes with**, not just the basis its output is sparse in | **VERIFIED, 3 seeds, n=113 ONLY, engine float64** | B4 | `results/i1_internal/I1_engine_n113_s{0,1,2}.npz` | `run_intervention.py` · prereg `PREREGISTER_i1_internal_intervention.md` (`07ddcd0`) · **F10** (`src/viz/plots.py::internal_intervention`) |
| C40 | The internal character claim holds at a **non-square-free** modulus, non-unit rows intact | **VERIFIED, 3 seeds, n=121 ONLY, engine float64** | B4 | `results/i1_internal/I1_engine_n121_s{0,1,2}.npz` | `run_intervention.py` · prereg `PREREGISTER_i1b_composite_intervention.md` (`712979d`) · Table `tab:internal` |
| ~~C7b~~ | ~~Regular classes carry more structure than non-regular~~ | **NOT SUPPORTED at 1 seed** | **App B** | `results/k03_grid_acts/` | `jclass_spectra.py`, `src/analysis/jblocks.py` |

**Count check:** 42 rows. VERIFIED 33 · VERIFIED-1-seed 3 (C7, C17, C18) · RETRACTED 5
(C19, C20, C27, C30, C7b) · confirmatory-not-a-discovery 1 (C37). Matches the four category
counts `scripts/session_brief.py` §5 prints; its TOTAL line reads 43 because it also counts a
two-cell summary row in `STATE.md` (session 17's C38 line) that sits outside the ledger table.
*C38, C39 and C40 were missing here until 2026-09-28 (revision WP0, P3).*

### Not a claim, retained for completeness

| row | what it is | status | where it goes |
|---|---|---|---|
| ~~O1 prediction~~ | a **scored prediction**, not a ledger claim: "IPR_mult falls below 6 at 40k epochs" | **HELD after correction (Entry 16)** — the original FAIL was our amplitude-vs-energy bug | App C (pre-registrations), as a worked example of a convention error |

---

## 2 · Qualifiers that must appear in the sentence

Grep a drafted section for the claim, then for its qualifier. A claim without its qualifier
is a false sentence, not an abbreviated one.

| claim | the qualifier, in the sentence that makes the claim |
|---|---|
| C7, C17, C18 | `(1 seed)` |
| C4 | the **key-frequency-count clause is withdrawn** — 4.6 ± 0.8, matching their 4 in only 3/5 seeds. And the agreement with their 0.579 holds **because both arms are float32** |
| C6 | **five** prime powers (7², 11², 5³, 3⁴, 13²) — **not six**; n=128 has no discrete log and is C36's |
| C22, C23, C32, C36 | **permutation p**, Phipson–Smyth (1+b)/(B+1) at **B = 10,000** (R1, 2026-09-21; this row said "200 draws" until 2026-09-28), **median** control — never a ratio to a control mean |
| C24 | **PR is retired at composite moduli**, not re-measured; the Gini clause was corrected from the wrong convention |
| C26 | Gini is averaged over the **final ~10k steps**; a single checkpoint is a ~2σ test |
| C28 | report the **enrichment, never the p** — failed runs reject too. And the step-1,000 magnitude is **below** what a memorising model reaches at convergence |
| C31 | the prime powers **do not match the prime** (72–89% vs 92–100%), a gap reported and unexplained |
| C33 | the §4 falsifier **fired**: G1, the criterion named PRIMARY, does not separate. The verdict rests on **G0 ∧ G2 ∧ G5**. Never "all six criteria passed" |
| C34 | **in PyTorch** — never C30's sentence. And: all of k02–k09 is **float32** |
| C36 | n=123 reproduces the "necessary but not sufficient" pattern; n=91's 0.976 **near-grok** is not counted as support |
| C37 | **the bands were seen before the pre-registration**; **this data cannot rank ω and zdd**: under midranks ρ(ω,G\|zdd) = +0.804 and ρ(zdd,G\|ω) = +0.578 (2026-09-23 audit); the ordinal-rank pair +0.735 > +0.627 that this row carried until 2026-09-28 was a tie artifact and is never quoted in either direction; pairs are **not independent** — the fraction is the statistic, the p is descriptive |
| C38 | **no working failed-run control**: the only two scorable failed-run pairs come from one run at acc 0.7502 and its own null reads p = 1.0; **transpose pairs are not support**, which leaves 0 primary pairs at n=121 and n=125 |
| C39, C40 | **n=113 only** (C39) and **n=121 only** (C40): two moduli, never pooled; engine **float64**; the null's upper tail reaches the effect (report the control max, not only the median); the logit-tensor scoping stands at the other **21** of 23 moduli; C40's below-chance excluded accuracy is **descriptive only** |
| every absolute sparsity number | its **precision regime**, and once per table, that between-modulus and within-arm comparisons are unaffected |
| every character index | its **coordinate system** — discrete-log or exponent-tuple |

---

## 3 · Findings that are not ledger claims but are routed to sections

`MASTER_PLAN.md` §4 sends these to the body; they have no C-number and must not acquire one
in the draft.

| | finding | section | artifact | script |
|---|---|---|---|---|
| **O20** | **The early CRT ramp is NOT the grokking circuit.** Three FAILED seeds at n=63 show the same monotone ramp (ρ = 1.000/0.988/0.995, p < 0.01 from step 1,000) and a *higher* final enrichment (1.459–1.655) than grokked runs show early (1.22–1.45) | **B6** + B9 | `results/o20_n63/`, `logs/o20_score.log` | `test_crt_null.py` · prereg `PREREGISTER_o20_failed_ramp.md` (`6042c56`) |
| **O21** | **Why the engine is dtype-invariant when torch is not — OPEN, no surviving candidate.** Three eliminated: intermediate dtype (0/115 promoted, positive control 100%), matmul accumulation order, softmax formulation (ratios 1.000–1.172, inside the pre-registered band) | **B8** + B9 | — (under 2 min, no training) | `run_o21_probe.py`, `run_o21_numerics.py` · prereg `PREREGISTER_o21_dtype.md` (`b7108d4`) |
| **O10** | No algebraic property predicts grokking time | **B9**, stated plainly as a negative | `results/k04_extended/` | `analyze_k04.py` |
| **O19** | Does the engine spike at all? **0/453, pre-registered as underpowered** (expected 0.68, P(0) ≈ 0.51) | **B9**, stated as underpowered | `results/n7_engine/` | `analyze_excursions.py` |
| **O16 / O8 / O11** | Unbalanced init gives a key set necessary but not sufficient; condition-specific | B8 or B9 | `results/k08_init/` | `analyze_n4.py` |

---

## 4 · Gaps this pass found — each is a defect, not a chore

**Three claims have weaker provenance than the ledger implies.** None is wrong; each is
under-evidenced against the reproducibility gate, and the paper must either fix or scope it.

| claim | the gap | cheapest fix |
|---|---|---|
| **C12** (Softmax Collapse) | **No saved artifact and no script in the repo.** The evidence is a printed table in `FINDINGS.md` §8.1 from an early ad-hoc run at n=17. `scripts/reproduce.sh` covers C13 (via `refcheck.py`) but **not C12** | a ~2-minute n=17 rerun that saves a stamped `.npz` and asserts the two numbers. Zero quota |
| **C16** (4.8× faster) | `test_model.py` verifies the **numerics** (2.68e-15); the **timing** — 2149 → 449 ms/step — is unstamped and machine-dependent | scope it in the draft as a measured engineering note with the machine named, or drop it to App E without the figure |
| **C11** (engine groks) | Its acceptance test `test_grok.py` **had no assert and ran a configuration guaranteed to fail** (`train_frac` 0.4 at n=17), from `init` until 2026-09-15. Now fixed and asserting at 0.8 | none — already fixed. But the paper must **not** cite `test_grok.py` as long-standing evidence; the assert is one day old |

**Figure gaps carried from `MASTER_PLAN.md` §2 A5** — three claims currently have no figure:

| figure | carries | status |
|---|---|---|
| **F4** — baseline/restricted/excluded per modulus + the 200-draw permutation null | **C22, C23, C36** — the paper's central evidence | **no figure exists.** `src/viz/plots.py::ablation_bars` is the nearest starting point |
| **F7** — the `{engine,torch} × {f32,f64}` 2×2 | **C34** | no figure |
| **F8** — grok curves classified by the **median of the final ~10 samples** | **C3** | no figure; `src/viz/plots.py::training_curves` exists but does not classify |

---

## 5 · Where each section's claims come from

| section | claims |
|---|---|
| **B1** Setup | C1, C2 |
| **B2** Methods | C25, C26, C24, C8 (protocol half) |
| **B3** Results I — grokking and the clock | C3, C6, C14, C21, C31, C35 |
| **B4** Results II — the clock is causal | C17, C22, C23, C32, C36, **C39, C40** |
| **B5** Results III — every stratum, its own local group | **C33**, C7, **C38** |
| **B6** Results IV — additive basis and CRT | C8, C9, C28, C37, **O20** |
| **B7** Calibration | C4, C5, C12, C13, C14, C15, C18 |
| **B8** Two implementations | C10, C11, C29, C34, **O21** |
| **B9** Limitations | O10, O19, O16/O8/O11, and every qualifier in §2 |
| **B13** Appendices | **A** all 42 · **B** C19, C20, C27, C30, C7b · **C** all 20 pre-registrations + the O1 prediction · **D** full tables · **E** C9b, C16 |

**Every one of the 42 has a body section or a named appendix. Nothing is `omitted`.**
