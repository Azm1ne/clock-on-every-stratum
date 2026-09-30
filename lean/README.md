# Appendix: the paper's mathematics in Lean 4

Every mathematical statement in the paper is stated here in Lean 4 against mathlib, and is
proved or marked as not yet proved. The table below is the index: one row per statement, from
the paper's LaTeX label to the Lean theorem and the line it is on.

**Status: statements only (session L0).** Every row is `stated`: the theorem compiles with
`sorry` in place of a proof. **No proof is written until the researcher has read each
statement against its LaTeX.** A wrong statement with a proof is worse than no proof, and a
machine can check a proof but not whether it proves the paper's claim.

What this gives, and what it does not. A `proved` row says the mathematics holds for **every**
`n`, not only the `n ≤ 200` covered by the brute-force checks in `src/tasks/algebra.py`. It
does **not** verify that Python code. Those checks stay, because they are what ties the code
the experiments ran to the theorem. Every empirical number in the paper (Gini, losses,
p-values) is a measurement, not a theorem, and is out of scope here. The validity of the
Phipson–Smyth estimator, which the paper cites and does not prove, is also out of scope.

## Build and check

```bash
# once: elan (https://github.com/leanprover/elan), then
cd lean && lake exe cache get      # prebuilt mathlib, pinned by lake-manifest.json
cd .. && scripts/lean_gate.sh      # build, then check every row below
scripts/lean_gate.sh --selfcheck   # the gate must reject a planted sorry and native_decide
```

Toolchain `leanprover/lean4:v4.35.0-rc3`; mathlib `v4.35.0-rc3`
(commit `c55e6e786f49471c72fbddbec5415808896aec1e`).

**`lake build` exiting 0 is not the check**, because `sorry` compiles with a warning. The gate
runs `#print axioms` on every `proved` row and fails unless each lists only `propext`,
`Classical.choice` and `Quot.sound`, so `sorry` and `native_decide` both fail. It also fails if
this table and `Strata/*.lean` name different theorems, or if a row's line number does not
point at its theorem.

## Definitions

`Strata/Basic.lean`: `J n d` (a `J`-class), `Regular`, `nilDepth` (ν), `mOf`/`wOf`/`qOf`
(`eq:mw`), `gini`, `pr`, `ipr2`, `proj` (`eq:project`), `dft`, `ramanujanSum` (not in
mathlib), and `predicted` (`eq:crtset`). `gini` and `pr` mirror
`src/analysis/sparsity.py` and agree with it on 200 random spectra. `predicted` mirrors
`test_crt_law.predicted`.

## Statements

| id | paper | Lean theorem | location | status |
|---|---|---|---|---|
| A0 | eq:units | `Strata.A0_card_units` | `Strata/A.lean:14` | stated |
| A1a | eq:jclass | `Strata.A1_card_J` | `Strata/A.lean:18` | stated |
| A1b | eq:jclass | `Strata.A1_partition` | `Strata/A.lean:22` | stated |
| A2 | prop:torsor, eq:torsor | `Strata.A2_torsor` | `Strata/A.lean:27` | stated |
| A3 | eq:mscale | `Strata.A3_mscale` | `Strata/A.lean:33` | stated |
| A4 | thm:stratum, eq:stratum, eq:torsor, eq:mw | `Strata.A4_stratum` | `Strata/A.lean:39` | stated |
| A5 | eq:jcompose | `Strata.A5_jcompose` | `Strata/A.lean:47` | stated |
| A6 | 01-setup.tex:79–80 | `Strata.A6_nilpotent_iff` | `Strata/A.lean:53` | stated |
| B2s | def:jclass, tab:moduli | `Strata.B2_regular_iff` | `Strata/B.lean:15` | stated |
| B1 | 01-setup.tex:76–77 | `Strata.B1_squarefree_regular` | `Strata/B.lean:20` | stated |
| B2 | tab:moduli | `Strata.B2_not_squarefree` | `Strata/B.lean:24` | stated |
| B3a | eq:descent | `Strata.B3_descent_squarefree` | `Strata/B.lean:28` | stated |
| B3b | eq:descent | `Strata.B3_descent_121` | `Strata/B.lean:33` | stated |
| B3c | eq:descent | `Strata.B3_descent_125` | `Strata/B.lean:37` | stated |
| B3d | 01-setup.tex:98–99, 06-results-clock.tex:148 | `Strata.B3_nilDepth_prime_pow` | `Strata/B.lean:42` | stated |
| B4a | eq:fibration | `Strata.B4_fibration_surjective` | `Strata/B.lean:47` | stated |
| B4b | eq:fibration | `Strata.B4_fibration_factors` | `Strata/B.lean:52` | stated |
| B4c | 01-setup.tex:236–237 | `Strata.B4_cells_ge` | `Strata/B.lean:58` | stated |
| B5 | eq:diagonal | `Strata.B5_diagonal` | `Strata/B.lean:64` | stated |
| B6 | eq:prodchar, eq:diagonal | `Strata.B6_card_dual` | `Strata/B.lean:70` | stated |
| C1 | prop:noop, eq:amplitude | `Strata.C1_noop` | `Strata/C.lean:14` | stated |
| C2a | prop:generator | `Strata.C2_generator_valid` | `Strata/C.lean:23` | stated |
| C2b | prop:generator | `Strata.C2_dlog` | `Strata/C.lean:29` | stated |
| C2c | prop:generator | `Strata.C2_index_perm` | `Strata/C.lean:36` | stated |
| C3a | eq:equivariance | `Strata.C3_gini_perm` | `Strata/C.lean:41` | stated |
| C3b | eq:equivariance | `Strata.C3_pr_perm` | `Strata/C.lean:46` | stated |
| C3c | eq:equivariance, eq:project | `Strata.C3_proj_conj` | `Strata/C.lean:52` | stated |
| C4a | eq:gini | `Strata.C4_gini_sorted` | `Strata/C.lean:60` | stated |
| C4b | eq:gini | `Strata.C4_gini_bounds` | `Strata/C.lean:66` | stated |
| C4c | eq:gini | `Strata.C4_gini_zero_iff` | `Strata/C.lean:71` | stated |
| C4d | eq:gini | `Strata.C4_gini_max_iff` | `Strata/C.lean:77` | stated |
| C5a | eq:pr | `Strata.C5_pr_bounds` | `Strata/C.lean:82` | stated |
| C5b | eq:pr | `Strata.C5_pr_max_iff` | `Strata/C.lean:87` | stated |
| C5c | eq:pr | `Strata.C5_pr_one_iff` | `Strata/C.lean:92` | stated |
| C6 | eq:conventions | `Strata.C6_ipr_pr` | `Strata/C.lean:97` | stated |
| D0 | 09-results-crt.tex:32–33 | `Strata.D0_multiples_support` | `Strata/D.lean:13` | stated |
| D1 | 09-results-crt.tex:115 | `Strata.D1_indicator_ramanujan` | `Strata/D.lean:19` | stated |
| D2a | 09-results-crt.tex:116 | `Strata.D2_closed_form` | `Strata/D.lean:24` | stated |
| D2b | 09-results-crt.tex:117 | `Strata.D2_coprime_small` | `Strata/D.lean:31` | stated |
| D2c | 09-results-crt.tex:117–118 | `Strata.D2_full` | `Strata/D.lean:36` | stated |
| D2d | 09-results-crt.tex:119 | `Strata.D2_squarefree_ne_zero` | `Strata/D.lean:41` | stated |
| D3 | 09-results-crt.tex:120–122 | `Strata.D3_count_165` | `Strata/D.lean:46` | stated |
| D4 | 09-results-crt.tex:35–37 | `Strata.D4_prime_power_vacuous` | `Strata/D.lean:52` | stated |
| E0 | (bridge for E1) | `Strata.E0_regularNat` | `Strata/E.lean:24` | stated |
| E1a | tab:moduli | `Strata.E1_table_arith` | `Strata/E.lean:56` | stated |
| E1b | tab:moduli | `Strata.E1_table_cyclic` | `Strata/E.lean:64` | stated |
| E1c | tab:moduli, 01-setup.tex:84–86 | `Strata.E1_table_nonreg` | `Strata/E.lean:69` | stated |
| E1d | 01-setup.tex:304 | `Strata.E1_table_counts` | `Strata/E.lean:73` | stated |
| E2 | 01-setup.tex:82–83 | `Strata.E2_chen_moduli` | `Strata/E.lean:79` | stated |
| E3 | 01-setup.tex:307 | `Strata.E3_units_128` | `Strata/E.lean:83` | stated |
| E4 | fig:stratification, 01-setup.tex:132–133 | `Strata.E4_strata_counts` | `Strata/E.lean:91` | stated |
| E5a | fig:stratum | `Strata.E5_fig1_15` | `Strata/E.lean:98` | stated |
| E5b | fig:stratum | `Strata.E5_fig1_20` | `Strata/E.lean:105` | stated |
| E5c | fig:descent | `Strata.E5_fig5_120` | `Strata/E.lean:113` | stated |
| E6 | 09-results-crt.tex:22–23, 36 | `Strata.E6_predicted` | `Strata/E.lean:117` | stated |

Tiers: **A** the stratum algebra (`01-setup.tex`) · **B** regularity, descent, fibration and
the character diagonal · **C** the measurement's own mathematics (`05-methods.tex`) · **D** the
indicators' spectra (`09-results-crt.tex`) · **E** concrete values the paper prints, by
`decide`.

Before any proof, every statement was checked numerically against brute force: tiers A and B at
every divisor pair of every `n ≤ 120`, the Ramanujan closed form at every `q < 60` and
`k < 70`, D0 and D1 at every `n < 50`, and every value in tier E. There were 0 disagreements.

## Findings: the Lean statement and the paper disagree

A disagreement is a finding. It is resolved in the paper, never by weakening the Lean
statement to match the prose.

- **L-F1, `fig:stratification` caption (`01-setup.tex:132`).** The caption says "There is one
  stratum at a prime". The strata are the blocks `J_d × J_e`, and a prime has two classes
  (the units and `{0}`), so there are **4**. The figure itself prints "2 classes, 4 strata".
  The other four counts in the caption (9, 16, 16, 256) are the squared class counts. See E4.
- **L-F2, `01-setup.tex:79–80`.** The paper says nilpotents "exist in ℤ/nℤ exactly when n
  is not square-free". `0` is nilpotent in every ring, so the claim holds only for
  **nonzero** nilpotents. A6 states the corrected form.

## Completeness

The statement list was made by sweeping the `.tex` for every `theorem`, `proposition` and
`definition` environment, every labelled equation, and every "exactly", "iff", "bijection"
and "bound", and then diffing the result against the draft list in the plan. The sweep added
A0, B4b–c, B6, C6, D0, D2b–d, D4 and E0–E6 to the draft. Found in the sweep and deliberately
excluded:

- `eq:chen`: Chen et al.'s Theorem 3.4, which the paper quotes and does not claim.
- The `30.5×` energy ratio at `n = 165` (`09-results-crt.tex:121`): a computed ratio of real
  sums, checked by brute force only.
- The loss and `p` clauses of `prop:generator`: C3 proves the permutation invariance and the
  projection conjugation that its proof rests on. The losses themselves need the trained
  model's logits.
- `eq:dlog`, `eq:character`, `eq:transform`: definitions, not claims. C2 covers what the paper
  claims about them.
