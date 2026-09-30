import Strata.Basic

/-!
# Tier E — concrete values the paper prints, by computation

`decide` only. A count too large for kernel reduction moves to its own file, whose extra axiom
(`Lean.ofReduceBool`, from `native_decide`) the gate names and fails on; it is never allowed
silently.
-/

open Finset Pointwise

namespace Strata

/-- A Nat-level mirror of `Regular`, so that a count over it is decidable. E0 ties the two. -/
def RegularNat (n d : ℕ) : Prop := ∃ ε : Fin n, Nat.gcd ε n = d ∧ (ε * ε : ℕ) % n = ε

instance (n d : ℕ) : Decidable (RegularNat n d) := by unfold RegularNat; infer_instance

/-- The number of non-regular `J`-classes of `ℤ/nℤ`. -/
def nonRegCount (n : ℕ) : ℕ := (n.divisors.filter fun d => ¬ RegularNat n d).card

/-- E0: the mirror agrees with `Regular`. -/
theorem E0_regularNat {n d : ℕ} [NeZero n] : RegularNat n d ↔ Regular n d := by
  unfold RegularNat Regular J
  constructor
  · rintro ⟨ε, hg, hi⟩
    refine ⟨(ε.val : ZMod n), ?_, ?_⟩
    · simp [ZMod.val_cast_of_lt ε.isLt, hg]
    · apply ZMod.val_injective
      rw [ZMod.val_mul, ZMod.val_cast_of_lt ε.isLt]
      exact hi
  · rintro ⟨ε, hmem, hi⟩
    refine ⟨⟨ε.val, ZMod.val_lt ε⟩, by simpa using hmem, ?_⟩
    have h := congrArg ZMod.val hi
    rwa [ZMod.val_mul] at h

/-- One row of `tab:moduli`: `n`, `ω(n)`, square-free, cyclic unit group, `φ(n)`, zero-divisor
density in thousandths as printed, `|𝒥|`, non-regular classes. -/
structure Row where
  n : ℕ
  ω : ℕ
  sqfree : Bool
  cyclic : Bool
  φ : ℕ
  zdd : ℕ
  nJ : ℕ
  nonReg : ℕ

/-- `tab:moduli`, transcribed from `paper/sections/01-setup.tex`. -/
def table : List Row :=
  [⟨49, 1, false, true, 42, 143, 3, 1⟩,   ⟨54, 2, false, true, 18, 667, 8, 4⟩,
   ⟨63, 2, false, false, 36, 429, 6, 2⟩,  ⟨75, 2, false, false, 40, 467, 6, 2⟩,
   ⟨81, 1, false, true, 54, 333, 5, 3⟩,   ⟨91, 2, true, false, 72, 209, 4, 0⟩,
   ⟨98, 2, false, true, 42, 571, 6, 2⟩,   ⟨99, 2, false, false, 60, 394, 6, 2⟩,
   ⟨100, 2, false, false, 40, 600, 9, 5⟩, ⟨105, 3, true, false, 48, 543, 8, 0⟩,
   ⟨113, 1, true, true, 112, 9, 2, 0⟩,   ⟨119, 2, true, false, 96, 193, 4, 0⟩,
   ⟨120, 3, false, false, 32, 733, 16, 8⟩, ⟨121, 1, false, true, 110, 91, 3, 1⟩,
   ⟨123, 2, true, false, 80, 350, 4, 0⟩,  ⟨125, 1, false, true, 100, 200, 4, 2⟩,
   ⟨127, 1, true, true, 126, 8, 2, 0⟩,   ⟨128, 1, false, false, 64, 500, 8, 6⟩,
   ⟨131, 1, true, true, 130, 8, 2, 0⟩,   ⟨143, 2, true, false, 120, 161, 4, 0⟩,
   ⟨147, 2, false, false, 84, 429, 6, 2⟩, ⟨165, 3, true, false, 80, 515, 8, 0⟩,
   ⟨169, 1, false, true, 156, 77, 3, 1⟩]

/-- E1a, `tab:moduli`: every arithmetic column. `zdd` is `(n − φ(n))/n` rounded to three
decimals; `|𝒥|` is the divisor count (A1 makes every `J_d` non-empty). -/
theorem E1_table_arith : ∀ r ∈ table,
    r.n.primeFactors.card = r.ω ∧ (Squarefree r.n ↔ r.sqfree = true) ∧
    r.n.totient = r.φ ∧ r.n.divisors.card = r.nJ ∧ nonRegCount r.n = r.nonReg ∧
    2 * 1000 * (r.n - r.n.totient) + r.n ≥ 2 * r.zdd * r.n ∧
    2 * 1000 * (r.n - r.n.totient) < (2 * r.zdd + 1) * r.n := by
  decide +kernel

/-- E1b, `tab:moduli`: the cyclic column. -/
theorem E1_table_cyclic : ∀ r ∈ table, IsCyclic (ZMod r.n)ˣ ↔ r.cyclic = true := by
  have pp : ∀ p k, p.Prime → p ≠ 2 → IsCyclic (ZMod (p ^ k))ˣ := fun p k hp h2 =>
    ZMod.isCyclic_units_of_prime_pow p hp h2 k
  have tpp : ∀ p k, p.Prime → p ≠ 2 → IsCyclic (ZMod (2 * p ^ k))ˣ := fun p k hp h2 =>
    (ZMod.isCyclic_units_two_mul_iff_of_odd _ ((hp.odd_of_ne_two h2).pow)).mpr (pp p k hp h2)
  have nc : ∀ a b, Odd a → a ≠ 1 → Odd b → b ≠ 1 → Nat.Coprime a b →
      ¬ IsCyclic (ZMod (a * b))ˣ := ZMod.not_isCyclic_units_of_mul_coprime
  have nc4 : ∀ k, k ≠ 0 → k ≠ 1 → ¬ IsCyclic (ZMod (4 * k))ˣ := fun k h0 h1 => by
    rw [ZMod.isCyclic_units_four_mul_iff]; omega
  intro r hr
  simp only [table, List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact iff_of_true (pp 7 2 (by norm_num) (by norm_num)) rfl
  · exact iff_of_true (tpp 3 3 (by norm_num) (by norm_num)) rfl
  · exact iff_of_false (nc 7 9 (by decide) (by decide) (by decide) (by decide) (by decide)) (by decide)
  · exact iff_of_false (nc 3 25 (by decide) (by decide) (by decide) (by decide) (by decide)) (by decide)
  · exact iff_of_true (pp 3 4 (by norm_num) (by norm_num)) rfl
  · exact iff_of_false (nc 7 13 (by decide) (by decide) (by decide) (by decide) (by decide)) (by decide)
  · exact iff_of_true (tpp 7 2 (by norm_num) (by norm_num)) rfl
  · exact iff_of_false (nc 9 11 (by decide) (by decide) (by decide) (by decide) (by decide)) (by decide)
  · exact iff_of_false (nc4 25 (by decide) (by decide)) (by decide)
  · exact iff_of_false (nc 3 35 (by decide) (by decide) (by decide) (by decide) (by decide)) (by decide)
  · exact iff_of_true (pp 113 1 (by norm_num) (by norm_num)) rfl
  · exact iff_of_false (nc 7 17 (by decide) (by decide) (by decide) (by decide) (by decide)) (by decide)
  · exact iff_of_false (nc4 30 (by decide) (by decide)) (by decide)
  · exact iff_of_true (pp 11 2 (by norm_num) (by norm_num)) rfl
  · exact iff_of_false (nc 3 41 (by decide) (by decide) (by decide) (by decide) (by decide)) (by decide)
  · exact iff_of_true (pp 5 3 (by norm_num) (by norm_num)) rfl
  · exact iff_of_true (pp 127 1 (by norm_num) (by norm_num)) rfl
  · exact iff_of_false (nc4 32 (by decide) (by decide)) (by decide)
  · exact iff_of_true (pp 131 1 (by norm_num) (by norm_num)) rfl
  · exact iff_of_false (nc 11 13 (by decide) (by decide) (by decide) (by decide) (by decide)) (by decide)
  · exact iff_of_false (nc 3 49 (by decide) (by decide) (by decide) (by decide) (by decide)) (by decide)
  · exact iff_of_false (nc 3 55 (by decide) (by decide) (by decide) (by decide) (by decide)) (by decide)
  · exact iff_of_true (pp 13 2 (by norm_num) (by norm_num)) rfl

/-- E1c, `tab:moduli` caption and `01-setup.tex:84–86`: non-regular count is `0` at every
square-free modulus and positive at every other one. -/
theorem E1_table_nonreg : ∀ r ∈ table, (r.nonReg = 0 ↔ r.sqfree = true) := by
  decide

/-- E1d, `01-setup.tex:304`: of the 23 moduli, 14 are not square-free and 6 are prime powers. -/
theorem E1_table_counts :
    table.length = 23 ∧ (table.filter fun r => !r.sqfree).length = 14 ∧
    (table.filter fun r => r.ω == 1 && !r.sqfree).length = 6 := by
  decide

/-- E2, `01-setup.tex:82–83`: Chen et al.'s own moduli have no non-regular class. -/
theorem E2_chen_moduli : nonRegCount 143 = 0 ∧ nonRegCount 154 = 0 ∧ nonRegCount 165 = 0 := by
  decide +kernel

/-- The unit `5` of `ℤ/128ℤ`, of order `32`. -/
private def g5 : (ZMod 128)ˣ := ZMod.unitOfCoprime 5 (by decide)

/-- `(a, b) ↦ (−1)^a · 5^b`, the map E3 inverts. -/
private def e3hom : Multiplicative (ZMod 2 × ZMod 32) →* (ZMod 128)ˣ where
  toFun x := (-1) ^ (Multiplicative.toAdd x).1.val * g5 ^ (Multiplicative.toAdd x).2.val
  map_one' := by decide
  map_mul' x y := by
    have h1 : ((-1 : (ZMod 128)ˣ)) ^ 2 = 1 := by decide
    have h5 : g5 ^ 32 = 1 := by decide
    simp only [toAdd_mul, Prod.fst_add, Prod.snd_add, ZMod.val_add]
    rw [← pow_eq_pow_mod _ h1, ← pow_eq_pow_mod _ h5, pow_add, pow_add]
    exact mul_mul_mul_comm _ _ _ _

/-- E3, `01-setup.tex:307`: `(ℤ/128ℤ)ˣ ≅ ℤ₂ × ℤ₃₂`, so it has no discrete log. -/
theorem E3_units_128 :
    Nonempty ((ZMod 128)ˣ ≃* Multiplicative (ZMod 2 × ZMod 32)) ∧ ¬ IsCyclic (ZMod 128)ˣ := by
  refine ⟨⟨(MulEquiv.ofBijective e3hom ?_).symm⟩, ?_⟩
  · rw [Fintype.bijective_iff_injective_and_card]
    refine ⟨by decide, ?_⟩
    rw [ZMod.card_units_eq_totient]
    decide
  · rw [show (128 : ℕ) = 4 * 32 from rfl, ZMod.isCyclic_units_four_mul_iff]
    decide

/-- E4, `fig:stratification` caption (`01-setup.tex:132–133`) and the figure itself: the
number of strata `J_d × J_e` is the square of the class count — `4, 9, 16, 16, 256` at
`113, 121, 125, 119, 120`. **The caption prints `1` at the prime; the figure prints `4`.**
Recorded as finding L-F1 in `lean/README.md`. -/
theorem E4_strata_counts :
    (Nat.divisors 113).card ^ 2 = 4 ∧ (Nat.divisors 121).card ^ 2 = 9 ∧
    (Nat.divisors 125).card ^ 2 = 16 ∧ (Nat.divisors 119).card ^ 2 = 16 ∧
    (Nat.divisors 120).card ^ 2 = 256 := by
  decide

/-- E5a, `fig:stratum` caption: at `n = 15`, `J₃ · J₃ = J₃` and `6` is its idempotent. -/
theorem E5_fig1_15 : J 15 3 * J 15 3 = J 15 3 ∧ (6 : ZMod 15) ∈ J 15 3 ∧
    (6 : ZMod 15) * 6 = 6 := by
  decide

/-- E5b, `fig:stratum` caption: at `n = 20`, `J₂` has no idempotent, `J₂ · J₄ = J₄`, the
stratum has `m = 4`, `w = 2`, `q = 5`, `16` cells taking `4` distinct values, and the ringed
cell `(6, 12)` lies in it. -/
theorem E5_fig1_20 :
    ¬ Regular 20 2 ∧ J 20 2 * J 20 4 = J 20 4 ∧
    mOf 20 2 4 = 4 ∧ wOf 20 2 4 = 2 ∧ qOf 20 2 4 = 5 ∧
    (J 20 2 ×ˢ J 20 4).card = 16 ∧ ((J 20 2 ×ˢ J 20 4).image fun p => p.1 * p.2).card = 4 ∧
    (6 : ZMod 20) ∈ J 20 2 ∧ (12 : ZMod 20) ∈ J 20 4 := by
  rw [← E0_regularNat]
  decide

/-- E5c, `fig:descent` caption: at `n = 120`, `16` classes, `8` of them non-regular. -/
theorem E5_fig5_120 : (Nat.divisors 120).card = 16 ∧ nonRegCount 120 = 8 := by
  decide +kernel

/-- E6, `09-results-crt.tex:22–23, 36`: `P(165)`, `P(120)`, and all `56` bins at `n = 113`. -/
theorem E6_predicted :
    predicted 165 = {15, 30, 33, 45, 55, 60, 66, 75} ∧
    predicted 120 = {15, 24, 30, 40, 45, 48, 60} ∧ (predicted 113).card = 56 := by
  decide +kernel

end Strata
