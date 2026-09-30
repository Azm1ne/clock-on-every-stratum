import Strata.Basic

/-!
# Tier C — the measurement's own mathematics, `paper/sections/05-methods.tex`
-/

open Finset

namespace Strata

/-- C1, `prop:noop`: for any character-transform coefficients `X` (rows indexed by `k`), a real
orthogonal `Q` on the feature axis leaves every row's energy, hence `eq:amplitude`, unchanged:
`‖(X (M Q))_k‖ = ‖(X M)_k‖`. -/
theorem C1_noop {K G D : Type*} [Fintype G] [Fintype D] [DecidableEq D]
    (X : Matrix K G ℂ) (M : Matrix G D ℝ) (Q : Matrix D D ℝ)
    (hQ : Q ∈ Matrix.orthogonalGroup D ℝ) (k : K) :
    ∑ j, Complex.normSq ((X * (M * Q).map ((↑) : ℝ → ℂ)) k j) =
      ∑ j, Complex.normSq ((X * M.map ((↑) : ℝ → ℂ)) k j) := by
  sorry

/-- C2a, `prop:generator`: if `gcd(t, φ) = 1` and `g` has order `φ`, then `g^t` has order `φ`
too, so it is a valid choice of primitive root. -/
theorem C2_generator_valid {G : Type*} [Group G] {g : G} {φ t : ℕ}
    (hg : orderOf g = φ) (ht : Nat.Coprime t φ) : orderOf (g ^ t) = φ := by
  sorry

/-- C2b, `prop:generator`: `dlog_{g^t} = t⁻¹ · dlog_g mod φ` — if `g^a = x` and `(g^t)^b = x`
then `t b ≡ a (mod φ)`. -/
theorem C2_dlog {G : Type*} [Group G] {g x : G} {φ t a b : ℕ}
    (hg : orderOf g = φ) (ha : g ^ a = x) (hb : (g ^ t) ^ b = x) :
    ((t * b : ℕ) : ZMod φ) = a := by
  sorry

/-- C2c, `prop:generator`: `k ↦ t k` is a bijection of `ℤ/φℤ` when `gcd(t, φ) = 1`, and it sends
the diagonal `(k, k)` to the diagonal. -/
theorem C2_index_perm {φ t : ℕ} (ht : Nat.Coprime t φ) :
    Function.Bijective (fun k : ZMod φ => (t : ZMod φ) * k) := by
  sorry

/-- C3a, `eq:equivariance`: Gini is a symmetric function of the spectrum. -/
theorem C3_gini_perm {M : ℕ} (A : Fin M → ℝ) (σ : Equiv.Perm (Fin M)) :
    gini (A ∘ σ) = gini A := by
  sorry

/-- C3b, `eq:equivariance`: PR is a symmetric function of the spectrum. -/
theorem C3_pr_perm {M : ℕ} (A : Fin M → ℝ) (σ : Equiv.Perm (Fin M)) :
    pr (A ∘ σ) = pr A := by
  sorry

/-- C3c, `eq:equivariance`: relabelling both character axes by `τ` conjugates the projection
`Π_S` of `eq:project` into `Π_{(τ×τ)S}`. -/
theorem C3_proj_conj {κ β : Type*} [Zero β] (τ : Equiv.Perm κ) (S : Set (κ × κ))
    [DecidablePred (· ∈ S)] [DecidablePred (· ∈ (τ.prodCongr τ) '' S)] (L : κ × κ → β) :
    proj ((τ.prodCongr τ) '' S) (L ∘ (τ.prodCongr τ).symm) =
      proj S L ∘ (τ.prodCongr τ).symm := by
  sorry

/-- C4a, `eq:gini`: on a sorted non-negative spectrum `gini` is the paper's formula
`Σ (2i − M − 1) A_(i) / (M Σ A_(i))` (with `i` counted from 1). -/
theorem C4_gini_sorted {M : ℕ} (A : Fin M → ℝ) (h0 : ∀ i, 0 ≤ A i) (hmono : Monotone A)
    (hs : ∑ i, A i ≠ 0) :
    gini A = (∑ i : Fin M, (2 * ((i : ℕ) + 1 : ℝ) - M - 1) * A i) / (M * ∑ i, A i) := by
  sorry

/-- C4b, `eq:gini`: `Gini ∈ [0, 1 − 1/M]`. -/
theorem C4_gini_bounds {M : ℕ} (A : Fin M → ℝ) (hM : 0 < M) :
    0 ≤ gini A ∧ gini A ≤ 1 - 1 / M := by
  sorry

/-- C4c, `eq:gini`: on a non-zero spectrum, `Gini = 0` iff it is flat. -/
theorem C4_gini_zero_iff {M : ℕ} (A : Fin M → ℝ) (hA : ∃ i, A i ≠ 0) :
    gini A = 0 ↔ ∀ i j, |A i| = |A j| := by
  sorry

/-- C4d, `eq:gini`: on a non-zero spectrum, `Gini = 1 − 1/M` iff one frequency carries
everything. -/
theorem C4_gini_max_iff {M : ℕ} (A : Fin M → ℝ) (hA : ∃ i, A i ≠ 0) :
    gini A = 1 - 1 / M ↔ ∃ i, ∀ j, j ≠ i → A j = 0 := by
  sorry

/-- C5a, `eq:pr`: on a non-zero spectrum, `PR ∈ [1, M]`. -/
theorem C5_pr_bounds {M : ℕ} (A : Fin M → ℝ) (hA : ∃ i, A i ≠ 0) :
    1 ≤ pr A ∧ pr A ≤ M := by
  sorry

/-- C5b, `eq:pr`: on a non-zero spectrum, `PR = M` iff it is flat. -/
theorem C5_pr_max_iff {M : ℕ} (A : Fin M → ℝ) (hA : ∃ i, A i ≠ 0) :
    pr A = M ↔ ∀ i j, |A i| = |A j| := by
  sorry

/-- C5c, `eq:pr`: on a non-zero spectrum, `PR = 1` iff one frequency carries everything. -/
theorem C5_pr_one_iff {M : ℕ} (A : Fin M → ℝ) (hA : ∃ i, A i ≠ 0) :
    pr A = 1 ↔ ∃ i, ∀ j, j ≠ i → A j = 0 := by
  sorry

/-- C6, `eq:conventions`: `IPR₂(u) = 1 / PR(u)` on the same spectrum. -/
theorem C6_ipr_pr {M : ℕ} (u : Fin M → ℝ) (hu : ∃ i, u i ≠ 0) : ipr2 u = 1 / pr u := by
  sorry

end Strata
