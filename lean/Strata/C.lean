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
  have hmap : (M * Q).map ((↑) : ℝ → ℂ) =
      (M.map ((↑) : ℝ → ℂ) : Matrix G D ℂ) * (Q.map ((↑) : ℝ → ℂ) : Matrix D D ℂ) := by
    ext i j; simp [Matrix.mul_apply]
  rw [hmap, ← Matrix.mul_assoc]
  set Y := X * M.map ((↑) : ℝ → ℂ)
  have hQQ : Q * Q.transpose = 1 := (Matrix.mem_orthogonalGroup_iff D ℝ).mp hQ
  have hrow : ∀ i i', ∑ j, Q i j * Q i' j = if i = i' then 1 else 0 := fun i i' => by
    have := congrFun (congrFun hQQ i) i'
    simpa [Matrix.mul_apply, Matrix.one_apply] using this
  apply Complex.ofReal_injective
  push_cast
  simp only [Complex.normSq_eq_conj_mul_self, Matrix.mul_apply, Matrix.map_apply, map_sum,
    map_mul, Complex.conj_ofReal, Finset.sum_mul_sum]
  calc ∑ j, ∑ i, ∑ i', (starRingEnd ℂ) (Y k i) * ↑(Q i j) * (Y k i' * ↑(Q i' j))
      = ∑ i, ∑ i', (starRingEnd ℂ) (Y k i) * Y k i' * ((∑ j, Q i j * Q i' j : ℝ) : ℂ) := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun i' _ => ?_
        push_cast
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun j _ => by ring
    _ = ∑ i, (starRingEnd ℂ) (Y k i) * Y k i := by
        simp only [hrow, apply_ite ((↑) : ℝ → ℂ), Complex.ofReal_one, Complex.ofReal_zero,
          mul_ite, mul_one, mul_zero, Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]

/-- C2a, `prop:generator`: if `gcd(t, φ) = 1` and `g` has order `φ`, then `g^t` has order `φ`
too, so it is a valid choice of primitive root. -/
theorem C2_generator_valid {G : Type*} [Group G] {g : G} {φ t : ℕ}
    (hg : orderOf g = φ) (ht : Nat.Coprime t φ) : orderOf (g ^ t) = φ := by
  rw [← hg]
  exact Nat.Coprime.orderOf_pow (by rw [hg]; exact ht.symm)

/-- C2b, `prop:generator`: `dlog_{g^t} = t⁻¹ · dlog_g mod φ` — if `g^a = x` and `(g^t)^b = x`
then `t b ≡ a (mod φ)`. -/
theorem C2_dlog {G : Type*} [Group G] {g x : G} {φ t a b : ℕ}
    (hg : orderOf g = φ) (ha : g ^ a = x) (hb : (g ^ t) ^ b = x) :
    ((t * b : ℕ) : ZMod φ) = a := by
  rw [ZMod.natCast_eq_natCast_iff, ← hg, ← pow_eq_pow_iff_modEq, pow_mul, hb, ha]

/-- C2c, `prop:generator`: `k ↦ t k` is a bijection of `ℤ/φℤ` when `gcd(t, φ) = 1`, and it sends
the diagonal `(k, k)` to the diagonal. -/
theorem C2_index_perm {φ t : ℕ} (ht : Nat.Coprime t φ) :
    Function.Bijective (fun k : ZMod φ => (t : ZMod φ) * k) :=
  Units.mulLeft_bijective (ZMod.unitOfCoprime t ht)

/-- C3a, `eq:equivariance`: Gini is a symmetric function of the spectrum. -/
theorem C3_gini_perm {M : ℕ} (A : Fin M → ℝ) (σ : Equiv.Perm (Fin M)) :
    gini (A ∘ σ) = gini A := by
  simp only [gini, Function.comp_apply, Equiv.sum_comp σ (fun i => |A i|),
    fun i => Equiv.sum_comp σ (fun j => abs (|A i| - |A j|))]
  rw [Equiv.sum_comp σ (fun i => ∑ j, abs (|A i| - |A j|))]

/-- C3b, `eq:equivariance`: PR is a symmetric function of the spectrum. -/
theorem C3_pr_perm {M : ℕ} (A : Fin M → ℝ) (σ : Equiv.Perm (Fin M)) :
    pr (A ∘ σ) = pr A := by
  simp only [pr, Function.comp_apply, Equiv.sum_comp σ (fun i => A i ^ 4),
    Equiv.sum_comp σ (fun i => A i ^ 2)]

/-- C3c, `eq:equivariance`: relabelling both character axes by `τ` conjugates the projection
`Π_S` of `eq:project` into `Π_{(τ×τ)S}`. -/
theorem C3_proj_conj {κ β : Type*} [Zero β] (τ : Equiv.Perm κ) (S : Set (κ × κ))
    [DecidablePred (· ∈ S)] [DecidablePred (· ∈ (τ.prodCongr τ) '' S)] (L : κ × κ → β) :
    proj ((τ.prodCongr τ) '' S) (L ∘ (τ.prodCongr τ).symm) =
      proj S L ∘ (τ.prodCongr τ).symm := by
  funext p
  simp only [proj, Function.comp_apply, Equiv.image_eq_preimage_symm, Set.mem_preimage]

private lemma sum_sum_eq_zero {M : ℕ} (f : Fin M → Fin M → ℝ) (h : ∀ i j, 0 ≤ f i j) :
    ∑ i, ∑ j, f i j = 0 ↔ ∀ i j, f i j = 0 := by
  rw [Finset.sum_eq_zero_iff_of_nonneg fun i _ => Finset.sum_nonneg fun j _ => h i j]
  simp only [Finset.mem_univ, true_implies]
  refine forall_congr' fun i => ?_
  rw [Finset.sum_eq_zero_iff_of_nonneg fun j _ => h i j]
  simp only [Finset.mem_univ, true_implies]

/-- At most one nonzero entry, stated pairwise or pointwise. -/
private lemma onehot {M : ℕ} (a : Fin M → ℝ) (hA : ∃ i, a i ≠ 0) :
    (∀ i j, i ≠ j → a i = 0 ∨ a j = 0) ↔ ∃ i, ∀ j, j ≠ i → a j = 0 := by
  obtain ⟨i₀, hi₀⟩ := hA
  refine ⟨fun h => ⟨i₀, fun j hj => (h i₀ j (Ne.symm hj)).resolve_left hi₀⟩, ?_⟩
  rintro ⟨i, hi⟩ x y hxy
  by_cases hx : x = i
  · exact Or.inr (hi y (hx ▸ Ne.symm hxy))
  · exact Or.inl (hi x hx)

/-- `Σᵢ Σⱼ f ≤ Σᵢ Σⱼ g` termwise, with equality iff every term is equal. -/
private lemma sum_sum_eq_iff_of_le {M : ℕ} (f g : Fin M → Fin M → ℝ) (h : ∀ i j, f i j ≤ g i j) :
    ∑ i, ∑ j, f i j = ∑ i, ∑ j, g i j ↔ ∀ i j, f i j = g i j := by
  have hz := sum_sum_eq_zero (fun i j => g i j - f i j) fun i j => sub_nonneg.mpr (h i j)
  simp only [Finset.sum_sub_distrib, sub_eq_zero] at hz
  rw [eq_comm, hz]
  exact forall_congr' fun i => forall_congr' fun j => eq_comm

private lemma abs_sub_eq_add {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    |x - y| = x + y ↔ x = 0 ∨ y = 0 := by
  constructor
  · intro h
    rcases abs_choice (x - y) with h' | h' <;> rw [h'] at h
    · right; linarith
    · left; linarith
  · rintro (rfl | rfl)
    · simp [abs_of_nonneg hy]
    · simp [abs_of_nonneg hx]

/-- The mean-difference numerator is at most `2 M S − 2 S`, termwise. -/
private lemma gini_term_le {M : ℕ} (a : Fin M → ℝ) (ha : ∀ i, 0 ≤ a i) (i j : Fin M) :
    |a i - a j| ≤ a i + a j - if i = j then 2 * a i else 0 := by
  split_ifs with hij
  · subst hij; rw [sub_self, abs_zero]; linarith
  · have := ha i; have := ha j
    rw [abs_le]; constructor <;> linarith

private lemma gini_rhs_sum {M : ℕ} (a : Fin M → ℝ) :
    ∑ i, ∑ j, (a i + a j - if i = j then 2 * a i else 0) = 2 * M * ∑ i, a i - 2 * ∑ i, a i := by
  simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.sum_ite_eq,
    Finset.mem_univ, ↓reduceIte, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, ← Finset.mul_sum]
  ring

/-- On a spectrum with `Σ|A| ≠ 0`, `gini` is the mean-difference ratio. -/
private lemma gini_eq {M : ℕ} (A : Fin M → ℝ) (hS : ∑ i, |A i| ≠ 0) :
    gini A = (∑ i, ∑ j, abs (|A i| - |A j|)) / (2 * M * ∑ i, |A i|) := by
  simp only [gini, hS, ↓reduceIte]

/-- A non-zero spectrum has `Σ|A| > 0` and `M > 0`. -/
private lemma pos_of_ne {M : ℕ} (A : Fin M → ℝ) (hA : ∃ i, A i ≠ 0) :
    0 < ∑ i, |A i| ∧ (0 : ℝ) < M := by
  obtain ⟨i, hi⟩ := hA
  exact ⟨Finset.sum_pos' (fun j _ => abs_nonneg _) ⟨i, mem_univ _, abs_pos.mpr hi⟩,
    Nat.cast_pos.mpr (Fin.pos i)⟩

private lemma one_sub_mul {M : ℕ} (hM : (0 : ℝ) < M) (S : ℝ) :
    (1 - 1 / (M : ℝ)) * (2 * M * S) = 2 * M * S - 2 * S := by
  field_simp

/-- C4a, `eq:gini`: on a sorted non-negative spectrum `gini` is the paper's formula
`Σ (2i − M − 1) A_(i) / (M Σ A_(i))` (with `i` counted from 1). -/
theorem C4_gini_sorted {M : ℕ} (A : Fin M → ℝ) (h0 : ∀ i, 0 ≤ A i) (hmono : Monotone A)
    (hs : ∑ i, A i ≠ 0) :
    gini A = (∑ i : Fin M, (2 * ((i : ℕ) + 1 : ℝ) - M - 1) * A i) / (M * ∑ i, A i) := by
  have habs : ∀ i, |A i| = A i := fun i => abs_of_nonneg (h0 i)
  have hS : ∑ i, |A i| ≠ 0 := by simpa only [habs] using hs
  rw [gini_eq A hS]
  simp only [habs]
  -- `s i j = [j < i] − [i < j]`, and `|A i − A j| = (A i − A j) s i j` since `A` is monotone
  obtain ⟨s, hs⟩ : ∃ s : Fin M → Fin M → ℝ,
      s = fun i j => (if j < i then 1 else 0) - if i < j then 1 else 0 := ⟨_, rfl⟩
  have hpair : ∀ i j, |A i - A j| = (A i - A j) * s i j := fun i j => by
    rcases lt_trichotomy i j with h | rfl | h
    · simp only [hs, h, h.not_gt, ↓reduceIte]
      rw [abs_of_nonpos (sub_nonpos.mpr (hmono h.le))]; ring
    · simp [hs]
    · simp only [hs, h, h.not_gt, ↓reduceIte]
      rw [abs_of_nonneg (sub_nonneg.mpr (hmono h.le))]; ring
  have hanti : ∀ i j, s i j = - s j i := fun i j => by simp only [hs]; ring
  have hrow : ∀ i : Fin M, ∑ j, s i j = 2 * ((i : ℕ) + 1 : ℝ) - M - 1 := fun i => by
    simp only [hs]
    rw [Finset.sum_sub_distrib, Finset.sum_boole, Finset.sum_boole, Finset.filter_gt_eq_Iio,
      Finset.filter_lt_eq_Ioi, Fin.card_Iio, Fin.card_Ioi]
    have := i.isLt
    rw [Nat.sub_sub, Nat.cast_sub (by omega)]
    push_cast; ring
  have hcol : ∀ j : Fin M, ∑ i, s i j = -(2 * ((j : ℕ) + 1 : ℝ) - M - 1) := fun j => by
    rw [← hrow j, ← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun i _ => hanti i j
  have hD : ∑ i, ∑ j, |A i - A j| = 2 * ∑ i : Fin M, (2 * ((i : ℕ) + 1 : ℝ) - M - 1) * A i := by
    have e : ∀ i, ∑ j, |A i - A j| = A i * ∑ j, s i j - ∑ j, A j * s i j := fun i => by
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun j _ => by rw [hpair]; ring
    rw [Finset.sum_congr rfl fun i _ => e i, Finset.sum_sub_distrib,
      Finset.sum_comm (f := fun i j => A j * s i j)]
    simp_rw [← Finset.mul_sum, hrow, hcol]
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [hD, mul_assoc, mul_div_mul_left _ _ two_ne_zero]

/-- C4b, `eq:gini`: `Gini ∈ [0, 1 − 1/M]`. -/
theorem C4_gini_bounds {M : ℕ} (A : Fin M → ℝ) (hM : 0 < M) :
    0 ≤ gini A ∧ gini A ≤ 1 - 1 / M := by
  have hM' : (0 : ℝ) < M := Nat.cast_pos.mpr hM
  have h1M : 1 / (M : ℝ) ≤ 1 := by
    rw [div_le_one hM']; exact_mod_cast hM
  by_cases hS : ∑ i, |A i| = 0
  · simp only [gini, hS, ↓reduceIte]; constructor <;> linarith
  have hSpos : 0 < ∑ i, |A i| :=
    lt_of_le_of_ne (Finset.sum_nonneg fun i _ => abs_nonneg _) (Ne.symm hS)
  have hden : 0 < 2 * (M : ℝ) * ∑ i, |A i| := by positivity
  rw [gini_eq A hS]
  refine ⟨div_nonneg (Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => abs_nonneg _)
    hden.le, ?_⟩
  rw [div_le_iff₀ hden, one_sub_mul hM', ← gini_rhs_sum]
  exact Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ =>
    gini_term_le (fun i => |A i|) (fun i => abs_nonneg _) i j

/-- C4c, `eq:gini`: on a non-zero spectrum, `Gini = 0` iff it is flat. -/
theorem C4_gini_zero_iff {M : ℕ} (A : Fin M → ℝ) (hA : ∃ i, A i ≠ 0) :
    gini A = 0 ↔ ∀ i j, |A i| = |A j| := by
  obtain ⟨hSpos, hM'⟩ := pos_of_ne A hA
  rw [gini_eq A hSpos.ne', div_eq_zero_iff, or_iff_left (by positivity),
    sum_sum_eq_zero _ fun i j => abs_nonneg _]
  simp only [abs_eq_zero, sub_eq_zero]

/-- C4d, `eq:gini`: on a non-zero spectrum, `Gini = 1 − 1/M` iff one frequency carries
everything. -/
theorem C4_gini_max_iff {M : ℕ} (A : Fin M → ℝ) (hA : ∃ i, A i ≠ 0) :
    gini A = 1 - 1 / M ↔ ∃ i, ∀ j, j ≠ i → A j = 0 := by
  obtain ⟨hSpos, hM'⟩ := pos_of_ne A hA
  have hden : 0 < 2 * (M : ℝ) * ∑ i, |A i| := by positivity
  rw [gini_eq A hSpos.ne', div_eq_iff hden.ne', one_sub_mul hM', ← gini_rhs_sum,
    sum_sum_eq_iff_of_le _ _ (gini_term_le (fun i => |A i|) fun i => abs_nonneg _),
    ← onehot A hA]
  refine forall_congr' fun i => forall_congr' fun j => ?_
  by_cases hij : i = j
  · subst hij; simp only [sub_self, abs_zero, ↓reduceIte, ne_eq, not_true_eq_false,
      false_implies, iff_true]; ring
  · simp only [hij, ↓reduceIte, sub_zero, true_implies, ne_eq, not_false_eq_true]
    rw [abs_sub_eq_add (abs_nonneg _) (abs_nonneg _), abs_eq_zero, abs_eq_zero]

private lemma sum_pow4_pos {M : ℕ} (u : Fin M → ℝ) (hu : ∃ i, u i ≠ 0) : 0 < ∑ i, u i ^ 4 := by
  obtain ⟨i, hi⟩ := hu
  exact Finset.sum_pos' (fun j _ => by positivity) ⟨i, mem_univ _, by positivity⟩

/-- Lagrange's identity: `Σᵢ Σⱼ (bᵢ − bⱼ)² = 2 M Σ b² − 2 (Σ b)²`. -/
private lemma lagrange {M : ℕ} (b : Fin M → ℝ) :
    ∑ i, ∑ j, (b i - b j) ^ 2 = 2 * M * ∑ i, b i ^ 2 - 2 * (∑ i, b i) ^ 2 := by
  have h : ∀ i j, (b i - b j) ^ 2 = b i ^ 2 + b j ^ 2 - 2 * (b i * b j) := fun i j => by ring
  have e1 : ∑ i, ∑ _j : Fin M, b i ^ 2 = M * ∑ i, b i ^ 2 := by simp [Finset.mul_sum]
  have e2 : ∑ _i : Fin M, ∑ j, b j ^ 2 = M * ∑ i, b i ^ 2 := by simp
  have e3 : ∑ i, ∑ j, 2 * (b i * b j) = 2 * (∑ i, b i) ^ 2 := by
    rw [sq, Finset.sum_mul_sum, Finset.mul_sum]
    simp only [Finset.mul_sum]
  simp only [h, Finset.sum_sub_distrib, Finset.sum_add_distrib]
  rw [e1, e2, e3]
  ring

/-- `(Σ b)² − Σ b² = Σᵢ Σⱼ [i ≠ j] bᵢ bⱼ`. -/
private lemma offdiag {M : ℕ} (b : Fin M → ℝ) :
    (∑ i, b i) ^ 2 - ∑ i, b i ^ 2 = ∑ i, ∑ j, if i = j then 0 else b i * b j := by
  have h : ∀ i j, (if i = j then 0 else b i * b j) = b i * b j - if i = j then b i * b j else 0 :=
    fun i j => by split_ifs <;> simp
  simp only [h, Finset.sum_sub_distrib, Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]
  rw [sq (∑ i, b i), Finset.sum_mul_sum]
  simp only [sq]

/-- On a non-zero spectrum `pr` is `(Σ b)² / Σ b²` with `b = A²`, and the denominator is positive. -/
private lemma pr_eq {M : ℕ} (A : Fin M → ℝ) (hA : ∃ i, A i ≠ 0) :
    0 < ∑ i, (A i ^ 2) ^ 2 ∧ pr A = (∑ i, A i ^ 2) ^ 2 / ∑ i, (A i ^ 2) ^ 2 := by
  have h4 : ∀ i, (A i ^ 2) ^ 2 = A i ^ 4 := fun i => by ring
  simp only [h4]
  exact ⟨sum_pow4_pos A hA, by simp only [pr, (sum_pow4_pos A hA).ne', ↓reduceIte]⟩

/-- C5a, `eq:pr`: on a non-zero spectrum, `PR ∈ [1, M]`. -/
theorem C5_pr_bounds {M : ℕ} (A : Fin M → ℝ) (hA : ∃ i, A i ≠ 0) :
    1 ≤ pr A ∧ pr A ≤ M := by
  obtain ⟨hQ, hpr⟩ := pr_eq A hA
  rw [hpr, one_le_div hQ, div_le_iff₀ hQ]
  have h1 := offdiag fun i => A i ^ 2
  have h2 := lagrange fun i => A i ^ 2
  have n1 : 0 ≤ ∑ i, ∑ j, if i = j then (0 : ℝ) else A i ^ 2 * A j ^ 2 :=
    Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => by split_ifs <;> positivity
  have n2 : 0 ≤ ∑ i : Fin M, ∑ j : Fin M, (A i ^ 2 - A j ^ 2) ^ 2 :=
    Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => by positivity
  constructor <;> linarith

/-- C5b, `eq:pr`: on a non-zero spectrum, `PR = M` iff it is flat. -/
theorem C5_pr_max_iff {M : ℕ} (A : Fin M → ℝ) (hA : ∃ i, A i ≠ 0) :
    pr A = M ↔ ∀ i j, |A i| = |A j| := by
  obtain ⟨hQ, hpr⟩ := pr_eq A hA
  rw [hpr, div_eq_iff hQ.ne']
  have h2 := lagrange fun i => A i ^ 2
  have hz := sum_sum_eq_zero (fun i j => (A i ^ 2 - A j ^ 2) ^ 2) fun i j => by positivity
  simp only [pow_eq_zero_iff two_ne_zero, sub_eq_zero, sq_eq_sq_iff_abs_eq_abs] at hz
  rw [← hz]
  constructor <;> intro h <;> linarith

/-- C5c, `eq:pr`: on a non-zero spectrum, `PR = 1` iff one frequency carries everything. -/
theorem C5_pr_one_iff {M : ℕ} (A : Fin M → ℝ) (hA : ∃ i, A i ≠ 0) :
    pr A = 1 ↔ ∃ i, ∀ j, j ≠ i → A j = 0 := by
  obtain ⟨hQ, hpr⟩ := pr_eq A hA
  rw [hpr, div_eq_one_iff_eq hQ.ne', ← onehot A hA]
  have h1 := offdiag fun i => A i ^ 2
  have hz := sum_sum_eq_zero (fun i j => if i = j then (0 : ℝ) else A i ^ 2 * A j ^ 2)
    fun i j => by split_ifs <;> positivity
  have hc : (∀ i j, (if i = j then (0 : ℝ) else A i ^ 2 * A j ^ 2) = 0) ↔
      ∀ i j, i ≠ j → A i = 0 ∨ A j = 0 := by
    refine forall_congr' fun i => forall_congr' fun j => ?_
    by_cases hij : i = j <;> simp [hij]
  rw [← hc, ← hz]
  constructor <;> intro h <;> linarith

/-- C6, `eq:conventions`: `IPR₂(u) = 1 / PR(u)` on the same spectrum. -/
theorem C6_ipr_pr {M : ℕ} (u : Fin M → ℝ) (hu : ∃ i, u i ≠ 0) : ipr2 u = 1 / pr u := by
  simp only [ipr2, pr, (sum_pow4_pos u hu).ne', ↓reduceIte, one_div_div]

end Strata
