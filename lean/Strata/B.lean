import Strata.A

/-!
# Tier B — regularity, descent, fibration, the character diagonal, `01-setup.tex`
-/

open Finset Pointwise

namespace Strata

variable {n : ℕ} [NeZero n]

/-- B2s, sharp form (`def:jclass` + `tab:moduli` caption): `J_d` is regular iff
`gcd(d, n/d) = 1`. B1 and B2 below, and tier E's non-regular counts, follow from it. -/
theorem B2_regular_iff {d : ℕ} (hd : d ∣ n) : Regular n d ↔ Nat.Coprime d (n / d) := by
  obtain ⟨k, rfl⟩ := hd
  have hn : d * k ≠ 0 := NeZero.ne _
  have hd0 : d ≠ 0 := left_ne_zero_of_mul hn
  have hk0 : k ≠ 0 := right_ne_zero_of_mul hn
  rw [Nat.mul_div_cancel_left k (Nat.pos_of_ne_zero hd0)]
  constructor
  · rintro ⟨ε, hε, hεε⟩
    have hx : Nat.gcd ε.val (d * k) = d := by simpa [J] using hε
    have hdx : d ∣ ε.val := by have := Nat.gcd_dvd_left ε.val (d * k); rwa [hx] at this
    obtain ⟨a, ha⟩ := hdx
    have hak : Nat.Coprime a k := by
      have := hx
      rw [ha, Nat.gcd_mul_left] at this
      exact (Nat.mul_eq_left hd0).mp this
    -- `ε² = ε` is `d a · d a ≡ d a (mod d k)`
    have h1 : d * (a * (d * a)) ≡ d * (a * 1) [MOD d * k] := by
      have := congrArg ZMod.val hεε
      rw [ZMod.val_mul, ha] at this
      have hlt : d * a < d * k := ha ▸ ZMod.val_lt ε
      have e1 : d * (a * (d * a)) = d * a * (d * a) := by ring
      rw [Nat.ModEq, e1, this, mul_one, Nat.mod_eq_of_lt hlt]
    have h2 : a * (d * a) ≡ a * 1 [MOD k] := Nat.ModEq.mul_left_cancel' hd0 h1
    have h3 : d * a ≡ 1 [MOD k] := Nat.ModEq.cancel_left_of_coprime hak.symm h2
    have h4 := h3.gcd_eq
    rw [Nat.gcd_one_left] at h4
    exact Nat.Coprime.coprime_mul_right h4
  · intro hcop
    -- the CRT idempotent: `ε ≡ 0 (mod d)`, `ε ≡ 1 (mod k)`
    have hxlt := Nat.chineseRemainder_lt_mul hcop 0 1 hd0 hk0
    obtain ⟨hx0, hx1⟩ := (Nat.chineseRemainder hcop 0 1).prop
    generalize (Nat.chineseRemainder hcop 0 1 : ℕ) = x at hxlt hx0 hx1
    refine ⟨(x : ZMod (d * k)), ?_, ?_⟩
    · simp only [J, Finset.mem_filter, Finset.mem_univ, true_and, ZMod.val_cast_of_lt hxlt]
      have hdx : d ∣ x := Nat.modEq_zero_iff_dvd.mp hx0
      have hxk : Nat.Coprime x k := by
        have := hx1.gcd_eq; rwa [Nat.gcd_one_left] at this
      rw [Nat.Coprime.gcd_mul x hcop, Nat.gcd_eq_right hdx, hxk.gcd_eq_one, mul_one]
    · rw [← Nat.cast_mul, ZMod.natCast_eq_natCast_iff]
      refine (Nat.modEq_and_modEq_iff_modEq_mul hcop).mp ⟨?_, ?_⟩
      · exact (hx0.mul hx0).trans (by simpa using hx0.symm)
      · exact (hx1.mul hx1).trans (by simpa using hx1.symm)

omit [NeZero n] in
/-- A divisor of a square-free `n` is coprime to its cofactor. -/
private lemma coprime_cofactor (h : Squarefree n) {d : ℕ} (hd : d ∣ n) : Nat.Coprime d (n / d) :=
  Nat.coprime_of_squarefree_mul (by rwa [Nat.mul_div_cancel' hd])

/-- B1, Chen et al. Prop. D.13 (`01-setup.tex:76–77`): square-free `n` ⇒ every `J_d` is
regular. -/
theorem B1_squarefree_regular (h : Squarefree n) {d : ℕ} (hd : d ∣ n) : Regular n d :=
  (B2_regular_iff hd).mpr (coprime_cofactor h hd)

/-- B2, `tab:moduli` caption: `n` not square-free ⇒ some `J_d` is non-regular. -/
theorem B2_not_squarefree (h : ¬ Squarefree n) : ∃ d, d ∣ n ∧ ¬ Regular n d := by
  rw [Nat.squarefree_iff_prime_squarefree] at h
  push Not at h
  obtain ⟨p, hp, hpp⟩ := h
  refine ⟨p, dvd_trans (dvd_mul_right p p) hpp, fun hr => ?_⟩
  have hc := (B2_regular_iff (dvd_trans (dvd_mul_right p p) hpp)).mp hr
  exact (Nat.Prime.coprime_iff_not_dvd hp).mp hc (Nat.dvd_div_of_mul_dvd hpp)

/-- `J_n = {0}`. -/
private lemma J_self : J n n = {0} := by
  ext x
  simp only [J, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
  constructor
  · intro h
    have hdx : n ∣ x.val := by have := Nat.gcd_dvd_left x.val n; rwa [h] at this
    exact (ZMod.val_eq_zero x).mp (Nat.eq_zero_of_dvd_of_lt hdx (ZMod.val_lt x))
  · rintro rfl
    rw [ZMod.val_zero, Nat.gcd_zero_left]

/-- B3a, `eq:descent`, square-free case: `J_d · J_d = J_d`. -/
theorem B3_descent_squarefree (h : Squarefree n) {d : ℕ} (hd : d ∣ n) :
    J n d * J n d = J n d := by
  rw [A5_jcompose hd hd]
  have hcop := coprime_cofactor h hd
  congr 1
  conv_lhs => rw [← Nat.mul_div_cancel' hd]
  rw [Nat.gcd_mul_left, hcop.gcd_eq_one, mul_one]

/-- B3b, `eq:descent`: `J₁₁ · J₁₁ = {0}` at `n = 121`. -/
theorem B3_descent_121 : J 121 11 * J 121 11 = {0} := by
  rw [A5_jcompose (by norm_num) (by norm_num), show Nat.gcd (11 * 11) 121 = 121 by norm_num,
    J_self]

/-- B3c, `eq:descent`: `J₅ · J₅ = J₂₅` and `J₂₅ · J₂₅ = {0}` at `n = 125`. -/
theorem B3_descent_125 : J 125 5 * J 125 5 = J 125 25 ∧ J 125 25 * J 125 25 = {0} := by
  refine ⟨?_, ?_⟩
  · rw [A5_jcompose (by norm_num) (by norm_num), show Nat.gcd (5 * 5) 125 = 25 by norm_num]
  · rw [A5_jcompose (by norm_num) (by norm_num), show Nat.gcd (25 * 25) 125 = 125 by norm_num,
      J_self]

/-- B3d, `01-setup.tex:98–99`, the abstract (`2^7`, depth 7) and `06-results-clock.tex:148`
(depths 2 to 4): `ν(p^a) = a`. Gives `ν(121) = 2`, `ν(125) = 3`, `ν(128) = 7`, `ν(81) = 4`. -/
theorem B3_nilDepth_prime_pow {p a : ℕ} (hp : p.Prime) (ha : 1 ≤ a) :
    nilDepth (p ^ a) = a := by
  have : NeZero (p ^ a) := ⟨pow_ne_zero a hp.ne_zero⟩
  -- every nilpotent is a multiple of `p`, so its `a`-th power vanishes
  have hmem : a ∈ {k | 1 ≤ k ∧ ∀ x : ZMod (p ^ a), IsNilpotent x → x ^ k = 0} := by
    refine ⟨ha, fun x ⟨m, hm⟩ => ?_⟩
    rw [← ZMod.natCast_zmod_val x, ← Nat.cast_pow] at hm ⊢
    rw [ZMod.natCast_eq_zero_iff] at hm ⊢
    have hpx : p ∣ x.val := hp.dvd_of_dvd_pow (dvd_trans (dvd_pow_self p (by omega)) hm)
    exact pow_dvd_pow_of_dvd hpx a
  -- `p` itself is nilpotent, and `p ^ k ≠ 0` for `k < a`
  refine le_antisymm (Nat.sInf_le hmem) (le_csInf ⟨a, hmem⟩ fun k ⟨hk1, hk⟩ => ?_)
  by_contra hlt
  push Not at hlt
  have h0 := hk (p : ZMod (p ^ a)) ⟨a, by rw [← Nat.cast_pow, ZMod.natCast_self]⟩
  rw [← Nat.cast_pow, ZMod.natCast_eq_zero_iff] at h0
  exact absurd (Nat.pow_dvd_pow_iff_le_right hp.one_lt |>.mp h0) (by omega)

/-- `q = n / gcd(de, n)` divides `n / d`, and `n / d ≠ 0`. -/
private lemma qOf_dvd {d e : ℕ} (hd : d ∣ n) : qOf n d e ∣ n / d ∧ n / d ≠ 0 := by
  have hn0 : 0 < n := Nat.pos_of_ne_zero (NeZero.ne n)
  have hd0 : 0 < d := Nat.pos_of_dvd_of_pos hd hn0
  refine ⟨?_, (Nat.div_pos (Nat.le_of_dvd hn0 hd) hd0).ne'⟩
  unfold qOf
  set m := mOf n d e
  have hmpos : 0 < m := Nat.gcd_pos_of_pos_right _ hn0
  obtain ⟨b, hb⟩ : m ∣ n := Nat.gcd_dvd_right _ _
  obtain ⟨a, ha⟩ : d ∣ m := Nat.dvd_gcd (dvd_mul_right d e) hd
  rw [hb, Nat.mul_div_cancel_left b hmpos, ha, mul_assoc, Nat.mul_div_cancel_left _ hd0]
  exact dvd_mul_left b a

/-- B4a, `eq:fibration`: reduction `(ℤ/(n/d))ˣ → G_m = (ℤ/q)ˣ` is surjective. -/
theorem B4_fibration_surjective {d e : ℕ} (hd : d ∣ n) (he : e ∣ n) :
    ∃ h : qOf n d e ∣ n / d, Function.Surjective (ZMod.unitsMap h) := by
  obtain ⟨hq, hnd⟩ := qOf_dvd (e := e) hd
  have : NeZero (n / d) := ⟨hnd⟩
  exact ⟨hq, ZMod.unitsMap_surjective hq⟩

/-- B4b, `eq:fibration`: the target depends on `(u, v)` only through their residues mod `q`. -/
theorem B4_fibration_factors {d e u u' v v' : ℕ} (hd : d ∣ n) (he : e ∣ n)
    (huu : u % qOf n d e = u' % qOf n d e) (hvv : v % qOf n d e = v' % qOf n d e) :
    (d * u) * (e * v) % n = (d * u') * (e * v') % n := by
  have hmn : mOf n d e ∣ n := Nat.gcd_dvd_right _ _
  have hmde : mOf n d e ∣ d * e := Nat.gcd_dvd_left _ _
  have hn : n = mOf n d e * qOf n d e := (Nat.mul_div_cancel' hmn).symm
  -- `(d u)(e v) mod n = m (w u v mod q)`, and the right side sees `u, v` only mod `q`
  have key : ∀ s t, (d * s) * (e * t) % n = mOf n d e * (wOf n d e * (s * t) % qOf n d e) :=
    fun s t => by
      have h1 : (d * s) * (e * t) = mOf n d e * (wOf n d e * (s * t)) := by
        unfold wOf; rw [← Nat.mul_assoc (mOf n d e), Nat.mul_div_cancel' hmde]; ring
      set m := mOf n d e
      set q := qOf n d e
      rw [h1, hn]
      exact Nat.mul_mod_mul_left _ _ _
  rw [key, key]
  congr 1
  exact Nat.ModEq.mul_left _ (Nat.ModEq.mul huu hvv)

/-- B4c, `01-setup.tex:236–237`: a stratum has `φ(n/d) φ(n/e) ≥ |G_m| = φ(q)` cells. -/
theorem B4_cells_ge {d e : ℕ} (hd : d ∣ n) (he : e ∣ n) :
    (qOf n d e).totient ≤ (n / d).totient * (n / e).totient := by
  obtain ⟨hq, hnd⟩ := qOf_dvd (e := e) hd
  have hne : 0 < (n / e).totient := Nat.totient_pos.mpr (Nat.div_pos (Nat.le_of_dvd
    (Nat.pos_of_ne_zero (NeZero.ne n)) he) (Nat.pos_of_dvd_of_pos he (Nat.pos_of_ne_zero (NeZero.ne n))))
  calc (qOf n d e).totient ≤ (n / d).totient :=
        Nat.le_of_dvd (Nat.totient_pos.mpr (Nat.pos_of_ne_zero hnd)) (Nat.totient_dvd_of_dvd hq)
    _ ≤ (n / d).totient * (n / e).totient := Nat.le_mul_of_pos_right _ hne

/-- B5, consequence 3 (`eq:diagonal`): for a finite abelian group `G` and `f : G → ℂ`, the
2-D character transform of `(a, b) ↦ f(ab)` vanishes off the diagonal `χ = ψ`. -/
theorem B5_diagonal {G : Type*} [CommGroup G] [Fintype G] (f : G → ℂ)
    (χ ψ : G →* ℂˣ) (hne : χ ≠ ψ) :
    ∑ a, ∑ b, (((χ a)⁻¹ * (ψ b)⁻¹ : ℂˣ) : ℂ) * f (a * b) = 0 := by
  -- substitute `b = a⁻¹ c`: the sum factors through `Σ_a (ψ χ⁻¹)(a)`, which vanishes
  have hsub : ∀ a, ∑ b, (((χ a)⁻¹ * (ψ b)⁻¹ : ℂˣ) : ℂ) * f (a * b) =
      ((ψ a / χ a : ℂˣ) : ℂ) * ∑ c, (((ψ c)⁻¹ : ℂˣ) : ℂ) * f c := by
    intro a
    rw [Finset.mul_sum]
    refine Fintype.sum_equiv (Equiv.mulLeft a) _ _ fun b => ?_
    simp only [Equiv.coe_mulLeft, map_mul, mul_inv_rev, Units.val_mul, Units.val_div_eq_div_val]
    rw [div_eq_mul_inv, ← Units.val_inv_eq_inv_val]
    have : (((ψ a)⁻¹ : ℂˣ) : ℂ) * (ψ a : ℂ) = 1 := by simp
    linear_combination (-(((χ a)⁻¹ : ℂˣ) : ℂ) * (((ψ b)⁻¹ : ℂˣ) : ℂ) * f (a * b)) * this
  rw [Finset.sum_congr rfl fun a _ => hsub a, ← Finset.sum_mul]
  have hne1 : (Units.coeHom ℂ).comp (ψ / χ) ≠ 1 := by
    intro h1
    apply hne
    ext a
    have := congrArg (fun F => F a) h1
    simp only [MonoidHom.coe_comp, Function.comp_apply, MonoidHom.div_apply, Units.coeHom_apply,
      MonoidHom.one_apply, Units.val_eq_one, div_eq_one] at this
    exact congrArg Units.val this.symm
  have := sum_hom_units_eq_zero _ hne1
  simp only [MonoidHom.coe_comp, Function.comp_apply, MonoidHom.div_apply,
    Units.coeHom_apply] at this
  rw [this, zero_mul]

/-- B6, `eq:prodchar`, `eq:diagonal`: `|Ĝ| = |G|`, hence `|Δ| = |G_m|`. -/
theorem B6_card_dual {G : Type*} [CommGroup G] [Fintype G] :
    Nat.card (G →* ℂˣ) = Fintype.card G := by
  rw [CommGroup.card_monoidHom_of_hasEnoughRootsOfUnity G ℂ, Nat.card_eq_fintype_card]

end Strata
