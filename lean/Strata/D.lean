import Strata.A

/-!
# Tier D — the indicators' spectra, `paper/sections/09-results-crt.tex`
-/

open Finset

namespace Strata

/-- `exp(2πi m / q) = ψ_q(m)`, `ψ_q` the standard additive character of `ℤ/qℤ`. -/
private lemma exp_eq_std {q : ℕ} [NeZero q] (m : ℤ) :
    Complex.exp (2 * Real.pi * Complex.I * m / q) = ZMod.stdAddChar (m : ZMod q) :=
  (ZMod.stdAddChar_coe m).symm

/-- `exp_eq_std` for a natural `m`. -/
private lemma exp_eq_std_nat {q : ℕ} [NeZero q] (m : ℕ) :
    Complex.exp (2 * Real.pi * Complex.I * m / q) = ZMod.stdAddChar (m : ZMod q) := by
  simpa using exp_eq_std (q := q) (m : ℤ)

/-- `c_q(k) = Σ_{u ∈ (ℤ/qℤ)ˣ} ψ_q(u k)`. -/
private lemma ramanujanSum_units {q : ℕ} [NeZero q] (k : ℕ) :
    ramanujanSum q k = ∑ u : (ZMod q)ˣ, ZMod.stdAddChar ((u : ZMod q) * (k : ZMod q)) := by
  unfold ramanujanSum
  refine Finset.sum_bij' (fun a ha => ZMod.unitOfCoprime a (Finset.mem_filter.mp ha).2)
    (fun u _ => (u : ZMod q).val) (fun _ _ => mem_univ _) (fun u _ => ?_) (fun a ha => ?_)
    (fun u _ => ?_) (fun a ha => ?_)
  · exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (ZMod.val_lt _), ZMod.val_coe_unit_coprime u⟩
  · simp only [ZMod.coe_unitOfCoprime]
    exact ZMod.val_cast_of_lt (Finset.mem_range.mp (Finset.mem_filter.mp ha).1)
  · exact Units.ext (by simp)
  · rw [ZMod.coe_unitOfCoprime, ← Nat.cast_mul, exp_eq_std_nat]

/-- `exp(-2πi m / q) = ψ_q(-m)`. -/
private lemma exp_neg_eq_std {q : ℕ} [NeZero q] (m : ℕ) :
    Complex.exp (-2 * Real.pi * Complex.I * m / q) = ZMod.stdAddChar (-(m : ZMod q)) := by
  have := exp_eq_std (q := q) (-(m : ℤ))
  push_cast at this
  rw [← this]; congr 1; ring

/-- D0, `09-results-crt.tex:32–33`: for `d ∣ n`, the indicator of the multiples of `d` has
additive Fourier support exactly on the multiples of `n/d`. -/
theorem D0_multiples_support {n d k : ℕ} [NeZero n] (hd : d ∣ n) :
    dft n (univ.filter fun x : ZMod n => d ∣ x.val) k ≠ 0 ↔ n / d ∣ k := by
  obtain ⟨q, rfl⟩ := hd
  have hn : d * q ≠ 0 := NeZero.ne _
  have hd0 : 0 < d := Nat.pos_of_ne_zero (left_ne_zero_of_mul hn)
  have hq0 : q ≠ 0 := right_ne_zero_of_mul hn
  rw [Nat.mul_div_cancel_left q hd0]
  have : NeZero q := ⟨hq0⟩
  have hlt : ∀ j : ZMod q, d * j.val < d * q := fun j =>
    Nat.mul_lt_mul_of_pos_left (ZMod.val_lt _) hd0
  -- the multiples of `d` are `j ↦ d j`, `j ∈ ℤ/qℤ`
  have hsum : dft (d * q) (univ.filter fun x : ZMod (d * q) => d ∣ x.val) k =
      ∑ j : ZMod q, (ZMod.stdAddChar.mulShift (-(k : ZMod q))) j := by
    unfold dft
    symm
    refine Finset.sum_nbij (fun j : ZMod q => ((d * j.val : ℕ) : ZMod (d * q)))
      (fun j _ => ?_) (fun j _ j' _ h => ?_) (fun x hx => ?_) (fun j _ => ?_)
    · exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, by
        rw [ZMod.val_cast_of_lt (hlt j)]; exact dvd_mul_right d _⟩
    · have h' := congrArg ZMod.val h
      simp only [ZMod.val_cast_of_lt (hlt j), ZMod.val_cast_of_lt (hlt j')] at h'
      exact ZMod.val_injective q (Nat.eq_of_mul_eq_mul_left hd0 h')
    · obtain ⟨a, ha⟩ := (Finset.mem_filter.mp (Finset.mem_coe.mp hx)).2
      have hak : a < q := by
        have := ZMod.val_lt x
        rw [ha] at this
        exact Nat.lt_of_mul_lt_mul_left this
      refine ⟨(a : ZMod q), Finset.mem_coe.mpr (Finset.mem_univ _), ?_⟩
      simp only [ZMod.val_cast_of_lt hak]
      rw [← ha, ZMod.natCast_zmod_val]
    · rw [AddChar.mulShift_apply, ZMod.val_cast_of_lt (hlt j),
        show -(k : ZMod q) * j = -(((j.val * k : ℕ)) : ZMod q) by
          push_cast; rw [ZMod.natCast_zmod_val]; ring,
        ← exp_neg_eq_std]
      congr 1
      have hdC : (d : ℂ) ≠ 0 := by exact_mod_cast hd0.ne'
      push_cast
      field_simp
  rw [hsum, AddChar.sum_ne_zero_iff_eq_zero, ← ZMod.natCast_eq_zero_iff]
  constructor
  · intro h
    by_contra hk
    exact ZMod.isPrimitive_stdAddChar q (neg_ne_zero.mpr hk) h
  · intro h
    rw [h, neg_zero]
    ext x
    simp

/-- D1, `09-results-crt.tex:115`: the transform of `𝟙_{J_d}` at `k` is the Ramanujan sum
`c_{n/d}(k)`. -/
theorem D1_indicator_ramanujan {n d k : ℕ} [NeZero n] (hd : d ∣ n) :
    dft n (J n d) k = ramanujanSum (n / d) k := by
  obtain ⟨q, rfl⟩ := hd
  have hn : d * q ≠ 0 := NeZero.ne _
  have hd0 : d ≠ 0 := left_ne_zero_of_mul hn
  have hq0 : q ≠ 0 := right_ne_zero_of_mul hn
  have hbij := A2_torsor (n := d * q) (dvd_mul_right d q)
  rw [Nat.mul_div_cancel_left q (Nat.pos_of_ne_zero hd0)] at hbij ⊢
  have : NeZero q := ⟨hq0⟩
  rw [ramanujanSum_units]
  unfold dft
  rw [← Finset.coe_univ] at hbij
  rw [← Finset.sum_nbij _ (fun a ha => hbij.mapsTo ha) hbij.injOn hbij.surjOn
    (f := fun u : (ZMod q)ˣ => ZMod.stdAddChar (-((u : ZMod q) * (k : ZMod q))))
    (fun u _ => ?_)]
  · rw [← Fintype.sum_equiv (Equiv.neg (ZMod q)ˣ) _ _ (fun u => rfl)]
    exact Fintype.sum_congr _ _ fun u => by simp [Units.val_neg, neg_mul]
  · have hlt : d * (u : ZMod q).val < d * q :=
      Nat.mul_lt_mul_of_pos_left (ZMod.val_lt _) (Nat.pos_of_ne_zero hd0)
    simp only [ZMod.val_cast_of_lt hlt]
    rw [show ((u : ZMod q) * (k : ZMod q)) = (((u : ZMod q).val * k : ℕ) : ZMod q) by
      push_cast; rw [ZMod.natCast_zmod_val], ← exp_neg_eq_std]
    congr 1
    push_cast
    field_simp

/-- `Σ_{e ∣ r} c_e(1) = [r = 1]`: sum `ψ_r` over `ℤ/rℤ`, split by `J`-class. -/
private lemma sum_divisors_ramanujanSum_one {r : ℕ} (hr : 0 < r) :
    ∑ e ∈ r.divisors, ramanujanSum e 1 = if r = 1 then 1 else 0 := by
  have : NeZero r := ⟨hr.ne'⟩
  rw [← Nat.sum_div_divisors]
  rw [Finset.sum_congr rfl fun d hd => (D1_indicator_ramanujan (Nat.dvd_of_mem_divisors hd)).symm]
  unfold dft J
  rw [Finset.sum_fiberwise_of_maps_to (s := univ) (g := fun x : ZMod r => Nat.gcd x.val r)
    (fun x _ => Nat.mem_divisors.mpr ⟨Nat.gcd_dvd_right _ _, hr.ne'⟩)]
  have h1 : ∀ x : ZMod r, Complex.exp (-2 * Real.pi * Complex.I * ((x.val * 1 : ℕ) : ℂ) / r) =
      ZMod.stdAddChar (-x) := fun x => by
    rw [exp_neg_eq_std, mul_one, ZMod.natCast_zmod_val]
  simp only [h1]
  rw [Fintype.sum_equiv (Equiv.neg (ZMod r)) (fun x => ZMod.stdAddChar (-x))
    (fun x => ZMod.stdAddChar x) (fun x => rfl),
    AddChar.sum_eq_ite]
  by_cases h : r = 1
  · subst h
    have hψ : (ZMod.stdAddChar : AddChar (ZMod 1) ℂ) = 0 := by
      ext x
      rw [Subsingleton.elim x 0, AddChar.map_zero_eq_one, AddChar.zero_apply]
    simp [hψ]
  · have hψ : (ZMod.stdAddChar : AddChar (ZMod r) ℂ) ≠ 0 := by
      intro h0
      have h01 : ZMod.stdAddChar (1 : ZMod r) = ZMod.stdAddChar (0 : ZMod r) := by
        rw [h0, AddChar.zero_apply, AddChar.map_zero_eq_one]
      exact h (ZMod.one_eq_zero_iff.mp (ZMod.injective_stdAddChar h01))
    simp [h, hψ]

/-- `c_r(1) = μ(r)`: the sum of the primitive `r`-th roots of unity. -/
private lemma ramanujanSum_one {r : ℕ} (hr : 0 < r) :
    ramanujanSum r 1 = ArithmeticFunction.moebius r := by
  have := (ArithmeticFunction.sum_eq_iff_sum_mul_moebius_eq
    (f := fun e => ramanujanSum e 1) (g := fun r => if r = 1 then (1 : ℂ) else 0)).mp
    (fun r hr => sum_divisors_ramanujanSum_one hr) r hr
  rw [← this, Finset.sum_eq_single (r, 1)]
  · simp
  · rintro ⟨a, b⟩ hab hne
    rw [Nat.mem_divisorsAntidiagonal] at hab
    have hb : b ≠ 1 := fun hb => hne (by subst hb; simp_all)
    simp [hb]
  · intro h; exact absurd (Nat.mem_divisorsAntidiagonal.mpr ⟨mul_one r, hr.ne'⟩) h

/-- `c_r(k) = c_r(1)` when `k` is a unit modulo `r`. -/
private lemma ramanujanSum_coprime {r k : ℕ} [NeZero r] (hk : Nat.Coprime k r) :
    ramanujanSum r k = ramanujanSum r 1 := by
  rw [ramanujanSum_units, ramanujanSum_units]
  refine Fintype.sum_equiv (Equiv.mulRight (ZMod.unitOfCoprime k hk)) _ _ fun u => ?_
  simp

/-- Summing `F ∘ f` over `G`, for a surjective hom `f`, counts each value `|ker f|` times. -/
private lemma sum_comp_surjective {G H : Type*} [Group G] [Group H] [Fintype G] [Fintype H]
    [DecidableEq H] (f : G →* H) (hf : Function.Surjective f) (F : H → ℂ) :
    ∑ g, F (f g) = (#{g | f g = 1} : ℂ) * ∑ h, F h := by
  rw [← Finset.sum_fiberwise (s := univ) (g := f) (f := fun g => F (f g)), Finset.mul_sum]
  refine Finset.sum_congr rfl fun h _ => ?_
  rw [Finset.sum_congr rfl fun g hg => by rw [(Finset.mem_filter.mp hg).2], Finset.sum_const,
    nsmul_eq_mul, MonoidHom.card_fiber_eq_of_mem_range f (hf h) ⟨1, map_one f⟩]

/-- D2a, `09-results-crt.tex:116`: `c_q(k) = μ(q/g) φ(q) / φ(q/g)`, `g = gcd(k, q)`. -/
theorem D2_closed_form {q k : ℕ} (hq : 0 < q) :
    ramanujanSum q k =
      (ArithmeticFunction.moebius (q / Nat.gcd k q) : ℂ) * q.totient /
        (q / Nat.gcd k q).totient := by
  have : NeZero q := ⟨hq.ne'⟩
  set g := Nat.gcd k q with hg
  have hg0 : 0 < g := Nat.gcd_pos_of_pos_right _ hq
  set r := q / g with hr
  set k' := k / g with hk'
  have hqr : q = g * r := (Nat.mul_div_cancel' (Nat.gcd_dvd_right k q)).symm
  have hkk : k = g * k' := (Nat.mul_div_cancel' (Nat.gcd_dvd_left k q)).symm
  have hr0 : 0 < r := Nat.div_pos (Nat.le_of_dvd hq (Nat.gcd_dvd_right k q)) hg0
  have : NeZero r := ⟨hr0.ne'⟩
  have hcop : Nat.Coprime k' r := Nat.coprime_div_gcd_div_gcd hg0
  have hrq : r ∣ q := ⟨g, by rw [hqr, mul_comm]⟩
  -- each term of `c_q(k)` is a term of `c_r(k')`, read through reduction mod `r`
  have hterm : ∀ u : (ZMod q)ˣ, ZMod.stdAddChar ((u : ZMod q) * (k : ZMod q)) =
      ZMod.stdAddChar (((ZMod.unitsMap hrq u : (ZMod r)ˣ) : ZMod r) * (k' : ZMod r)) := by
    intro u
    have e1 : (u : ZMod q) * (k : ZMod q) = (((u : ZMod q).val * k : ℕ) : ZMod q) := by
      push_cast; rw [ZMod.natCast_zmod_val]
    have e2 : ((ZMod.unitsMap hrq u : (ZMod r)ˣ) : ZMod r) * (k' : ZMod r) =
        (((u : ZMod q).val * k' : ℕ) : ZMod r) := by
      rw [ZMod.unitsMap_val, ← ZMod.natCast_val]; push_cast; rfl
    rw [e1, e2, ← exp_eq_std_nat, ← exp_eq_std_nat]
    congr 1
    have hqC : (q : ℂ) = (g : ℂ) * r := by exact_mod_cast hqr
    have hkC : (k : ℂ) = (g : ℂ) * k' := by exact_mod_cast hkk
    have hgC : (g : ℂ) ≠ 0 := by exact_mod_cast hg0.ne'
    have hrC : (r : ℂ) ≠ 0 := by exact_mod_cast hr0.ne'
    push_cast
    rw [hqC, hkC]
    field_simp
  have hsum := sum_comp_surjective (ZMod.unitsMap hrq) (ZMod.unitsMap_surjective hrq)
    (fun v => ZMod.stdAddChar ((v : ZMod r) * (k' : ZMod r)))
  have hcard := sum_comp_surjective (ZMod.unitsMap hrq) (ZMod.unitsMap_surjective hrq)
    (fun _ => (1 : ℂ))
  simp only [Finset.sum_const, Finset.card_univ, ZMod.card_units_eq_totient, nsmul_eq_mul,
    mul_one] at hcard
  rw [ramanujanSum_units, Finset.sum_congr rfl fun u _ => hterm u, hsum,
    ← ramanujanSum_units, ramanujanSum_coprime hcop, ramanujanSum_one hr0, hcard]
  have hφ : (r.totient : ℂ) ≠ 0 := by exact_mod_cast (Nat.totient_pos.mpr hr0).ne'
  field_simp

/-- D2b, `09-results-crt.tex:117`: magnitude at most 1 where `k` is coprime to `q`. -/
theorem D2_coprime_small {q k : ℕ} (hq : 0 < q) (hk : Nat.Coprime k q) :
    ‖ramanujanSum q k‖ ≤ 1 := by
  rw [D2_closed_form hq, Nat.Coprime.gcd_eq_one hk, Nat.div_one]
  have hφ : (q.totient : ℂ) ≠ 0 := by exact_mod_cast (Nat.totient_pos.mpr hq).ne'
  rw [mul_div_cancel_right₀ _ hφ, Complex.norm_intCast]
  exact_mod_cast ArithmeticFunction.abs_moebius_le_one

/-- D2c, `09-results-crt.tex:117–118`: the value `φ(q)` where `k` shares all of `q`. -/
theorem D2_full {q k : ℕ} (hq : 0 < q) (hk : q ∣ k) : ramanujanSum q k = q.totient := by
  rw [D2_closed_form hq, Nat.gcd_eq_right hk, Nat.div_self hq]
  simp

/-- D2d, `09-results-crt.tex:119`: at square-free `q` a Ramanujan sum never vanishes — so at
`n = 165` every indicator's energy is nonzero at every frequency. -/
theorem D2_squarefree_ne_zero {q k : ℕ} (hq : Squarefree q) : ramanujanSum q k ≠ 0 := by
  have hq0 : 0 < q := Nat.pos_of_ne_zero hq.ne_zero
  have hdvd : q / Nat.gcd k q ∣ q := Nat.div_dvd_of_dvd (Nat.gcd_dvd_right k q)
  have hr0 : 0 < q / Nat.gcd k q :=
    Nat.div_pos (Nat.le_of_dvd hq0 (Nat.gcd_dvd_right k q)) (Nat.gcd_pos_of_pos_right _ hq0)
  rw [D2_closed_form hq0]
  have hμ : (ArithmeticFunction.moebius (q / Nat.gcd k q) : ℂ) ≠ 0 := by
    exact_mod_cast ArithmeticFunction.moebius_ne_zero_iff_squarefree.mpr
      (hq.squarefree_of_dvd hdvd)
  have hφ : (q.totient : ℂ) ≠ 0 := by exact_mod_cast (Nat.totient_pos.mpr hq0).ne'
  have hφ' : ((q / Nat.gcd k q).totient : ℂ) ≠ 0 := by
    exact_mod_cast (Nat.totient_pos.mpr hr0).ne'
  exact div_ne_zero (mul_ne_zero hμ hφ) hφ'

/-- D3, `09-results-crt.tex:120–122`: at `n = 165`, `42` of the `82` non-DC frequencies share a
factor with `n`. (The `30.5×` energy ratio beside it is computed, not proved: brute force.) -/
theorem D3_count_165 :
    ((Icc 1 (165 / 2)).filter fun k => Nat.gcd k 165 ≠ 1).card = 42 ∧ 165 / 2 = 82 := by
  decide

/-- D4, `09-results-crt.tex:35–37`: at `ω(n) = 1` the predicted set is every one of the
`⌊n/2⌋` bins, so a prime power can never count as support. -/
theorem D4_prime_power_vacuous {n : ℕ} (h : n.primeFactors.card = 1) :
    predicted n = Icc 1 (n / 2) := by
  obtain ⟨p, hp⟩ := Finset.card_eq_one.mp h
  have hn : n ≠ 0 := by rintro rfl; simp at hp
  have hn' : p ^ n.factorization p = n := by
    conv_rhs => rw [← Nat.prod_factorization_pow_eq_self hn]
    rw [Finsupp.prod, Nat.support_factorization, hp, Finset.prod_singleton]
  unfold predicted
  refine Finset.filter_true_of_mem fun k _ => ⟨p, by simp [hp], ?_⟩
  rw [hn', Nat.div_self (Nat.pos_of_ne_zero hn)]
  exact one_dvd k

end Strata
