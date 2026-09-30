import Strata.Basic

/-!
# Tier A — the stratum algebra, `paper/sections/01-setup.tex`
-/

open Finset Pointwise

namespace Strata

variable {n : ℕ} [NeZero n]

/-- A0, `eq:units`: `|(ℤ/nℤ)ˣ| = φ(n)`. -/
theorem A0_card_units : Fintype.card (ZMod n)ˣ = n.totient :=
  ZMod.card_units_eq_totient n

/-- `x ∈ J_d` unfolds to `gcd(x, n) = d`. -/
private lemma mem_J {d : ℕ} {x : ZMod n} : x ∈ J n d ↔ Nat.gcd x.val n = d := by
  simp [J]

/-- A2, `prop:torsor`, `eq:torsor`: `u ↦ d·u mod n` is a bijection `(ℤ/(n/d))ˣ → J_d`,
whether or not `J_d` is regular. -/
theorem A2_torsor {d : ℕ} (hd : d ∣ n) :
    Set.BijOn (fun u : (ZMod (n / d))ˣ => ((d * (u : ZMod (n / d)).val : ℕ) : ZMod n))
      Set.univ (J n d : Set (ZMod n)) := by
  obtain ⟨k, rfl⟩ := hd
  have hn : d * k ≠ 0 := NeZero.ne _
  have hd0 : 0 < d := Nat.pos_of_ne_zero (left_ne_zero_of_mul hn)
  have hk0 : k ≠ 0 := right_ne_zero_of_mul hn
  rw [Nat.mul_div_cancel_left k hd0]
  have : NeZero k := ⟨hk0⟩
  -- `d * u.val < d * k`, so the cast is exact
  have hlt : ∀ u : (ZMod k)ˣ, d * (u : ZMod k).val < d * k := fun u =>
    Nat.mul_lt_mul_of_pos_left (ZMod.val_lt _) hd0
  refine ⟨fun u _ => ?_, fun u _ u' _ h => ?_, fun x hx => ?_⟩
  · rw [Finset.mem_coe, mem_J, ZMod.val_cast_of_lt (hlt u), Nat.gcd_mul_left,
      ZMod.val_coe_unit_coprime u, mul_one]
  · have h' := congrArg ZMod.val h
    simp only [ZMod.val_cast_of_lt (hlt u), ZMod.val_cast_of_lt (hlt u')] at h'
    exact Units.ext (ZMod.val_injective k (Nat.eq_of_mul_eq_mul_left hd0 h'))
  · rw [Finset.mem_coe, mem_J] at hx
    have hdx : d ∣ x.val := by
      have := Nat.gcd_dvd_left x.val (d * k); rwa [hx] at this
    obtain ⟨a, ha⟩ := hdx
    have hcop : Nat.Coprime a k := by
      have := hx
      rw [ha, Nat.gcd_mul_left] at this
      exact (Nat.mul_eq_left hd0.ne').mp this
    have hak : a < k := by
      have := ZMod.val_lt x
      rw [ha] at this
      exact Nat.lt_of_mul_lt_mul_left this
    refine ⟨ZMod.unitOfCoprime a hcop, Set.mem_univ _, ?_⟩
    simp only [ZMod.coe_unitOfCoprime, ZMod.val_cast_of_lt hak]
    rw [← ha, ZMod.natCast_zmod_val]

/-- A1a, `eq:jclass`: `|J_d| = φ(n/d)`. -/
theorem A1_card_J {d : ℕ} (hd : d ∣ n) : (J n d).card = (n / d).totient := by
  have hn : n / d ≠ 0 := by
    obtain ⟨k, rfl⟩ := hd
    have hn : d * k ≠ 0 := NeZero.ne _
    rw [Nat.mul_div_cancel_left k (Nat.pos_of_ne_zero (left_ne_zero_of_mul hn))]
    exact right_ne_zero_of_mul hn
  have : NeZero (n / d) := ⟨hn⟩
  have h := A2_torsor hd
  rw [← Finset.coe_univ] at h
  rw [← ZMod.card_units_eq_totient, ← Finset.card_univ]
  exact (h.finsetCard_eq _).symm

/-- A1b, `eq:jclass`: `ℤ/nℤ = ⊔_{d ∣ n} J_d` — every residue lies in exactly one class. -/
theorem A1_partition (x : ZMod n) : ∃! d, d ∣ n ∧ x ∈ J n d :=
  ⟨Nat.gcd x.val n, ⟨Nat.gcd_dvd_right _ _, mem_J.mpr rfl⟩,
    fun _ ⟨_, h⟩ => (mem_J.mp h).symm⟩

/-- A3, `eq:mscale`: `(m t) mod (m q) = m (t mod q)`, for any integer `t`. -/
theorem A3_mscale (m q t : ℤ) (hm : 0 < m) : (m * t) % (m * q) = m * (t % q) :=
  Int.mul_emod_mul_of_pos t q hm

/-- A4, `thm:stratum`, `eq:stratum`: with `x = d u`, `y = e v` as in `eq:torsor` and
`m = gcd(de, n)`, `de = m w`, `q = n / m` (`eq:mw`): `d ∣ m`, `e ∣ m`, `gcd(w, q) = 1`,
`uv` is a unit modulo `q`, and `x y mod n = m (w (u v) mod q)`. -/
theorem A4_stratum {d e u v : ℕ} (hd : d ∣ n) (he : e ∣ n)
    (hu : Nat.Coprime u (n / d)) (hv : Nat.Coprime v (n / e)) :
    d ∣ mOf n d e ∧ e ∣ mOf n d e ∧
    Nat.Coprime (wOf n d e) (qOf n d e) ∧ Nat.Coprime (u * v) (qOf n d e) ∧
    (d * u) * (e * v) % n = mOf n d e * (wOf n d e * (u * v) % qOf n d e) := by
  unfold wOf qOf
  set m := mOf n d e with hm
  have hmpos : 0 < m := Nat.gcd_pos_of_pos_right _ (Nat.pos_of_ne_zero (NeZero.ne n))
  have hmde : m ∣ d * e := Nat.gcd_dvd_left _ _
  have hmn : m ∣ n := Nat.gcd_dvd_right _ _
  have hdm : d ∣ m := Nat.dvd_gcd (dvd_mul_right d e) hd
  have hem : e ∣ m := Nat.dvd_gcd (dvd_mul_left e d) he
  -- `q = n / m` divides `n / d` and `n / e`, since `d ∣ m ∣ n`
  have hq : ∀ c, c ∣ m → n / m ∣ n / c := fun c hc => by
    obtain ⟨a, ha⟩ := hc
    obtain ⟨b, hb⟩ := hmn
    have hc0 : 0 < c := Nat.pos_of_ne_zero fun h => by simp [h] at ha; omega
    rw [hb, Nat.mul_div_cancel_left b hmpos, ha, mul_assoc, Nat.mul_div_cancel_left _ hc0]
    exact dvd_mul_left b a
  refine ⟨hdm, hem, Nat.coprime_div_gcd_div_gcd hmpos,
    Nat.coprime_mul_iff_left.mpr
      ⟨hu.coprime_dvd_right (hq d hdm), hv.coprime_dvd_right (hq e hem)⟩, ?_⟩
  have h1 : (d * u) * (e * v) = m * (d * e / m * (u * v)) := by
    rw [← Nat.mul_assoc m, Nat.mul_div_cancel' hmde]; ring
  conv_lhs => rw [h1, ← Nat.mul_div_cancel' hmn]
  exact Nat.mul_mod_mul_left _ _ _

/-- A5, `eq:jcompose`: `J_d · J_e = J_{gcd(de, n)}` — containment and equality. -/
theorem A5_jcompose {d e : ℕ} (hd : d ∣ n) (he : e ∣ n) :
    J n d * J n e = J n (Nat.gcd (d * e) n) := by
  set m := Nat.gcd (d * e) n with hm
  have hmpos : 0 < m := Nat.gcd_pos_of_pos_right _ (Nat.pos_of_ne_zero (NeZero.ne n))
  have hmn : m ∣ n := Nat.gcd_dvd_right _ _
  -- a residue `m * r` with `r` coprime to `q = n / m` lies in `J_m`
  have key : ∀ r, Nat.Coprime r (n / m) → Nat.gcd (m * r) n = m := fun r hr => by
    conv_lhs => rw [← Nat.mul_div_cancel' hmn]
    rw [Nat.gcd_mul_left, hr.gcd_eq_one, mul_one]
  ext z
  rw [Finset.mem_mul]
  constructor
  · -- containment: `thm:stratum`
    rintro ⟨x, hx, y, hy, rfl⟩
    obtain ⟨U, -, rfl⟩ := (A2_torsor hd).surjOn (Finset.mem_coe.mpr hx)
    obtain ⟨V, -, rfl⟩ := (A2_torsor he).surjOn (Finset.mem_coe.mpr hy)
    obtain ⟨-, -, hw, huv, hid⟩ := A4_stratum (n := n) hd he
      (ZMod.val_coe_unit_coprime U) (ZMod.val_coe_unit_coprime V)
    have hq : qOf n d e = n / m := rfl
    rw [hq] at hw huv hid
    rw [mem_J, ← Nat.cast_mul, ZMod.val_natCast, hid]
    refine key _ ?_
    rw [Nat.Coprime, ← Nat.gcd_rec]
    exact (Nat.coprime_mul_iff_left.mpr ⟨hw, huv⟩).symm
  · -- equality: `y = e` and `x = d u`, with `u` lifted from `(ℤ/q)ˣ` to `(ℤ/(n/d))ˣ`
    intro hz
    rw [mem_J] at hz
    set q := n / m with hq
    have hmq : m * q = n := Nat.mul_div_cancel' hmn
    have hq0 : q ≠ 0 := fun h => NeZero.ne n (by rw [← hmq, h, mul_zero])
    -- `z = m s` with `s` a unit modulo `q`
    have hmz : m ∣ z.val := by have := Nat.gcd_dvd_left z.val n; rwa [hz] at this
    obtain ⟨s, hs⟩ := hmz
    have hsq : s < q := by
      have := ZMod.val_lt z
      rw [hs] at this
      exact Nat.lt_of_mul_lt_mul_left (hmq ▸ this)
    have hscop : Nat.Coprime s q := by
      have := hz
      rw [hs, ← hmq, Nat.gcd_mul_left] at this
      exact (Nat.mul_eq_left hmpos.ne').mp this
    -- `w = de / m` is a unit modulo `q`
    have hwcop : Nat.Coprime (d * e / m) q := Nat.coprime_div_gcd_div_gcd hmpos
    -- `q ∣ n / d`, since `d ∣ m`
    have hdm : d ∣ m := Nat.dvd_gcd (dvd_mul_right d e) hd
    obtain ⟨a, ha⟩ := hdm
    have hd0 : 0 < d := Nat.pos_of_ne_zero fun h => by simp [h] at ha; omega
    have hnd : n / d = a * q := by
      rw [← hmq, ha, mul_assoc, Nat.mul_div_cancel_left _ hd0]
    have hqd : q ∣ n / d := hnd ▸ dvd_mul_left q a
    have : NeZero (n / d) := ⟨fun h => NeZero.ne n (by
      rw [← Nat.div_mul_cancel hd, h, zero_mul])⟩
    have : NeZero q := ⟨hq0⟩
    -- lift `t = w⁻¹ s` from `(ℤ/q)ˣ` to `(ℤ/(n/d))ˣ` (`eq:fibration`)
    obtain ⟨U, hU⟩ := ZMod.unitsMap_surjective hqd
      ((ZMod.unitOfCoprime _ hwcop)⁻¹ * ZMod.unitOfCoprime s hscop)
    have hut : (((U : ZMod (n / d)).val : ℕ) : ZMod q) =
        (((ZMod.unitOfCoprime _ hwcop)⁻¹ * ZMod.unitOfCoprime s hscop : (ZMod q)ˣ) : ZMod q) := by
      rw [ZMod.natCast_val, ← hU, ZMod.unitsMap_val]
    have hwu : (((d * e / m) * (U : ZMod (n / d)).val : ℕ) : ZMod q) = s := by
      rw [Nat.cast_mul, hut, ← ZMod.coe_unitOfCoprime _ hwcop, ← Units.val_mul, ← mul_assoc,
        mul_inv_cancel, one_mul, ZMod.coe_unitOfCoprime]
    refine ⟨_, (A2_torsor hd).mapsTo (Set.mem_univ U), (e : ZMod n), ?_, ?_⟩
    · rw [mem_J, ZMod.val_natCast, ← Nat.gcd_rec, Nat.gcd_eq_right he]
    · rw [← Nat.cast_mul, ← ZMod.natCast_zmod_val z, ZMod.natCast_eq_natCast_iff',
        Nat.mod_eq_of_lt (ZMod.val_lt z), hs]
      generalize (U : ZMod (n / d)).val = u at hwu ⊢
      have h1 : d * u * e = m * ((d * e / m) * u) := by
        rw [← mul_assoc, Nat.mul_div_cancel' (Nat.gcd_dvd_left _ _)]; ring
      rw [h1, ← hmq, Nat.mul_mod_mul_left, (ZMod.natCast_eq_natCast_iff' _ _ _).mp hwu,
        Nat.mod_eq_of_lt hsq]

/-- A6, `01-setup.tex:79–80`: a **nonzero** nilpotent exists in `ℤ/nℤ` iff `n` is not
square-free. The paper's parenthetical omits "nonzero"; `0` is nilpotent in every ring. -/
theorem A6_nilpotent_iff : (∃ x : ZMod n, IsNilpotent x ∧ x ≠ 0) ↔ ¬ Squarefree n := by
  have h := (isReduced_zmod (n := n))
  simp only [NeZero.ne n, or_false] at h
  rw [← h]
  constructor
  · rintro ⟨x, hx, hx0⟩ hr
    exact hx0 (hr.eq_zero x hx)
  · intro hr
    by_contra hc
    exact hr ⟨fun x hx => by_contra fun h0 => hc ⟨x, hx, h0⟩⟩

end Strata
