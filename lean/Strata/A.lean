import Strata.Basic

/-!
# Tier A — the stratum algebra, `paper/sections/01-setup.tex`
-/

open Finset Pointwise

namespace Strata

variable {n : ℕ} [NeZero n]

/-- A0, `eq:units`: `|(ℤ/nℤ)ˣ| = φ(n)`. -/
theorem A0_card_units : Fintype.card (ZMod n)ˣ = n.totient := by
  sorry

/-- A1a, `eq:jclass`: `|J_d| = φ(n/d)`. -/
theorem A1_card_J {d : ℕ} (hd : d ∣ n) : (J n d).card = (n / d).totient := by
  sorry

/-- A1b, `eq:jclass`: `ℤ/nℤ = ⊔_{d ∣ n} J_d` — every residue lies in exactly one class. -/
theorem A1_partition (x : ZMod n) : ∃! d, d ∣ n ∧ x ∈ J n d := by
  sorry

/-- A2, `prop:torsor`, `eq:torsor`: `u ↦ d·u mod n` is a bijection `(ℤ/(n/d))ˣ → J_d`,
whether or not `J_d` is regular. -/
theorem A2_torsor {d : ℕ} (hd : d ∣ n) :
    Set.BijOn (fun u : (ZMod (n / d))ˣ => ((d * (u : ZMod (n / d)).val : ℕ) : ZMod n))
      Set.univ (J n d : Set (ZMod n)) := by
  sorry

/-- A3, `eq:mscale`: `(m t) mod (m q) = m (t mod q)`, for any integer `t`. -/
theorem A3_mscale (m q t : ℤ) (hm : 0 < m) : (m * t) % (m * q) = m * (t % q) := by
  sorry

/-- A4, `thm:stratum`, `eq:stratum`: with `x = d u`, `y = e v` as in `eq:torsor` and
`m = gcd(de, n)`, `de = m w`, `q = n / m` (`eq:mw`): `d ∣ m`, `e ∣ m`, `gcd(w, q) = 1`,
`uv` is a unit modulo `q`, and `x y mod n = m (w (u v) mod q)`. -/
theorem A4_stratum {d e u v : ℕ} (hd : d ∣ n) (he : e ∣ n)
    (hu : Nat.Coprime u (n / d)) (hv : Nat.Coprime v (n / e)) :
    d ∣ mOf n d e ∧ e ∣ mOf n d e ∧
    Nat.Coprime (wOf n d e) (qOf n d e) ∧ Nat.Coprime (u * v) (qOf n d e) ∧
    (d * u) * (e * v) % n = mOf n d e * (wOf n d e * (u * v) % qOf n d e) := by
  sorry

/-- A5, `eq:jcompose`: `J_d · J_e = J_{gcd(de, n)}` — containment and equality. -/
theorem A5_jcompose {d e : ℕ} (hd : d ∣ n) (he : e ∣ n) :
    J n d * J n e = J n (Nat.gcd (d * e) n) := by
  sorry

/-- A6, `01-setup.tex:79–80`: a **nonzero** nilpotent exists in `ℤ/nℤ` iff `n` is not
square-free. The paper's parenthetical omits "nonzero"; `0` is nilpotent in every ring. -/
theorem A6_nilpotent_iff : (∃ x : ZMod n, IsNilpotent x ∧ x ≠ 0) ↔ ¬ Squarefree n := by
  sorry

end Strata
