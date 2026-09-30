import Strata.Basic

/-!
# Tier B — regularity, descent, fibration, the character diagonal, `01-setup.tex`
-/

open Finset Pointwise

namespace Strata

variable {n : ℕ} [NeZero n]

/-- B2s, sharp form (`def:jclass` + `tab:moduli` caption): `J_d` is regular iff
`gcd(d, n/d) = 1`. B1 and B2 below, and tier E's non-regular counts, follow from it. -/
theorem B2_regular_iff {d : ℕ} (hd : d ∣ n) : Regular n d ↔ Nat.Coprime d (n / d) := by
  sorry

/-- B1, Chen et al. Prop. D.13 (`01-setup.tex:76–77`): square-free `n` ⇒ every `J_d` is
regular. -/
theorem B1_squarefree_regular (h : Squarefree n) {d : ℕ} (hd : d ∣ n) : Regular n d := by
  sorry

/-- B2, `tab:moduli` caption: `n` not square-free ⇒ some `J_d` is non-regular. -/
theorem B2_not_squarefree (h : ¬ Squarefree n) : ∃ d, d ∣ n ∧ ¬ Regular n d := by
  sorry

/-- B3a, `eq:descent`, square-free case: `J_d · J_d = J_d`. -/
theorem B3_descent_squarefree (h : Squarefree n) {d : ℕ} (hd : d ∣ n) :
    J n d * J n d = J n d := by
  sorry

/-- B3b, `eq:descent`: `J₁₁ · J₁₁ = {0}` at `n = 121`. -/
theorem B3_descent_121 : J 121 11 * J 121 11 = {0} := by
  sorry

/-- B3c, `eq:descent`: `J₅ · J₅ = J₂₅` and `J₂₅ · J₂₅ = {0}` at `n = 125`. -/
theorem B3_descent_125 : J 125 5 * J 125 5 = J 125 25 ∧ J 125 25 * J 125 25 = {0} := by
  sorry

/-- B3d, `01-setup.tex:98–99`, the abstract (`2^7`, depth 7) and `06-results-clock.tex:148`
(depths 2 to 4): `ν(p^a) = a`. Gives `ν(121) = 2`, `ν(125) = 3`, `ν(128) = 7`, `ν(81) = 4`. -/
theorem B3_nilDepth_prime_pow {p a : ℕ} (hp : p.Prime) (ha : 1 ≤ a) :
    nilDepth (p ^ a) = a := by
  sorry

/-- B4a, `eq:fibration`: reduction `(ℤ/(n/d))ˣ → G_m = (ℤ/q)ˣ` is surjective. -/
theorem B4_fibration_surjective {d e : ℕ} (hd : d ∣ n) (he : e ∣ n) :
    ∃ h : qOf n d e ∣ n / d, Function.Surjective (ZMod.unitsMap h) := by
  sorry

/-- B4b, `eq:fibration`: the target depends on `(u, v)` only through their residues mod `q`. -/
theorem B4_fibration_factors {d e u u' v v' : ℕ} (hd : d ∣ n) (he : e ∣ n)
    (huu : u % qOf n d e = u' % qOf n d e) (hvv : v % qOf n d e = v' % qOf n d e) :
    (d * u) * (e * v) % n = (d * u') * (e * v') % n := by
  sorry

/-- B4c, `01-setup.tex:236–237`: a stratum has `φ(n/d) φ(n/e) ≥ |G_m| = φ(q)` cells. -/
theorem B4_cells_ge {d e : ℕ} (hd : d ∣ n) (he : e ∣ n) :
    (qOf n d e).totient ≤ (n / d).totient * (n / e).totient := by
  sorry

/-- B5, consequence 3 (`eq:diagonal`): for a finite abelian group `G` and `f : G → ℂ`, the
2-D character transform of `(a, b) ↦ f(ab)` vanishes off the diagonal `χ = ψ`. -/
theorem B5_diagonal {G : Type*} [CommGroup G] [Fintype G] (f : G → ℂ)
    (χ ψ : G →* ℂˣ) (hne : χ ≠ ψ) :
    ∑ a, ∑ b, (((χ a)⁻¹ * (ψ b)⁻¹ : ℂˣ) : ℂ) * f (a * b) = 0 := by
  sorry

/-- B6, `eq:prodchar`, `eq:diagonal`: `|Ĝ| = |G|`, hence `|Δ| = |G_m|`. -/
theorem B6_card_dual {G : Type*} [CommGroup G] [Fintype G] :
    Nat.card (G →* ℂˣ) = Fintype.card G := by
  sorry

end Strata
