import Strata.Basic

/-!
# Tier D — the indicators' spectra, `paper/sections/09-results-crt.tex`
-/

open Finset

namespace Strata

/-- D0, `09-results-crt.tex:32–33`: for `d ∣ n`, the indicator of the multiples of `d` has
additive Fourier support exactly on the multiples of `n/d`. -/
theorem D0_multiples_support {n d k : ℕ} [NeZero n] (hd : d ∣ n) :
    dft n (univ.filter fun x : ZMod n => d ∣ x.val) k ≠ 0 ↔ n / d ∣ k := by
  sorry

/-- D1, `09-results-crt.tex:115`: the transform of `𝟙_{J_d}` at `k` is the Ramanujan sum
`c_{n/d}(k)`. -/
theorem D1_indicator_ramanujan {n d k : ℕ} [NeZero n] (hd : d ∣ n) :
    dft n (J n d) k = ramanujanSum (n / d) k := by
  sorry

/-- D2a, `09-results-crt.tex:116`: `c_q(k) = μ(q/g) φ(q) / φ(q/g)`, `g = gcd(k, q)`. -/
theorem D2_closed_form {q k : ℕ} (hq : 0 < q) :
    ramanujanSum q k =
      (ArithmeticFunction.moebius (q / Nat.gcd k q) : ℂ) * q.totient /
        (q / Nat.gcd k q).totient := by
  sorry

/-- D2b, `09-results-crt.tex:117`: magnitude at most 1 where `k` is coprime to `q`. -/
theorem D2_coprime_small {q k : ℕ} (hq : 0 < q) (hk : Nat.Coprime k q) :
    ‖ramanujanSum q k‖ ≤ 1 := by
  sorry

/-- D2c, `09-results-crt.tex:117–118`: the value `φ(q)` where `k` shares all of `q`. -/
theorem D2_full {q k : ℕ} (hq : 0 < q) (hk : q ∣ k) : ramanujanSum q k = q.totient := by
  sorry

/-- D2d, `09-results-crt.tex:119`: at square-free `q` a Ramanujan sum never vanishes — so at
`n = 165` every indicator's energy is nonzero at every frequency. -/
theorem D2_squarefree_ne_zero {q k : ℕ} (hq : Squarefree q) : ramanujanSum q k ≠ 0 := by
  sorry

/-- D3, `09-results-crt.tex:120–122`: at `n = 165`, `42` of the `82` non-DC frequencies share a
factor with `n`. (The `30.5×` energy ratio beside it is computed, not proved: brute force.) -/
theorem D3_count_165 :
    ((Icc 1 (165 / 2)).filter fun k => Nat.gcd k 165 ≠ 1).card = 42 ∧ 165 / 2 = 82 := by
  sorry

/-- D4, `09-results-crt.tex:35–37`: at `ω(n) = 1` the predicted set is every one of the
`⌊n/2⌋` bins, so a prime power can never count as support. -/
theorem D4_prime_power_vacuous {n : ℕ} (h : n.primeFactors.card = 1) :
    predicted n = Icc 1 (n / 2) := by
  sorry

end Strata
