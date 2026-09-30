import Mathlib

/-!
# Definitions shared by every tier

Each definition mirrors the paper (`paper/sections/`) or the Python it cites. A definition
mirrored from Python is checked against that Python on planted inputs before any theorem about
it is trusted (plan, "Rules carried in").
-/

open Finset

namespace Strata

/-- `def:jclass`, `eq:jclass`: `J_d = {x ∈ ℤ/nℤ : gcd(x, n) = d}`. -/
def J (n d : ℕ) [NeZero n] : Finset (ZMod n) :=
  univ.filter fun x => Nat.gcd x.val n = d

/-- `def:jclass`: a class is regular when it contains an idempotent. -/
def Regular (n d : ℕ) [NeZero n] : Prop :=
  ∃ ε ∈ J n d, ε * ε = ε

/-- `01-setup.tex:98`: the nilpotency depth
`ν(n) = min {k ≥ 1 : x^k = 0 for every nilpotent x ∈ ℤ/nℤ}`. -/
noncomputable def nilDepth (n : ℕ) : ℕ :=
  sInf {k | 1 ≤ k ∧ ∀ x : ZMod n, IsNilpotent x → x ^ k = 0}

/-- `eq:mw`: `m = gcd(de, n)`. -/
def mOf (n d e : ℕ) : ℕ := Nat.gcd (d * e) n
/-- `eq:mw`: `de = m w`. -/
def wOf (n d e : ℕ) : ℕ := d * e / mOf n d e
/-- `eq:mw`: `q = n / m`. -/
def qOf (n d e : ℕ) : ℕ := n / mOf n d e

/-- `eq:gini`, in its sort-free form: the mean absolute difference over twice the mean,
taken on `|A|` and `0` when `Σ|A| = 0`, as `src/analysis/sparsity.py::gini` does.
`C4` ties it to the paper's sorted formula. -/
noncomputable def gini {M : ℕ} (A : Fin M → ℝ) : ℝ :=
  if ∑ i, |A i| = 0 then 0
  else (∑ i, ∑ j, abs (|A i| - |A j|)) / (2 * M * ∑ i, |A i|)

/-- `eq:pr`: `PR(A) = (Σ A²)² / Σ A⁴`, `0` when the spectrum is zero, as
`src/analysis/sparsity.py::participation_ratio` does when handed the energy `A²`. -/
noncomputable def pr {M : ℕ} (A : Fin M → ℝ) : ℝ :=
  if ∑ i, A i ^ 4 = 0 then 0 else (∑ i, A i ^ 2) ^ 2 / ∑ i, A i ^ 4

/-- `eq:conventions`: `IPR₂(u) = (‖u‖₄ / ‖u‖₂)⁴ = Σ u⁴ / (Σ u²)²` (Doshi et al.). -/
noncomputable def ipr2 {M : ℕ} (u : Fin M → ℝ) : ℝ :=
  (∑ i, u i ^ 4) / (∑ i, u i ^ 2) ^ 2

/-- `eq:project`: the spectral projection onto a set `S` of character pairs. -/
def proj {κ β : Type*} [Zero β] (S : Set (κ × κ)) [DecidablePred (· ∈ S)]
    (L : κ × κ → β) : κ × κ → β :=
  fun p => if p ∈ S then L p else 0

/-- The additive DFT of a set of residues at frequency `k` (sign convention of
`test_crt_law.freq_energy`; the Ramanujan sums below are real, so the sign does not matter). -/
noncomputable def dft (n : ℕ) [NeZero n] (S : Finset (ZMod n)) (k : ℕ) : ℂ :=
  ∑ x ∈ S, Complex.exp (-2 * Real.pi * Complex.I * (x.val * k : ℕ) / n)

/-- The Ramanujan sum `c_q(k) = Σ_{0 ≤ a < q, gcd(a,q)=1} exp(2πi a k / q)`. Not in mathlib. -/
noncomputable def ramanujanSum (q k : ℕ) : ℂ :=
  ∑ a ∈ (range q).filter (Nat.Coprime · q),
    Complex.exp (2 * Real.pi * Complex.I * (a * k : ℕ) / q)

/-- `eq:crtset`, as `test_crt_law.predicted`: the multiples, up to `⌊n/2⌋`, of `n / q` for
each maximal prime power `q ∥ n`. -/
def predicted (n : ℕ) : Finset ℕ :=
  (Icc 1 (n / 2)).filter fun k =>
    ∃ p ∈ n.primeFactors, n / p ^ n.factorization p ∣ k

end Strata
