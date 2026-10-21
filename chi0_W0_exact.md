# Direct computation of χ₀ and W₀ from the resolvent

For a finite impurity Hamiltonian, both quantities can be computed directly,
without extrapolating a Matsubara grid:

$$
\chi_0=\lim_{\nu\to0}[-2\operatorname{Re}R(i\nu)],
\qquad
W_0=\lim_{\nu\to0}[-\nu\operatorname{Im}R(i\nu)].
$$

The key is to separate the exactly zero-energy part of the spectrum from the
excited states.

## 1. Exact spectral formulas

For the resolvent used here,

$$
R(z)=\langle v|[z-(H-E_0)]^{-1}|v\rangle,
\qquad |v\rangle=O|\psi_0\rangle,
$$

its Lehmann representation is

$$
R(z)=\sum_n\frac{W_n}{z-\Delta_n},
\qquad
W_n=|\langle n|v\rangle|^2,
\qquad \Delta_n=E_n-E_0.
$$

Here, $|\psi_0\rangle$ is an exact normalized ground state and $E_0$ is its
energy. For Hermitian $O$, this is the usual operator autocorrelation resolvent.
For a connected resolvent, replace $O$ by $O-\langle O\rangle$ throughout.

On the imaginary axis,

$$
-2\operatorname{Re}R(i\nu)
=2\sum_{\Delta_n>0}\frac{W_n\Delta_n}{\Delta_n^2+\nu^2},
$$

and

$$
-\nu\operatorname{Im}R(i\nu)
=\sum_n\frac{W_n\nu^2}{\Delta_n^2+\nu^2}.
$$

Therefore,

$$
\boxed{\chi_0=2\sum_{\Delta_n>0}\frac{W_n}{\Delta_n}},
\qquad
\boxed{W_0=\sum_{\Delta_n=0}W_n}.
$$

These formulas apply directly to a finite spectrum. In a continuum or
thermodynamic limit, the inverse-energy moment defining $\chi_0$ may diverge,
and the order of limits must be specified.

## 2. Direct computation without full diagonalization

Let

$$
A=H-E_0,
$$

let $P_0$ project onto the **entire ground-state eigenspace**, and set
$Q=1-P_0$. Then

$$
\boxed{W_0=\langle v|P_0|v\rangle=\|P_0v\|^2},
\qquad
\boxed{\chi_0=2\langle v|A^+|v\rangle},
$$

where $A^+$ is the Moore–Penrose pseudoinverse: it inverts positive excitation
energies and annihilates the ground manifold.

A practical implementation is:

1. Obtain the ground states needed to construct $P_0v$.
2. Form $b=Qv$.
3. Solve the projected linear system

   $$
   Ax=b,\qquad P_0x=0.
   $$

4. Evaluate

   $$
   \chi_0=2\operatorname{Re}\langle b|x\rangle.
   $$

Equivalently, solve with $QAQ$ restricted to the range of $Q$, where it is
positive definite for a finite gapped spectrum. This avoids evaluating a
singular resolvent at $z=0$.

### Nondegenerate ground state

For a nondegenerate ground state,

$$
W_0=|\langle\psi_0|O|\psi_0\rangle|^2,
\qquad
b=(O-\langle O\rangle)|\psi_0\rangle.
$$

Thus the connected resolvent has $W_0=0$ exactly.

For a degenerate ground state, subtracting the mean removes only the component
parallel to the selected $|\psi_0\rangle$. Transitions to other degenerate
ground states can still contribute to $W_0$. The projector must include all
zero-energy states accessible to the seed in the Hilbert space being evaluated.

## 3. Extraction from Lanczos

The same quantities can be extracted from the Lanczos projected Hamiltonian.
Suppose $v\ne0$, Lanczos starts from $v/\|v\|$, and the shifted tridiagonal
matrix has normalized eigenpairs $(\theta_j,u_j)$. Its spectral weights are

$$
w_j=\|v\|^2 |(u_j)_1|^2.
$$

Then

$$
W_0^{\mathrm{Lan}}=\sum_{\theta_j=0}w_j,
\qquad
\chi_0^{\mathrm{Lan}}=2\sum_{\theta_j>0}\frac{w_j}{\theta_j}.
$$

These are exact for a complete seed-generated Krylov space in exact arithmetic.
With truncated Lanczos they are approximations requiring convergence checks.
If $v=0$, both quantities vanish and no Lanczos run is needed.

**Projecting out the ground manifold before Lanczos is preferable for $\chi_0$.**
A slightly misplaced zero pole can otherwise produce a spurious large
contribution. With the projected seed $b$, all exact spectral weight lies at
positive excitation energies, and the prefactor in the Lanczos weights is
$\|b\|^2$.

In floating-point arithmetic, identifying zero eigenvalues requires a tolerance
justified by eigenpair residuals and ground-energy accuracy. A small positive
gap must not be classified as zero solely because it is below a chosen
Matsubara frequency.

## 4. What “exact” means in the ASCI solver

These procedures remove **frequency-extrapolation error**. Their accuracy still
depends on the determinant space, ground-state accuracy, and linear-solver or
Lanczos convergence. A result can be exact within the selected Hamiltonian
while still differing from the full-FCI result.

The reference energy and ground projector must belong to the same Hamiltonian
used for the response. In particular, an approximate ASCI reference energy
that differs from that Hamiltonian's ground energy shifts the would-be zero
pole and changes the strict limits.

Two distinctions matter:

- A tiny positive gap contributes to $\chi_0$, but **not** to the exact $W_0$.
  Finite-$\nu$ data can make it look like a zero pole.
- If $W_0>0$, the definition of $\chi_0$ above still returns the regular,
  excited-state contribution: the zero pole is purely imaginary on the
  Matsubara axis. It should not automatically be identified with a finite
  thermodynamic static susceptibility of a degenerate moment.

For this solver, the cleanest route is **ground-manifold projection for $W_0$,
followed by a projected zero-frequency solve for $\chi_0$**.

## 5. Can the existing code simply evaluate the resolvent at zero?

**Only if the resolvent has no zero-energy weight.** Evaluating the existing
resolvent at $z=0$ is not a general way to compute both quantities.

Separate the resolvent as

$$
R(z)=\frac{W_0}{z}+R_{\mathrm{reg}}(z),
\qquad
R_{\mathrm{reg}}(0)=-\sum_{\Delta_n>0}\frac{W_n}{\Delta_n}.
$$

### If W₀ = 0

The scalar resolvent has a finite limit, and

$$
\boxed{\chi_0=-2R(0)}.
$$

Here $R(0)$ means the regular extension of the scalar resolvent to zero.
Evaluating at zero can therefore give $\chi_0$ directly, provided the numerical
evaluator handles it.

There is a numerical subtlety: $H-E_0$ remains singular even when the seed has
no ground-state overlap. A seed-based Lanczos calculation can avoid that null
direction in exact arithmetic, but numerical leakage or an inaccurate $E_0$
can spoil the result. A generic full-matrix inverse at zero would still fail.

For an exact, nondegenerate ground state, using the connected operator
$O-\langle O\rangle$ guarantees $W_0=0$. For a degenerate ground manifold,
subtracting the mean alone does not guarantee this.

### If W₀ > 0

Then $R(0)$ is undefined because of the pole. Computing at exactly zero and
discarding the imaginary part is not valid: at $z=0$, the singular expression
is not a valid complex number.

Nevertheless, the susceptibility limit is finite for a finite gapped excited
spectrum:

$$
\chi_0=-2R_{\mathrm{reg}}(0).
$$

Extracting it requires removing the zero pole or evaluating the limit from
nonzero frequencies. Likewise,

$$
\boxed{W_0=\lim_{z\to0}zR(z)}
$$

is a **residue**, not the value of $R$ at zero. Direct substitution gives an
undefined product $0\times\infty$.

### Keeping the code unchanged: small nonzero frequencies

If the existing interface accepts sufficiently small nonzero imaginary
frequencies, both quantities can be estimated using

$$
\chi(\nu)=-2\operatorname{Re}R(i\nu),
\qquad
W(\nu)=-\nu\operatorname{Im}R(i\nu).
$$

For a finite system, when $\nu$ is below the smallest positive excitation energy
carrying weight,

$$
\chi(\nu)=\chi_0
-2\nu^2\sum_{\Delta_n>0}\frac{W_n}{\Delta_n^3}
+O(\nu^4),
$$

and

$$
W(\nu)=W_0
+\nu^2\sum_{\Delta_n>0}\frac{W_n}{\Delta_n^2}
+O(\nu^4).
$$

This supports extrapolation in $\nu^2$, although its accuracy still depends on
the computed pole positions and weights. Making $\nu$ arbitrarily small does
not cure an inaccurately positioned zero pole.

**Bottom line:** evaluating at zero can work for $\chi_0$ when the zero pole is
absent and the evaluator handles the singular Hamiltonian correctly. It cannot
generally yield $W_0$, or handle a resolvent with $W_0>0$, by direct substitution.
