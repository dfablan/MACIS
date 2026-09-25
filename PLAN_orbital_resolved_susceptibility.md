# Plan: general orbital-resolved spin susceptibility $\chi_{\mu\nu;\gamma\delta}$ in MACIS

## Context

`Orbital_Resolved_Susceptibility.tex` generalizes the scalar $S_z^{\rm imp}$ resolvent to the
four-index object

$$R_{\mu\nu;\gamma\delta}(z)=\braket{\phi_{\mu\nu}|[z-(H-E_0)]^{-1}|\phi_{\gamma\delta}},
\qquad \ket{\phi_{\mu\nu}}=S_{\mu\nu}\ket{\psi_0},$$

with $S_{\mu\nu}=c^\dagger_{\mu\uparrow}c_{\nu\uparrow}-c^\dagger_{\mu\downarrow}c_{\nu\downarrow}$
and $\mu,\nu$ running over the $n_{\rm imp}$ impurity orbitals.

`Orbital_Resolved_Resolvent_Plan.md` analyses the three-band FeSC case under $D_{4h}$ and reduces
the $81$ elements to $9$ independent functions. **This plan deliberately does not assume that
symmetry.** The target is a solver that computes the full $n_{\rm imp}^2\times n_{\rm imp}^2$
matrix for any impurity — two or three degenerate bands on a square lattice with no crystal
field, the FeSC $D_{4h}$ case, or a symmetry-broken bath — and *reports* whatever block structure
is present rather than presupposing it.

The design consequence is that `BandResolvent` (`include/macis/gf/bandlan.hpp:271`) becomes the
primary path, not an optimization. It takes a set of seed vectors and returns the entire matrix
resolvent $\braket{v_k|[z-(H-E_0)]^{-1}|v_l}$ in one band-Lanczos pass, which is exactly the object
above. Symmetry-adapted seeding and polarization identities are demoted to an optional cost
optimization for the case where the point group happens to be known.

### Why the naive call does not work

Feeding the $n_{\rm imp}^2$ raw $\ket{\phi_{\mu\nu}}$ straight to `BandResolvent` fails.
`QRdecomp_tr` (`src/macis/gf/bandlan.cxx:192`) hard-returns all zeros on a linearly dependent seed
set (`:193-196`), and the seeds *are* dependent — generically, and in one case provably:

$$\sum_\mu \ket{\phi_{\mu\mu}} = 2\,S_z^{\rm imp}\ket{\psi_0},$$

which vanishes identically whenever $n_{\rm imp}=n_{\rm active}$ and $S_z^{\rm tot}=0$, since
$S_z^{\rm imp}$ is then a good quantum number. This is the zero-norm case already guarded in
`RunResolventDiagonal` (`dynamical_properties.hpp:405-411`). More generally every symmetry of $H$
that annihilates some combination of bilinears produces another exact null direction, and near-null
directions appear whenever a symmetry is weakly broken.

> **Review note (2026-09-24).** The provable null needs $n_{\rm imp}=n_{\rm active}$, which never
> holds in a DMFT solve with a bath. In production, deflation will rarely discard anything. It is
> still worth keeping, because it costs nothing and covers the bath-free and symmetric test cases.
> "Required" overstates how often it fires.

So rank deflation is **required infrastructure**, not a diagnostic. Making it explicit is what
buys the generality: the deflation is precisely where symmetry enters, and it is *measured* from
the ground state rather than assumed.

---

## Architecture

Three pieces, in dependency order.

```
Stage 1  apply_spin_bilinear        build the n_imp^2 seeds
Stage 2  Gram deflation + BandResolvent + back-transform     the matrix resolvent
Stage 3  post-processing            eq. (29) eigenmodes, sum rules, symmetry report
```

---

## Stage 1 — the bilinear operator

### 1.1 `include/macis/gf/dynamical_properties.hpp`

```cpp
template <size_t nbits>
Eigen::VectorXd apply_spin_bilinear(const Eigen::VectorXd &wfn0,
                                    const std::vector<std::bitset<nbits>> &dets,
                                    const std::map<std::bitset<nbits>, size_t,
                                                   bitset_less_comparator<nbits>> &det_index,
                                    size_t mu, size_t nu);
```

Returns $S_{\mu\nu}\ket{\psi_0}$ in the **same** `dets` basis. Per determinant $k$, per spin
$\sigma$: with `sporb_nu = nu + (dn ? nbits/2 : 0)` and likewise for `mu`, if $\nu$ is occupied and
($\mu=\nu$ or $\mu$ empty), flip both bits, look up the target index, accumulate
`sign * (up ? +1.0 : -1.0) * wfn0[k]`.

Pass the index map in rather than rebuilding it — Stage 2 calls this $n_{\rm imp}^2$ times and the
map is $O(|{\rm dets}|\log|{\rm dets}|)$ to build.

Reuse, do not reinvent:

- **Sign**: `single_excitation_sign` (`include/macis/sd_operations.hpp:594`) — the canonical
  $a^\dagger_p a_q$ phase used by the Hamiltonian generator, the RDM builder and ASCI. Mirror the
  call convention at `sd_operations.hpp:612` and the accumulation idiom in `rdm_contributions_2`
  (`include/macis/util/rdms.hpp:59-60`), which is the same $a^\dagger_v a_o$ contraction.
  Do **not** use `GetInsertionUpSign` (`gf.hpp:98`); its bookkeeping differs and it does not test
  occupancy.
- **Lookup**: `p.dets` is not in bitset order on either solver path (CAS uses `prev_permutation`
  descending; ASCI sorts by $|C|$, `asci/determinant_sort.hpp:19-25`), so binary search is invalid.
  Use `std::map<std::bitset<nbits>, size_t, bitset_less_comparator<nbits>>`, the pattern at
  `gf.hpp:419-425`.
- $\mu=\nu$ must fall through to the diagonal result, giving a free cross-check against
  `weighted_imp_value` (`dynamical_properties.hpp:174`).

### 1.2 Bit-convention hazard

Three incompatible conventions coexist in this codebase. Stage 1 must live entirely in the first:

1. **Raw bitset** (`sd_operations.hpp`, `hamiltonian_generator.hpp`): alpha orbital $i$ at bit $i$,
   beta at bit $i+N/2$. **Use this one.**
2. `GetInsertionUpSign`/`GetInsertionDoSign` (`gf.hpp:98,117`): same layout, but a
   count-strictly-below evaluated on the *pre-insertion* state, no occupancy test.
3. `decompose_det` (`observables/impurity_rdm.hpp:42-45`): **reversed** — orbital $i$ at bit
   $n_{\rm imp}-1-i$ — repacked into a `uint64_t` lookup key with up/down nesting inverted.

Never let a `decompose_det` bit position reach sign arithmetic. The existing diagonal machinery
(`weighted_imp_value`) is entirely in convention 3; the new operator is entirely in convention 1;
the $\mu=\nu$ cross-check above is what verifies they agree.

### 1.3 Determinant-space closure — measure it, do not gate on it

$S_{\mu\nu}$ conserves $N_\alpha$ and $N_\beta$ separately, so the **complete FCI space is closed
under it**. `generate_hilbert_space` (`sd_operations.hpp:550-566`) builds exactly that — the full
outer product of all $\binom{n_{\rm act}}{n_\alpha}\times\binom{n_{\rm act}}{n_\beta}$ strings — and
`SolveImpurityED` fills `p.dets` from it (`src/macis/impurity_solver.cpp:372-373`). On the CAS/ED
path the construction is therefore exact.

All three ASCI paths truncate (`asci/determinant_search.hpp:591-604`, capped at `asci/grow.hpp:53`);
`SolveImpurityCheapASCI` inherits the truncated basis unchanged (`impurity_solver.cpp:785`). There
$S_{\mu\nu}\ket{\psi_0}$ leaks out of `p.dets` for $\mu\neq\nu$.

**This is not a reason to refuse to run.** The existing production $S_z$ resolvent already evaluates
the Lanczos inside the truncated space, so its poles are those of $PHP$, not $H$. Projecting the
seed as well,

$$R_{\rm proj}(z)=\phi_P^{T}\big[z-(PHP-E_0)\big]^{-1}\phi_P,
\qquad \phi_P = P\,S_{\mu\nu}\ket{\psi_0},$$

is an ordinary Galerkin projection of the resolvent — the same *kind* of approximation already
accepted on the Hamiltonian side, with sum rule $\|\phi_P\|^2$. Requiring an exact operator while
tolerating a truncated $H$ would be inconsistent.

What *is* new for $\mu\neq\nu$ is a second error channel: lost seed norm. So measure it.

**Norm capture fraction.** Walk `base_dets`, apply $S_{\mu\nu}$, and accumulate coefficients into a
hash map keyed by image determinant, **including images outside `base_dets`**. Then

$$\texttt{capture}_{\mu\nu}=\frac{\sum_{D\in\,\text{base\_dets}}|c_D|^2}{\sum_{D\,\text{all}}|c_D|^2}
=\frac{\|\phi_P\|^2}{\|S_{\mu\nu}\ket{\psi_0}\|^2}.$$

Coefficients must be accumulated per image determinant *before* squaring, since several sources map
to the same image — hence the map. One pass, $O(|{\rm dets}|\cdot n_{\rm imp}^2)$, no RDM needed and
no Hamiltonian built on the enlarged space (the Hamiltonian is the expensive part of §1.4, the
vector is not).

Report `capture` per element in `<label>_gram.dat`. Warn below `GF.ORB_MIN_CAPTURE` (default `0.95`);
never throw on it. On CAS/ED it is exactly $1$, which doubles as a test that the closure argument
above is right.

Keep the `orb_rot`-identity guard (`impurity_solver.hpp:417-438`) — that one *is* a hard error, since
a rotated basis makes the operator meaningless rather than merely truncated.

### 1.4 If capture turns out to be poor

Fall back to enlarging the basis: `base_dets` $\cup$ $\bigcup_{\mu\nu}S_{\mu\nu}(\texttt{base\_dets})$,
then build $H$ over the union. This is exactly the `gf_dets` construction the Green's-function path
already performs (`get_GF_basis_AS_1El`, `gf.hpp:229`; assembly at `gf.hpp:580-614`), so it is an
existing pattern rather than new machinery, and `generate_singles_spin` (`sd_operations.hpp:419`)
generates the image. Expect a 10–50× larger space and a correspondingly larger CSR build.

Expected outcome: capture should be high, because $S_{\mu\nu}\ket{\psi_0}$ is a single excitation off
$\ket{\psi_0}$ and singles off the dominant configurations are precisely what ASCI's candidate
ranking retains. That is a prediction, not a guarantee — which is why §1.3 measures it rather than
assuming it. Do not build this fallback until the diagnostic says it is needed.

---

## Stage 2 — the general matrix resolvent

### 2.1 Algorithm

Let $M=n_{\rm imp}^2$, $L=|{\rm dets}|$, and $\Phi$ the $L\times M$ matrix of seed columns
$\ket{\phi_{\mu\nu}}$ in a fixed pair ordering (row-major in $(\mu,\nu)$).

1. **Seeds.** Build all $M$ columns via `apply_spin_bilinear`.
2. **Gram.** $\mathcal{G}=\Phi^{T}\Phi$, $M\times M$, real symmetric positive semidefinite.
   Costs $M^2/2$ dot products — no Lanczos, negligible.
3. **Deflate.** Eigendecompose $\mathcal{G}=U\Lambda U^{T}$ (Eigen `SelfAdjointEigenSolver`, at most
   an $81\times81$ problem). Keep the $r$ modes with $\lambda_k>\texttt{tol}\cdot\lambda_{\max}$;
   default `tol = 1e-10`, exposed as `GF.ORB_DEFLATE_TOL`.
4. **Orthonormal seeds.** $\Psi=\Phi\,U_r\Lambda_r^{-1/2}$, an $L\times r$ matrix with
   $\Psi^{T}\Psi=I_r$ by construction.
5. **Band Lanczos.** Pass $\Psi$ to `BandResolvent` with `nvecs = r`, `ispart = true`,
   `len_vec = L`. Returns $\tilde R(z)=\Psi^{T}[z-(H-E_0)]^{-1}\Psi$, $r\times r$.
6. **Back-transform.** With $B=U_r\Lambda_r^{1/2}$ ($M\times r$),

$$\boxed{\;R(z) = B\,\tilde R(z)\,B^{T}\;}$$

The discarded directions satisfy $\|\Phi u\|^2=u^{T}\mathcal{G}u\approx 0$ — they are combinations
of bilinears that annihilate the ground state, so projecting them out is exact, not an
approximation.

> **Review note (2026-09-24).** It is exact only up to the tolerance. `tol` is applied to Gram
> eigenvalues, i.e. to *squared* norms. At `1e-10` a discarded direction can carry amplitude
> $\|\Phi u\|\sim\sqrt{10^{-10}}\,\|\Phi\|=10^{-5}\|\Phi\|$, and the cross terms it drops
> put a relative error of about $10^{-5}$ on $R$. Apply the tolerance to singular values
> ($\sqrt{\lambda_k/\lambda_{\max}}$), or lower the default to about `1e-14`. Imperfect
> orthonormality of $\Psi$ (error $\sim\epsilon\,\lambda_{\max}/\lambda_k$) is *not* a problem:
> `BandResolvent` re-orthonormalises through `QRdecomp_tr` and folds $R$ back into the `S`
> matrix. The result is therefore exactly $\Psi^T G\Psi$ for the $\Psi$ actually passed in, and
> $B\tilde R B^T = U_rU_r^T\,\Phi^TG\Phi\,U_rU_r^T$ holds regardless. **That null space is the symmetry content of the problem, measured rather than
assumed.**

Feeding `BandResolvent` an already-orthonormal set also removes its failure mode entirely: its
internal `QRdecomp_tr` is then perfectly conditioned and returns $R=I$, so the `S`-matrix
construction at `bandlan.cxx:300-308` reduces to the eigenvector matrix.

### 2.2 `BandResolvent` call mechanics

- `vecs` is flat and **vector-major**: `vecs[j * len_vec + i]` = component $i$ of vector $j$
  (`bandlan.cxx:206`; same layout `BuildWfn4Lanczos` produces as `wfns[iorb * nterms + ndet]`).
- It takes `std::vector<double> &vecs` by **non-const reference and destroys it** — `QRdecomp_tr`
  overwrites in place and `BandLan` ends with `qs.clear()` (`bandlan.hpp:245`). Pass a copy.
- Output is `res[iw][k * nvecs + l]` $=\braket{v_k|G(z)|v_l}$ — already the `.tex` index
  convention, dagger in the right place, no index-reversal bookkeeping.
- Needs `nLanIts` comfortably above $r$; `GetEigsysBand` uses `min(nvecs, nLanIts-1)`
  (`bandlan.cxx:280`).
- **Review note:** `nLanIts` is the *total* Krylov dimension, not a count per seed. The default
  1000 with $r=36$ is only about 28 block steps, far less resolved than the scalar $S_z$ run with
  the same setting. On real-frequency grids, raise `GF.NLANITS` roughly $\propto r$. `bandH`
  and the eigensolve in `GetEigsysBand` are dense $n_{\rm Lan}^2$ / $n_{\rm Lan}^3$, so this is
  cheap up to a few thousand.
- Memory is $2r$ vectors of length $L$ (`bandlan.hpp:164`), not $r\cdot n_{\rm Lan}$. Cost is $r$
  matvecs per iteration.
- Watch stdout for `FOUND A ZERO VECTOR AT POSITION` (`bandlan.hpp:225`): the early-exit branch is
  commented out (`:220-226`), so dependence emerging mid-Krylov is silently zeroed rather than
  reported. After deflation this should never fire; if it does, the deflation tolerance is too loose.

### 2.3 Entry point — `include/macis/impurity_solver.hpp`

```cpp
template <size_t N>
auto evaluate_resolvent_orbital_matrix(double EASCI, macis::impurity_params<N> &p,
                                       macis::SDBuildHamiltonianGenerator<N> &ham_gen,
                                       macis::GFSettings &gf_settings,
                                       const std::string &label);
```

Reuse verbatim: `detail::build_bosonic_resolvent_grid` (`:266`) — the bosonic Matsubara grid is
correct here for the same reason it is for $S_z$ — the `E0 = EASCI - (p.E_core + p.E_inactive)`
shift (`:448`), and the CSR build from `RunResolventGS` (`dynamical_properties.hpp:86-88`).

Note this bypasses `RunResolventGS`, which is single-vector only; the Hamiltonian build
(`make_dist_csr_hamiltonian`) is shared but `BandResolvent` is called directly.

### 2.4 Output

Two files, both long-format for easy post-processing:

- `<label>_orbital_resolvent.dat` — one row per `(iw, pair_k, pair_l)`:
  `Re(w) Im(w) mu nu gamma delta Re(R) Im(R)`. Mirrors the spirit of `write_GF` (`gf.hpp:498`)
  but flat, since a rank-4 object does not fit that writer's layout.
- `<label>_gram.dat` — the $M\times M$ Gram matrix plus its eigenvalue spectrum and the retained
  rank $r$.

Rank-0 write guard as in `detail::write_resolvent_singlef` (`impurity_solver.hpp:286-299`).

### 2.5 Driver — `main/run_asci_impsolv_dop.cxx`

Three edits mirroring the `GF.TZ_RESOLVENT` block (`:557-558`, `:571`, `:640-654`):

1. `bool spin_orb_matrix_resolvent = false;` + `OPT_KEYWORD("GF.SPIN_ORB_MATRIX_RESOLVENT", ..., bool)` near
   `:564`, plus `GF.ORB_DEFLATE_TOL` (`double`, default `1e-10`) and `GF.ORB_MIN_CAPTURE`
   (`double`, default `0.95`, warn-only — see §1.3).
2. Add the flag to the gating disjunction at `:571`.
3. Add the call near `:652`.

**No band-ordering assumption is needed.** Because every pair is computed, nothing in the solver
depends on which band is $xz$ vs $xy$; that labelling matters only when interpreting the output.
This is a direct benefit of dropping the symmetry-adapted route.

---

## Stage 3 — post-processing

No new C++; these consume `<label>_orbital_resolvent.dat`.

1. **Sum rule, per element.** $\int d\omega\,\mathcal{S}_{\mu\nu;\gamma\delta}(\omega)
   =\braket{\phi_{\mu\nu}}{\phi_{\gamma\delta}}=\mathcal{G}_{\mu\nu;\gamma\delta}$ — the Gram matrix
   is written out precisely so every one of the $M^2$ continued fractions can be validated
   independently against a quantity that cost no Lanczos iterations.
2. **Equation (29).** Diagonalize ${\rm Im}\,\chi(\Omega)$ — the anti-Hermitian part of $R$ — at
   each $\Omega$. Eigenvectors $M^{(n)}_{\mu\nu}$ are the orbital wavefunctions of the spin
   excitations, eigenvalues their spectral weights. With the full matrix in hand this is a
   statement about the whole spectrum, which is the point of doing it generally.
3. **Symmetry report.** Block-structure detection on $\mathcal{G}$ and on $R(z)$: which elements sit
   at the noise floor, and what the retained rank $r$ is. For degenerate bands this *discovers* the
   multiplet structure; for the $D_{4h}$ case it should reproduce the 36 zeros and 9 independent
   functions of `Orbital_Resolved_Resolvent_Plan.md` §2.4 without having assumed them — which is
   simultaneously the strongest correctness check available and the answer to "is the dominant mode
   inside the diagonal block."
4. **Frozen weights.** The full $w_0^{\alpha\beta}$ matrix including off-diagonals. Per the physics
   plan §8.1 these carry the Hund's-locking weight and are *not* captured by the diagonal alone.

### Normalization

`weighted_imp_value` with `DiagChannel::Spin` computes $\sum_i w_i S_z^i$, whereas the `.tex` uses
$S_{\mu\mu}=2S_z^\mu$. Stage 1/2 work directly with $S_{\mu\nu}$, i.e. the `.tex` convention, so
**the matrix resolvent is 4× the corresponding $S_z$-convention quantity** on the diagonal block.
State the convention in the output header. Physics-plan §6 is written in the $S_z^\alpha$
convention and is inconsistent with its own §1 shorthand by exactly this factor — worth fixing in
that note.

---

## Extension: the charge channel $N_{\mu\nu}$

Everything above is written for the spin bilinear $S_{\mu\nu}$. `~/Work/Notes/Sz_Resolvent/
Orbital_Channels_and_Observable_Choice.md` classifies the full set of one-body impurity bilinears
and puts this extension in context; the relevant points from it are collected here rather than
re-derived.

### What the operator is

$$N_{\mu\nu}=\sum_\sigma c^\dagger_{\mu\sigma}c_{\nu\sigma}=c^\dagger_{\mu\uparrow}c_{\nu\uparrow}+c^\dagger_{\mu\downarrow}c_{\nu\downarrow},$$

the same-sign companion of $S_{\mu\nu}=c^\dagger_{\mu\uparrow}c_{\nu\uparrow}-c^\dagger_{\mu\downarrow}c_{\nu\downarrow}$.
`Orbital_Channels_and_Observable_Choice.md` §1 shows $N_{\mu\nu}$ transforms as a $3\otimes3$ tensor
under $SO(3)_{\rm orb}$ exactly as $S_{\mu\nu}$ does, decomposing $1\oplus3\oplus5$ (singlet $N_{\rm
imp}$, orbital-moment triplet $L^\gamma$, quadrupole quintet $Q$, with $T^3,T^8$ two of the five
$Q$ components). Together the two channels are the full $18=9+9$-dimensional classification
(spin $0,1$) $\times$ (orbital $l=0,1,2$) — §2 of that note.

### Why this is a small change, not a new derivation

Stage 1's sign machinery is already channel-agnostic. `apply_spin_bilinear` differs from an
`N_{\mu\nu}` builder only in the relative sign between the two spin-block terms — the
`(spin ? -1.0 : 1.0)` factor in the per-determinant loop becomes `+1.0` unconditionally for charge.
Everything downstream — Gram, deflation, `BandResolvent`, back-transform (Stage 2) — operates on
whichever seeds it is handed and needs no change.

Concretely:

1. **`dynamical_properties.hpp`**: generalize `apply_spin_bilinear` /
   `spin_bilinear_capture_fraction` to take a `DiagChannel ch` (reusing the enum already defined at
   `dynamical_properties.hpp:306`, matching the existing `weighted_imp_value` convention rather than
   inventing a new tag), and rename to `apply_orbital_bilinear` /
   `orbital_bilinear_capture_fraction` since "spin" no longer describes both branches. Same for
   `RunResolventOrbitalMatrix` (→ `RunResolventOrbitalBilinearMatrix`, taking `DiagChannel channel`).
2. **`impurity_solver.hpp`**: `evaluate_resolvent_orbital_matrix` gets the same `DiagChannel`
   parameter, forwarded through; fold it into the output label (`"Sz"` vs `"N"`) so the two channels
   don't overwrite each other's `.dat` files.
3. **`run_asci_impsolv_dop.cxx`**: a second keyword, `GF.CHARGE_ORB_MATRIX_RESOLVENT` (bool
   `charge_orb_matrix_resolvent`), mirroring `GF.SPIN_ORB_MATRIX_RESOLVENT` rather than folding both
   into one flag — consistent with the existing one-flag-per-physical-channel pattern
   (`sz_resolvent`, `tz_resolvent`, `stag_sz_resolvent`).

### What the charge channel actually buys, and what does not need it

Per the note's §2/§3/§7 table, most of the charge sector is already reachable with the existing
diagonal machinery:

- **$l=0$** ($N_{\rm imp}$): trivial, uniform `CHARGE` weights, already available via
  `evaluate_resolvent_diagonal` — no new code needed.
- **$l=2$** (the quadrupole $Q$): 2 of 5 components ($T^3$, `GF.TZ_RESOLVENT`; $T^8$) are already
  diagonal-operator accessible and implemented. The matrix resolvent's contribution here is the
  remaining 3 off-diagonal quintet components plus the full $5\times5$ block including frozen
  weights $w_0^{\alpha\beta}$ (Stage 3 item 4 above, "Frozen weights", mirrored for charge).
- **$l=1$** ($L^\gamma$, the orbital angular momentum): **genuinely off-diagonal in the
  determinant basis** — note §7. This is the piece the diagonal-operator interface (§9 of the older
  `PLAN_orbital_resolvent.md`) cannot reach at all. It is exactly what Stage 1's bit-flip
  construction was built for: $\mu\neq\nu$ seeds via `single_excitation_sign` require no new
  machinery beyond the sign generalization above. **The charge-channel matrix resolvent is the
  first thing in this codebase that reaches $L^\gamma$**, not just a parallel run of what the
  diagonal path already does for $Q$.

### A physics wrinkle: the diagonal trace is not a null direction here

The spin channel's exact null direction ($\sum_\mu S_{\mu\mu}\ket{\psi_0}=0$ at $n_{\rm imp}=n_{\rm
active}$, $S_z^{\rm tot}=0$) has no charge analogue in the same form. At $n_{\rm imp}=n_{\rm
active}$, $\sum_\mu N_{\mu\mu}=N_{\rm tot}$ is conserved with eigenvalue $N_{\rm tot}\neq0$
generically, so $\sum_\mu N_{\mu\mu}\ket{\psi_0}=N_{\rm tot}\ket{\psi_0}$ — **parallel to the
ground state, not annihilated by it**. This is not a Gram rank deficiency ($\|\Phi u\|^2=N_{\rm
tot}^2\neq0$ for that direction), but it means that particular seed is *entirely* elastic content at
$n_{\rm imp}=n_{\rm active}$, and dominated by the elastic pole away from it. `subtract_mean`
(already threaded through `RunResolventOrbitalMatrix`, see the review section above) is effectively
mandatory for any diagonal-block charge element, more so than for spin, where it is only needed
away from half filling. Add a test analogous to the spin deflation test (`orbital matrix resolvent
deflates the exact null direction`) that checks $\sum_\mu N_{\mu\mu}\ket{\psi_0}\parallel\ket{\psi_0}$
instead of $=0$, and confirms `subtract_mean` removes that pole from every diagonal-block element.

### Cross-channel test from the note

Spin and charge sectors cannot mix under $H$ ($S$-type operators are spin triplet, $N$-type spin
singlet — note §1), so the spin/charge cross-block of a combined Gram matrix is exactly zero; this
is a free correctness check if both channels are ever seeded together, though the plan above keeps
them as two independent runs rather than one $2M\times2M$ problem, matching the driver's
one-flag-per-channel pattern.

The note's §5.2 gives a stronger quantitative check once both channels exist: at the Kanamori
$J=0$ point ($U'=U$), the symmetry enlarges to $SU(6)$ and every one-body bilinear shares one
resolvent function after normalization by $\operatorname{tr}(T^aT^b)$, i.e.
$s_0=s_1=s_2=q_1=q_2$ (note's notation: $s$ = spin-channel, $q$ = charge-channel, subscript =
orbital $l$). $s_0=s_2=q_2$ is checkable today with the diagonal machinery alone (see the note's
Table 1); with the charge matrix resolvent, $q_1=s_1$ becomes reachable too, closing the loop and
**validating the spin- and charge-channel sign/normalization conventions against each other** —
something the existing $T^3$ vs $T^8$ check (both in `CHARGE`) cannot do, since it never touches
the spin-channel normalization.

The same orbitally-non-degenerate-ground-state caveat as the $T^3=T^8$ check (note §5.1: `Wigner
Eckart` with multiplicity one requires the ground state not be orbitally degenerate) applies to any
charge-channel quintet agreement, and should be checked (ground-state $L$, dimension) before
reading a mismatch there as a code bug either.

### Tests to mirror

Every Stage 1/2 test added for the spin path (occupied-orbital sign case, adjoint
$N_{\mu\nu}^\dagger=N_{\nu\mu}$, diagonal-trace-parallel-to-$\psi_0$ in place of the null-direction
test, diagonal-block vs `RunResolventWeighted`+`DiagChannel::Charge` — already supported via
`make_orbital_cartan_weights` — sum rule, `subtract_mean` pole cancellation) has a direct
charge-channel analogue and should be added alongside it, plus the $J=0$ $SU(6)$ cross-channel
identity above once both channels are implemented.

---

## Optional later: symmetry-adapted seeding

When the point group *is* known and trusted, seeding with symmetry-adapted combinations reduces
work: the block structure is then imposed rather than discovered, $r$ drops per block, and the
$1$-dimensional blocks fall back automatically to the single-vector continued fraction
(`bandlan.cxx:242-261`), which is today's tested path.

Worth noting for that route: $H$ real symmetric $\Rightarrow$ $G=[z-(H-E_0)]^{-1}$ is complex
*symmetric*, so $a^{T}Gb=b^{T}Ga$ and polarization is **three-term**,
$R_{ab}=\tfrac12[Q_{a+b}-Q_a-Q_b]$, not the four-term identity in the `.tex`. For $D_{4h}$ that
gives 9 runs for 9 independent functions.

This is an optimization of a solved problem, not a prerequisite. Do not build it until the general
path works and profiling says it matters.

---

## Also available today, for free

The **diagonal-orbital block alone** needs no new code at all: $S_{\mu\mu}$ and every linear
combination of the $S_{\mu\mu}$ is diagonal in the determinant basis, so those seeds are just weight
vectors handed to the existing `evaluate_resolvent_diagonal` (`impurity_solver.hpp:407`) with
`DiagChannel::Spin`. Useful in two ways during development:

- As a **validation target** for Stage 2 — the diagonal block of the matrix resolvent must
  reproduce these to round-off, computed by a completely independent code path (convention 3 vs
  convention 1).
- As the **exactly-captured subset on ASCI**: diagonal operators never leave the determinant basis,
  so their capture fraction (§1.3) is identically $1$ even when the off-diagonal seeds lose norm.
  Comparing the two on the same ASCI run isolates the seed-truncation error from the Hamiltonian
  truncation error both share.

E.g. with three bands: `(1,1,0)`, `(0,0,1)`, `(1,1,1)`, `(1,-1,0)` in the Spin channel, and
`(1,-1,0)` in the Charge channel for the charge nematic of physics-plan §8.2.

---

## Cost

| | seeds $M$ | rank $r$ after deflation | band-Lanczos runs |
|---|---|---|---|
| 2 degenerate bands, 1 site | 4 | $\le 4$ | 1 |
| 3 degenerate bands, 1 site | 9 | $\le 9$ | 1 |
| 3 bands, $D_{4h}$ | 9 | $\le 9$ (expect block structure) | 1 |
| 3 bands, 2 sites | 36 | $\le 36$ | 1 |

One band-Lanczos run in every case, $r$ matvecs per iteration, $2r$ stored vectors. No new DMFT, no
bath refit, no self-consistency change, no MPI or CSR changes. The $n_{\rm imp}=6$ cluster is the
only case where $r$ is large enough to be worth profiling before committing.

---

## Verification

Tests go in `tests/dynamical_properties.cxx` — already listed in `tests/CMakeLists.txt:26`, so **no
CMake change**. Catch2 v2; mirror the exact-Lehmann case at `:352-429` (`RunResolventWeighted` /
orbital $T^3$): a small `n_imp=2, n_active=4` system, `dense_hamiltonian` (`:34-47`) →
`SelfAdjointEigenSolver` for the exact reference, compared with
`Approx().epsilon(1e-6).margin(1e-8)`.

**Stage 1**
1. Sign and target of `apply_spin_bilinear` on hand-built determinants via `make_det` (`:24-30`),
   using distinct weights so an index swap cannot pass — the trick already at `:243-265`.
2. Adjoint identity $S_{\mu\nu}^\dagger=S_{\nu\mu}$, i.e. $\mathcal{G}$ symmetric.
3. $\mu=\nu$ agreement with `apply_diagonal_operator` + `weighted_imp_value`, $w=e_\mu$ —
   cross-validates bit convention 1 against convention 3.
4. **Capture fraction** (§1.3): exactly $1$ on a full FCI basis; on a deliberately truncated
   determinant list, equal to a hand-computed reference. Assert it never throws.

**Stage 2**
5. **Exact Lehmann for the full matrix.** Build $R_{\mu\nu;\gamma\delta}$ from the dense
   eigendecomposition and compare against the deflate → `BandResolvent` → back-transform pipeline,
   element by element. The single most important test.
6. **Deflation correctness**: construct a case with a known null direction — $n_{\rm imp} =
   n_{\rm active}$ at $S_z^{\rm tot}=0$, where $\sum_\mu\ket{\phi_{\mu\mu}}=0$ exactly — and assert
   $r = M-1$ and that the back-transformed $R$ still matches the dense reference.
7. **Diagonal-block agreement** with `evaluate_resolvent_diagonal` on the same system.
8. **Sum rule**: zeroth moment of each element equals $\mathcal{G}$.

**Runtime, on a converged run**

```
cmake --build build --target macis_test -j
./build/tests/macis_test "Dynamical properties*"
```

(The existing `build/` tree looks incompletely configured — no `tests/` or `main/` subdirectories,
empty `CMAKE_BUILD_TYPE`. Expect to re-run configure, ideally `-DCMAKE_BUILD_TYPE=Release`.)

- The trace direction $\sum_{\mu,\gamma} R_{\mu\mu;\gamma\gamma}$ (summed over **both** indices) must reproduce the existing
  `Sz_resolvent.dat` up to the factor of 4. Free regression check; run it first.
- Degenerate-band case: assert the Gram matrix and $R$ show the expected degeneracies with no
  crystal field. Any splitting means a symmetry-broken bath — which this machinery will now show
  directly instead of silently corrupting a symmetry-adapted seeding.
- $D_{4h}$ case: confirm the 36 predicted zeros and 9 independent functions emerge.

---

## Critical files

| file | role |
|---|---|
| `include/macis/gf/dynamical_properties.hpp` | `apply_spin_bilinear` (1.1); existing `RunResolventGS:73`, `apply_diagonal_operator:118`, `weighted_imp_value:174`, zero-norm guard `:405-411` |
| `include/macis/gf/bandlan.hpp` | `BandResolvent:271` — primary path; `BandLan:146` memory/zero-vector behaviour `:164`, `:220-226` |
| `src/macis/gf/bandlan.cxx` | `BandResolvent:172` — QR `:192-196`, vector layout `:206`, S-matrix `:300-308`, resolvent assembly `:327-339` |
| `include/macis/impurity_solver.hpp` | `evaluate_resolvent_orbital_matrix` (2.3); existing `evaluate_resolvent_diagonal:407`, rotation guard `:417-438`, grid `:266`, writer `:286` |
| `include/macis/sd_operations.hpp` | `single_excitation_sign:594` (reuse), `generate_hilbert_space:550-566` (closure argument) |
| `include/macis/util/rdms.hpp` | `rdm_contributions_2:59-60` — accumulation idiom to mirror |
| `include/macis/gf/gf.hpp` | `BuildWfn4Lanczos:404-459` — structural template and vector layout; map pattern `:419-425`; `write_GF:498` |
| `main/run_asci_impsolv_dop.cxx` | `OPT_KEYWORD:87-90`, keywords `:548-570`, gate `:571`, calls `:630-658` |
| `src/macis/impurity_solver.cpp` | `SolveImpurityED:313`, `p.dets` fill `:372-373` |
| `tests/dynamical_properties.cxx` | `make_det:24`, `dense_hamiltonian:34`, mirror target `:352-429` |

---

## Review of the first implementation (2026-09-24)

Stages 1–2 plus the driver keyword are implemented as uncommitted changes. The algorithm matches
§2.1, and no correctness bug was found in what is there. The inline **Review notes** above cover
the plan itself. The implementation issues are below, most important first.

1. **`GF.DELTA_RESOLVENT` is silently ignored on this path.** The driver comment says the option
   applies to "every resolvent channel above", but `evaluate_resolvent_orbital_matrix` takes no
   `subtract_mean`. For odd electron counts (a doublet ground state) or `spin_dep` runs,
   $\langle S_{\mu\mu}\rangle\neq0$ and the elastic pole survives. Fix: before the Gram step,
   `seeds -= psi0 * (psi0.transpose() * seeds)`. $\mathcal G$ then becomes the fluctuation
   covariance and the sum rule still holds with it. Thread `subtract_mean` through and append
   `delta_suffix` to the label.
   **Fixed:** `RunResolventOrbitalMatrix` and `evaluate_resolvent_orbital_matrix` now take
   `subtract_mean`, and the driver passes `delta_resolvent` with label `"Sz" + delta_suffix`, so
   the output goes to `Sz_delta_orbital_resolvent.dat` / `Sz_delta_gram.dat`. Capture fractions
   still describe the bare operator, since the subtracted component $\langle S\rangle\psi_0$ is
   in-basis.
2. **Stage 1 was only validated against itself.** The original Lehmann test built its reference
   overlaps with `apply_spin_bilinear`, so a sign or index error would cancel out. The original
   hand-built case uses adjacent orbitals (sign always $+1$) and exercises only the spin-up term.
   *Addressed by the new tests below.*
3. **Tests from §Verification were missing** (#2 adjoint, #4 fractional capture, #6 deflation,
   #7 diagonal block, #8 sum rule). The `gram == gram^T` check is tautological for
   $\Phi^T\Phi$. *Now added, see below.*
4. **A doc comment is misplaced.** The new code was inserted between `apply_diagonal_operator`'s
   doxygen block and its function, so that comment now documents `apply_spin_bilinear`. The new
   functions and the `GF.SPIN_ORB_MATRIX_RESOLVENT` keyword have no docs, unlike their neighbours.
5. **Duplicated pass and memory use.**
   - `spin_bilinear_capture_fraction` repeats the loop in `apply_spin_bilinear`. In-basis images
     are already in the seed, so only *leaked* images need the map:
     `capture = |seed|^2 / (|seed|^2 + leaked)`, from one pass that returns both.
   - Diagonal pairs have capture $\equiv 1$ and can be skipped.
   - `seeds`, `psi` and `vecs` are three dense $L\times M$ copies, about 3 GB each at
     $L=10^7$, $M=36$. Free `seeds` once `psi` exists, or write `vecs` directly.
6. **Minor.**
   - The orb_rot guard is copied from `evaluate_resolvent_diagonal`; factor it into `detail::`.
   - The output label `"Sz"` yields `Sz_gram.dat`; consider `"Smunu"`.
   - The long-format file has $n_\omega M^2$ rows (about 1.3M at $M=36$, $n_\omega=1000$).

### Tests added (`tests/dynamical_properties.cxx`)

Shared helpers were added to the anonymous namespace: `orbital_matrix_integrals`,
`half_filled_fci_dets`, `spin_bilinear_seeds` and `lehmann_element`.

| test case | covers |
|---|---|
| `spin bilinear signs across an occupied orbital` | Stage 1 #1: $S_{20}$ hopping over occupied orbital 1 in both spin blocks (sign $-1$, opposite overall signs from the spin-down term), the reverse hop $S_{02}$, and a fractional capture of exactly $1/9$ where two sources interfere on one leaked image. Squaring before accumulating would give $0.18/0.92$ instead. |
| `spin bilinear adjoint on the FCI space` | Stage 1 #2: $S_{\mu\nu}^T=S_{\nu\mu}$ as $36\times36$ matrices, independent of any wave function |
| `orbital matrix resolvent deflates the exact null direction` | Stage 2 #6: $n_{\rm imp}=n_{\rm active}=4$, asserts $\sum_\mu S_{\mu\mu}\ket{\psi_0}=0$, $r=M-1=15$, and an element-by-element Lehmann match |
| `orbital matrix diagonal block vs RunResolventWeighted` | Stage 2 #7: $R_{\mu\mu;\mu\mu}=4R_w[e_\mu]$ and $R_{00;00}+R_{11;11}+2R_{00;11}=4R_w[(1,1)]$ in the Spin channel. This path shares no code with Stage 1 (conventions 3 vs 1). |
| `orbital matrix resolvent subtract_mean cancels the elastic pole` | item 1: on a (2α, 1β) doublet with $\langle S_{\mu\mu}\rangle\neq0$, ${\rm plain}-{\rm delta}=m_km_l/z$ for every element, and the fluctuation Gram equals $\mathcal G-mm^T$ |
| `orbital matrix resolvent sum rule` | Stage 2 #8: ${\rm Re}[zR(z)]=\mathcal G$ at $z=10^6 i$ (the $M_1/z$ term is purely imaginary there), plus `result.gram` equal to an independent $\Phi^T\Phi$ |

**Not yet compiled or run.** This machine has no MPI (`cmake` fails at
`find_package(MPI)` with `MACIS_ENABLE_MPI=ON`). The `r = M-1` assertion assumes the 15
remaining seed directions are independent. That holds generically for this Hamiltonian, which has
no spatial symmetry, but it has not been confirmed numerically.
