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
approximation. **That null space is the symmetry content of the problem, measured rather than
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

Three edits mirroring the `GF.ORB_RESOLVENT` block (`:557-558`, `:571`, `:640-654`):

1. `bool orb_matrix_resolvent = false;` + `OPT_KEYWORD("GF.ORB_MATRIX_RESOLVENT", ..., bool)` near
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

- The trace direction $\sum_\mu R_{\mu\mu;\gamma\gamma}$ must reproduce the existing
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
