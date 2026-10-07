# Plan: capture-gated basis expansion for the orbital matrix resolvent

> **STATUS: IMPLEMENTED (2026-10-07), key `GF.ORB_EXPAND_BASIS`.** Differences from the design below:
>
> - `get_GF_basis_AS_1El` is **not** refactored (§A). The growth is a separate
>   `grow_basis_by_singles` in `dynamical_properties.hpp` that mirrors its loop, so the GF path
>   cannot change and test 1 is not needed. It also never adds or grows from a `base_dets`
>   determinant. `trunc_size = 0` means no cap.
> - §B is `orbital_bilinear_image` (`OrbitalBilinearImage`); the max-|b| merge is
>   `merge_leaked_images`.
> - When the basis was expanded, `capture_expanded` is recomputed for **every** pair, not only
>   the marked ones.
> - Tests 2–7 are in `tests/dynamical_properties.cxx`.
> - End to end: a 2-band J = 0 model with an orbital-diagonal bath (8 orbitals, ASCI filling the
>   1296-determinant ground sector). With expansion, every element in both channels matches the
>   CAS run to 1e-14. Without it, the off-diagonal elements are 0 (CAS: |R| ≈ 3). With
>   `GFSEEDTHRES = 1E-3`, the off-diagonal error is 2.8e-1 at `TOT_SD = 1`, 5.9e-6 at
>   `TOT_SD = 2`, and 6e-15 at `TOT_SD = 3`. Converge `TOT_SD` in production.
> - Not done: the 2-rank MPI hang (§Open issues) is still undiagnosed, so only single-rank runs
>   are verified.
> - The singles growth also adds determinants in sectors that no seed reaches. They are
>   harmless, since H has no matrix elements between sectors, but they enlarge the basis.
>   `TRUNC_SIZE` caps them.

## Context

The orbital-resolved spin resolvent (`PLAN_orbital_resolved_susceptibility.md`, Stages 1–2) projects
each seed $S_{\mu\nu}\ket{\psi_0}$ onto the ASCI determinant basis and reports the kept norm as the
**capture fraction** (§1.3 of that plan, `spin_bilinear_captured_fraction`,
`include/macis/gf/dynamical_properties.hpp:156`). That plan's §1.4 held back a basis-expansion
fallback "until the diagnostic says it is needed". Real calculations now say so: at Kanamori
$J=0$ the capture fraction is **exactly zero** for some off-diagonal pairs $\mu\neq\nu$, and it
stays exactly zero at $J>0$ (§Diagnosis, "$J>0$: parity survives").

### Diagnosis

At $J=0$, density-density Kanamori has no spin-flip or pair-hopping term. When the one-body part
(bath hopping, hybridization, crystal field) does not connect orbitals $\mu$ and $\nu$, $H$
conserves the electron count $N_{\mu\sigma}$ of each orbital *flavor* separately (impurity orbital
plus the bath orbitals attached to it, per spin). Then:

- ASCI grows its basis only through $H$-connected determinants, so every ASCI determinant lies in
  the ground state's flavor sector $(\ldots, N_\mu, \ldots, N_\nu, \ldots)$.
- $S_{\mu\nu}$, $\mu\neq\nu$, moves one electron from flavor $\nu$ to flavor $\mu$. Every image
  lies in the sector $(N_\mu+1, N_\nu-1)$, which ASCI never explored, so the capture fraction is
  exactly $0$.
- On CAS/ED the basis contains every sector, so the capture fraction stays $1$.

The pairs that go to zero are exactly the $(\mu,\nu)$ no one-body term connects. That is the first
thing to confirm against the integrals of the failing run (§Verification, step 0).

#### $J>0$: parity survives

Spin-flip and pair-hopping break the per-flavor, per-spin counts, but not all of the conservation.
With the same orbital-diagonal one-body part:

- Spin-flip $c^\dagger_{\mu\uparrow}c_{\nu\uparrow}c^\dagger_{\nu\downarrow}c_{\mu\downarrow}$
  changes $N_{\mu\uparrow}$ and $N_{\mu\downarrow}$ by $\pm1$ but leaves $N_\mu$ unchanged.
- Pair-hopping $c^\dagger_{\mu\uparrow}c^\dagger_{\mu\downarrow}c_{\nu\downarrow}c_{\nu\uparrow}$
  changes $N_\mu$ by $2$.

So $H$ still conserves the flavor parity $\Pi_\mu=(-1)^{N_\mu}$ of every flavor. In the $O(2)$
two-band case this is the reflection $c_\nu\to-c_\nu$. The same chain of argument goes through with
sectors labelled by parities instead of counts:

- Every ASCI determinant has the ground state's parities.
- $X_{\mu\nu}$ ($S$ or $N$, $\mu\neq\nu$) changes $N_\mu$ and $N_\nu$ by $\pm1$, so it flips
  $\Pi_\mu$ and $\Pi_\nu$. Every image falls outside the basis, and the capture is again exactly $0$,
  not just small. Adding more determinants of the same parity cannot fix it.
- The diagonal bilinears preserve every $\Pi_\mu$, which is also the physical reason the
  diagonal $\leftrightarrow$ interorbital cross elements vanish. On ASCI those zeros are therefore
  **guaranteed**, so they cannot test anything.

Observed 2026-10-06 on the two-band Kanamori run `…/2bands/Doping/selected_calculations/J_0.2/U_27.00`
($U=27$, $J/U=0.2$, Irrep bath, 2.5 M ASCI determinants, `NROTS = 0`). All determinants in `wfn.out`
have even $N$ in flavor 1. The capture fraction is $0$ for $(0,1)$ and $(1,0)$ in both the spin and
charge channels, and the whole interorbital block is identically zero.

The parity argument needs no $J=0$ assumption. It fails only if a one-body term connects $\mu$ and
$\nu$ (interorbital hybridization, bath sites shared between flavors), or if the determinant basis
uses orbitals that mix flavors, such as a natural-orbital rotation of the bath across flavors.

### Why this matters

A zero-capture seed becomes a zero column of the Gram matrix. Deflation discards it and the output
reports $R_{\mu\nu;\cdot}=0$. That zero is a basis artifact, but the Stage 3 symmetry report would
read it as a symmetry-imposed zero.

- **Degenerate orbitals and bath.** At $J=0$ the symmetry is $SU(2n)$ and the missing element is
  recoverable:
  $R_{\mu\nu;\mu\nu}=\tfrac12\,(R_{\mu\mu;\mu\mu}+R_{\nu\nu;\nu\nu}-2R_{\mu\mu;\nu\nu})$.
- **Degenerate orbitals at $J>0$.** The $SU(2n)$ identity no longer holds, so nothing can be
  recovered. With two bands and $O(2)$ symmetry, $r_1=C-D$ (the $A_2$ channel, likewise $q_1$)
  exists only in the interorbital block. $C+D$ is the only independent route to $r_2$, so the
  $O(2)$ check $A-B=C+D$ cannot be made. A zero $C,D$ makes that check fail on every frequency.
  This is a basis artifact, not a symmetry violation.
- **Non-degenerate orbitals** (crystal field, $D_{4h}$ with $xy$ split off). The inter-orbital
  particle–hole response carries independent information and is simply missing.

The fix is to expand the basis **only for the pairs whose capture fraction falls below
`GF.ORB_MIN_CAPTURE`**. The Green's-function path already does this for $c^\dagger\ket{\psi_0}$,
which also leaves the ground-state sector.

---

## Why `get_GF_basis_AS_1El` cannot be called as-is

`get_GF_basis_AS_1El` (`include/macis/gf/gf.hpp:236-380`) does two things:

1. **Seed images** (`:280-298`, amplitudes `:311-335`). For one spin-orbital, it collects
   $c^\dagger_{i\sigma}D$ or $c_{i\sigma}D$ for every $D$ in the ground-state basis. This is a
   one-operator, particle-number-changing image with `GetInsertion*Sign` phases. The resolvent
   needs $\sum_\sigma \pm c^\dagger_{\mu\sigma}c_{\nu\sigma}D$, so this step does not apply.
2. **Growth** (`:337-373`). Starting from the image determinants with $|b|\ge$ `GFseedThres`, it
   adds `tot_SD` layers of active-space singles (`generate_singles_spin_as`,
   `sd_operations.hpp:302`) until about `trunc_size` determinants. The active space is the set of
   orbitals with `asThres <= occs[i] <= 1 - asThres`. This step does not depend on the operator and
   is what we reuse.

The capture pass already produces a better step 1. `spin_bilinear_captured_fraction` builds
`images`, a map from every image determinant to its accumulated amplitude, using
`single_excitation_sign`. The leaked entries (images not in `det_index`) are exactly the seed
determinants and $|b|$ values that step 2 needs. Today the map is discarded after the norm ratio
is computed.

---

## Design

### A. Factor out the growth step — `include/macis/gf/gf.hpp`

```cpp
// Grows `found` (with its lookup `found_pos`) by `settings.tot_SD` layers of
// active-space single excitations. Layer 1 starts only from seeds[i] with
// |amplitudes[i]| >= settings.GFseedThres. Stops once the basis passes
// settings.trunc_size.
template <size_t nbits>
void grow_basis_by_singles(std::vector<std::bitset<nbits>> &found,
                           bitset_map<nbits> &found_pos,
                           const Eigen::VectorXd &amplitudes,
                           const std::vector<uint32_t> &as_orbs,
                           size_t norbs, const GFSettings &settings);

std::vector<uint32_t> active_space_orbitals(const std::vector<double> &occs,
                                            double asThres);
```

`get_GF_basis_AS_1El` keeps its step 1 and calls these two helpers. **Its behaviour must not change.**
Keep the loop bounds `cgf <= trunc_size`, the `nSD == 1`-only seed thresholding, and the insertion
order exactly as they are today, including the layer-by-layer `startSD`/`endSD` walk.

### B. Return the leaked images — `dynamical_properties.hpp`

Merge the two per-pair passes (this is also review item 5 of the susceptibility plan). Rename
`spin_bilinear_captured_fraction` or add an overload that returns

```cpp
template <size_t nbits>
struct BilinearImage {
  double capture;                            // |P phi|^2 / |phi|^2, as today
  std::vector<std::bitset<nbits>> leaked;    // images outside det_index
  Eigen::VectorXd leaked_amplitude;          // their accumulated coefficients
};
```

Diagonal pairs never leak (capture $\equiv 1$), so skip them.

### C. Gated expansion — inside `RunResolventOrbitalMatrix`

After the capture pass and before building the seeds:

1. **Gate.** Mark pair $(\mu,\nu)$ for expansion if `capture < settings.orb_min_capture` **and**
   `settings.orb_expand_basis` is set. With the gate off, nothing below runs and the result must
   be bit-identical to today's.
2. **Seeds for growth.** Take the union of `leaked` over the marked pairs, merging duplicates.
   Different pairs can leak into the same determinant, so combine them as $\max|b|$ across pairs,
   **not** as a sum: amplitudes from different operators must not interfere.
3. **Grow.** Call `grow_basis_by_singles` on that union, with `as_orbs` from `p.occs` (spatial
   occupations halved, `impurity_solver.cpp:377`, the same input the Green's-function path uses).
4. **One shared basis.** `expanded = base_dets` **followed by** the new determinants, with duplicates
   of `base_dets` removed. Keeping `base_dets` first means $\psi_0$ is just zero-padded:
   `wfn0_ext = [wfn0; 0]`.
   - Band Lanczos needs one Hamiltonian, so every pair (expanded or not) is seeded in this one
     basis.
   - Mixing sectors is harmless. $H$ has no matrix elements between them, seeds in different
     sectors are orthogonal, and their Gram cross-block is exactly zero. **But** the growth also
     adds determinants to $\psi_0$'s own sector, which is not harmless; see §Warning.
   - `subtract_mean` stays correct because $\psi_0$ is still represented exactly.
5. **Seeds on the expanded basis.** Call the existing `apply_spin_bilinear(wfn0_ext, expanded,
   expanded_index, mu, nu)`. The rest of the pipeline (Gram, deflation, `BandResolvent`,
   back-transform) is unchanged. The CSR build (`dynamical_properties.hpp:274`) simply runs over
   `expanded`.
6. **Report capture twice.** Report `capture_base` (the diagnostic, as today) and
   `capture_expanded` (after growth). Every marked pair's image seed determinants are in the basis
   by construction, so `capture_expanded == 1` for each marked pair. Assert it.

### D. Settings and driver

- `GFSettings`: add `bool orb_expand_basis = false;`. Reuse `trunc_size`, `tot_SD`, `GFseedThres`,
  `asThres` and `norbs` rather than adding a parallel set.
- **`norbs` defaults to 0** (`GFSettings`), and `generate_singles_spin_as` with `norbs = 0`
  generates nothing. When it is unset on this path, fall back to `p.n_active`, and log that it did.
- `trunc_size` must bound the **union** of all marked pairs, not each call. Check it against
  `expanded.size() - base_dets.size()`.
- Driver (`main/run_asci_impsolv_dop.cxx:617-623`): `OPT_KEYWORD("GF.ORB_EXPAND_BASIS", ...)`
  next to `GF.ORB_MIN_CAPTURE`.
- `impurity_solver.hpp:536-541`. Keep the warning, but when expansion is on, say that the pair
  was expanded and print both capture numbers.

### E. Output

- `<label>_gram.dat`: add a `capture_base capture_expanded expanded(0/1)` column per pair, plus
  the base and expanded basis sizes.
- If a pair is still below threshold after expansion (only possible with the gate off), write its
  resolvent rows but flag them as `unresolved`. Never let them pass silently as zeros into the
  symmetry report.

---

## Warning: the growth step re-enters the ground-state sector

Design C.4 assumes the growth adds determinants only to the target sectors. It does not.

- `generate_singles_spin_as` (`sd_operations.hpp:302`) generates every single excitation among
  `as_orbs`, whatever the flavor. On a doped run almost every orbital passes `asThres`. In
  `J_0.2/U_27.00` the occupations run from $1.1\times10^{-3}$ to $0.995$ against
  `asThres = 1e-4`, so all 16 orbitals are active.
- A single that moves an electron back from flavor $\mu$ to flavor $\nu$ returns a target-sector
  determinant to $\psi_0$'s sector. This already happens in the first layer, at any $J$
  (including $J=0$).
- ASCI truncated $\psi_0$'s sector, so most of these determinants are not in `base_dets`. They
  enter `expanded` as new determinants in $\psi_0$'s sector.

With the gate on, the consequences hold even when every marked pair reaches
`capture_expanded = 1`:

1. **The diagonal block changes.** The diagonal seeds live in $\psi_0$'s sector, so their Lanczos
   now runs over a larger basis than ASCI's. $A$ and $B$ (hence $r_0, r_2, q_0, q_2$) move away
   from the gate-off values, although no diagonal pair was marked.
2. **$\psi_0$ is no longer an eigenvector.**
   - $\psi_0$ diagonalizes $H$ projected onto `base_dets`, not onto `expanded`.
   - The new determinants are singles or doubles of `base_dets`, so $H$ couples them to $\psi_0$.
   - The spectrum of $\psi_0$'s sector on `expanded` can then fall below $E_0$. That puts poles at
     $\omega<0$ and gives the fluctuation seeds weight on a state lower than $\psi_0$, which is
     unphysical at $T=0$.
   - The "$\psi_0$ is still represented exactly" argument of C.4 is true but not sufficient.
3. **The planned tests cannot see it.** Tests 2–4 use the *complete* FCI ground-state sector as
   the "ASCI" basis, so the growth finds no new determinant in $\psi_0$'s sector. Test 5 covers
   only gate-off.

**Fix.** Use one or both of the following:

- **Separate Lanczos runs per sector (preferred).**
  - Run the unmarked pairs on `base_dets` exactly as today, and the marked pairs on the grown
    basis.
  - The block between the two sectors is exactly zero by symmetry, so nothing is lost.
  - The diagonal block is bit-identical to gate-off by construction.
  - This replaces the "one shared basis" of C.4, at the cost of a second CSR build and Lanczos run.
- **Filter the grown basis by sector.**
  - Drop every grown determinant whose flavor parities match $\psi_0$'s.
  - Parities are conserved at every $J$. At $J=0$ the filter also removes other count sectors of
    the same parity, but no seed reaches those.
  - Flavors are the connected components of the one-body hopping graph.
  - This alone makes the shared basis of C.4 safe. Combined with separate runs, it only saves
    memory, because the marked seeds never reach $\psi_0$'s sector anyway.

If the gate fires on a pair whose seed stays in $\psi_0$'s sector (a truncation leak with
$0<$ capture $<1$, possible when a one-body term connects $\mu$ and $\nu$), neither fix applies:
there is no sector to separate. Then either re-diagonalize $\psi_0$ on `expanded`, or leave that
pair unexpanded and report its capture.

**Test** (test 8 under §Verification): a truncated base. Keep the top-$k$ determinants of the
ground-state sector and re-diagonalize there, so $\psi_0$ is an exact eigenvector of the truncated
problem, as with ASCI. Then, with the gate on, assert that:

- the diagonal block equals the gate-off result to machine precision;
- $\psi_0$'s sector has no pole below $E_0$.

---

## Cost

- **Sectors.** Each off-diagonal pair lands in its own sector $(N_\mu+1, N_\nu-1)$. At $J=0$
  with 3 orbitals, up to 6 extra sectors, each with its own growth. At $J>0$ the target is a
  parity sector, and $(\mu,\nu)$ and $(\nu,\mu)$ flip the same two parities, so they share one.
  That gives 1 extra sector for 2 orbitals and up to 3 for 3 orbitals.
- **Size.** Expect the same order of magnitude as a Green's-function basis per pair, in total
  capped by `trunc_size`. The CSR build over `expanded` dominates, as in the Green's-function
  path.
- **Band Lanczos.** $r$ is unchanged, since the number of seeds is still $M$. The vector length
  grows to `expanded.size()`, and so does memory ($2r$ vectors, plus three dense $L\times M$
  seed copies; see review item 5).

---

## Accuracy

This is the same level of approximation as the Green's-function basis, not an exact result.

- Once the image determinants are in the basis, the zeroth moment (sum rule, $\mathcal G$) is
  exact.
- Response inside the new sector depends on how well the added singles layers describe
  relaxation there. ASCI never optimised that sector.
- With `tot_SD = 2` and `GFseedThres = 0`, the basis contains every determinant $H$ reaches from
  the seed in one application, so the first moment $\braket{\phi|H|\phi}$ is exact too. Use this
  as a check: compare `Re[z^2 R - z G]` at large $|z|$ against $\braket{\phi|H-E_0|\phi}$ computed
  directly.
- Converge the physics by increasing `tot_SD` and `trunc_size` until the expanded elements stop
  changing, and report that.

---

## Verification

Tests go in `tests/dynamical_properties.cxx` (already in `tests/CMakeLists.txt`). Build and
run on a single rank:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build macis_test
./build/tests/macis_test "Dynamical properties*"
```

**0. Confirm the diagnosis on the failing run.** Before writing code, check the production
integrals: are the zero-capture pairs exactly those that no one-body term links? Then check the
ASCI determinants (`wfn.out`):

- At $J=0$, is $(N_{\mu\uparrow}, N_{\mu\downarrow})$ per flavor identical on every determinant?
- At $J>0$ these counts legitimately vary, so a "no" here does **not** rule the plan out. Check
  instead that the flavor parity $(-1)^{N_\mu}$ is identical on every determinant.

If neither holds, the cause is something else and this plan does not apply.

**Unit tests**

1. **Refactor regression.** No Catch2 test covers `get_GF_basis_AS_1El` today; only the drivers
   use it. Before refactoring, add one that snapshots its output (determinant list, in order) on
   the `n_imp=2, n_active=4` test system, for particle and hole and for both spins. After
   refactoring, the output must be identical.
2. **Reproduce the zero.** Build a $J=0$ density-density model with an orbital-diagonal bath
   (`n_imp=2`, `n_active=4`, each impurity orbital hybridised only with its own bath orbital).
   Take the FCI determinants filtered to the ground state's flavor sector as the "ASCI" basis.
   Assert that the capture fraction is exactly $0$ for $\mu\neq\nu$ and exactly $1$ for
   $\mu=\nu$. This pins the diagnosis in a test. Repeat at $J>0$, with spin-flip and
   pair-hopping switched on, filtering the FCI determinants by the ground state's flavor
   **parities** instead of its counts. The capture must still be exactly $0$.
3. **Expansion recovers the exact answer.** Same system, gate on, `tot_SD` large enough to fill
   the (small) target sectors. Assert `capture_expanded == 1` and compare every $R$ element with
   the dense Lehmann reference on the full FCI space (`lehmann_element`,
   `Approx().epsilon(1e-6).margin(1e-8)`). Run it on both the $J=0$ and the $J>0$ systems of
   test 2.
4. **$SU(4)$ identity.** Degenerate version of test 3 (equal on-site energies and hybridisations,
   $U'=U$, $J=0$). Assert
   $R_{01;01}=\tfrac12(R_{00;00}+R_{11;11}-2R_{00;11})$. First check that the ground state is
   non-degenerate: at $J=0$ it can be degenerate across flavor sectors, and then the identity
   does not hold for a single state.
5. **Gate off, or all captures above threshold → bit-identical output** to the current
   `RunResolventOrbitalMatrix` on the existing orbital-matrix tests.
6. **Growth seeds do not interfere.** Construct two pairs that leak into the same determinant with
   opposite amplitudes. Assert that the determinant is still grown from ($\max|b|$, not the sum).
7. **`norbs = 0` fallback** gives the same basis as `norbs = n_active`.
8. **Truncated base, gate on** (§Warning). With a top-$k$ base and $\psi_0$ re-diagonalized on it,
   the diagonal block equals the gate-off result to machine precision, and $\psi_0$'s sector has
   no pole below $E_0$.

**Production**

- Rerun the failing $J=0$ case with `GF.ORB_EXPAND_BASIS = true`. Every pair should report
  `capture_expanded = 1`.
- Degenerate bands: the $SU(2n)$ identity should hold within the `tot_SD` / `trunc_size`
  convergence.
- Small $J>0$: the expanded result should connect continuously to $J=0$.
- $J>0$, two degenerate bands (the `J_0.2/U_27.00` case under §Diagnosis): $(0,1)$ and $(1,0)$
  should report `capture_expanded = 1`. $C$ and $D$ should be nonzero. The $O(2)$ identity
  $A-B=C+D$ should hold within the `tot_SD` / `trunc_size` convergence, provided the bath is
  orbital-symmetric.

---

## Open issues

- **2-rank MPI hang.** `mpirun -np 2 ./macis_test "Dynamical properties*"` hung (killed after
  10 min, no output); a single rank passes all 17 tests. Not yet diagnosed. The expansion makes
  bases larger and multi-rank production runs more likely, so resolve this before relying on it.
  The expanded basis is built identically on every rank (deterministic `std::map` order), so no
  broadcast is needed, but confirm that once the hang is understood.
- **Degenerate $J=0$ ground state.** If the ground state is degenerate across flavor sectors,
  ASCI returns one member of the multiplet. Every element, including the diagonal block, then
  comes from a symmetry-broken state. That is a separate issue from capture; check the low-lying
  spectrum before interpreting $J=0$ results.
- **Charge channel.** Once `apply_orbital_bilinear` takes a `DiagChannel`
  (`PLAN_orbital_resolved_susceptibility.md` §Extension), $N_{\mu\nu}$ leaks into the same
  sectors for the same reason. The expansion applies unchanged.

---

## Critical files

| file | role |
|---|---|
| `include/macis/gf/gf.hpp` | `get_GF_basis_AS_1El:236` — split out `grow_basis_by_singles` (`:337-373`) and `active_space_orbitals` (`:271-274`) |
| `include/macis/gf/dynamical_properties.hpp` | `spin_bilinear_captured_fraction:156` → return leaked images; `RunResolventOrbitalMatrix:199` — gate, expand, zero-pad $\psi_0$, CSR over `expanded` (`:274`) |
| `include/macis/sd_operations.hpp` | `generate_singles_spin_as:302` (reused as-is) |
| `include/macis/impurity_solver.hpp` | `evaluate_resolvent_orbital_matrix` — warning `:536-541`, writer |
| `main/run_asci_impsolv_dop.cxx` | `GF.ORB_EXPAND_BASIS` next to `:622` |
| `src/macis/impurity_solver.cpp` | `p.occs` (spatial occupations halved, `:377`) |
| `tests/dynamical_properties.cxx` | tests 1–7 above |
