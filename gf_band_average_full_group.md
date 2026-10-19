# GF_BAND_AVERAGE: average over the whole group, not one element per parity image

**Proposal, 2026-10-10. Not implemented.** Concerns `band_orbit_average_gf`
(`include/macis/impurity_solver.hpp:189-345`, at `50136d6`). Reproducible check:
`gf_degenerate_average_toy_ed.py` (next to this file, runs in about 1 s).

## Recommendation

In `band_orbit_average_gf`, replace the average over **one group element per image of the
winner's parity key** with the average over **every element of the `SYMMETRIZE_DETS` group**:

    G(a,b)  <-  1/|G| * sum_{g in G} G(g^-1 a, g^-1 b)

Drop the early return for an invariant key (`if(reps.size() <= 1) return;`, `:278`). Apply the
same average in the no-parity-labels branch (`:199-225`), which today only warns. The cost is
|G| = 6 index permutations of an n × n matrix per frequency, which is negligible.

The current code averages over coset representatives only. That equals the orbit average only when
the solved state is invariant under the **stabilizer** of its parity key. At J = 0 that assumption
fails in two ways, and both occur in production:

- **(A1) Invariant key, broken state.** At J = 0 there is no pair hopping, so the count in each
  band is conserved (`parity_sectors.cpp:208` already warns `counts_conserved`). An (e,e,e) sector
  then holds three degenerate count subsectors, [8,6,6], [6,8,6] and [6,6,8], which the group
  permutes. Davidson returns one of them. The key is invariant, so `reps.size() == 1` and the code
  returns as if the state were symmetric.
- **(A2) Odd key whose stabilizer breaks the state.** Take a J = 0 member such as |0↑ 1↓⟩ in
  (o,o,e). The stabilizer swap 0↔1 maps it to |1↑ 0↓⟩, a *different* degenerate state with the
  same key. Averaging over coset representatives then gives a band-asymmetric GF whose asymmetry
  depends on which element represents each coset.

`check_rdm_symmetry` (`impurity_solver.cpp:304-321`) catches A1 (deviation ≈ 2). It cannot see
A2, because it compares the spin-summed n_active × n_active 1-RDM, and that matrix is symmetric
under 0↔1 for |0↑ 1↓⟩.

## Why the full-group average is the right answer, not just a symmetric one

At J = 0 the Kanamori interaction is (U/2)N(N−1), and with a band-diagonal, band-identical bath
(what `prepare_det_symmetry` verifies on T, Td and V) the Hamiltonian is U(2·nbands)-invariant. With
a closed-shell bath, the ground manifold at the impurity d² is the 15-dimensional irrep. Within the
Sz = 0 solve (NALPHA = NBETA), its 9 members form a single irrep of U(3)↑ × U(3)↓.

- The operator Σ_a c_a↑ (z − H)⁻¹ c†_a↑ (plus its hole part) is invariant under that group.
- By Schur's lemma, the **band-summed up-spin GF has the same value for every degenerate member**.
- It equals the band trace of the equal-weight T → 0⁺ ensemble GF, which is ∝ 1 in flavour space.
- Averaging over all of S3 makes the diagonal uniform, so it returns exactly that ensemble GF,
  **whichever degenerate member Davidson returned and whichever tied parity sector won.**

The coset average does not have this property.

### Toy ED (`gf_degenerate_average_toy_ed.py`)

The model has 3 bands, J = 0, U = 8, µ = 12, and one bath level per band at ε = −1.5 with V = 0.6.
The ground state is N = 8 with an exactly 15-fold degeneracy. Errors are max over a and iωₙ of
|G_a↑ − g_ens| / max|g_ens|:

| member | parity key | raw (today, invariant key) | coset average, all 8 choices of representatives | full S3 average |
|---|---|---:|---|---:|
| 0↑0↓ (doublon) | (e,e,e) | 3.4 | not averaged (m = 1) | **1e-14** |
| 0↑1↓ | (o,o,e) | 3.4 | 2.5e-14 … 1.7 (**2/8 exact**) | **3e-14** |
| 1↑2↓ | (e,o,o) | 3.4 | 1.0e-14 … 1.7 (**2/8 exact**) | **1e-14** |
| 0↑1↑ (Sz = +1) | (o,o,e) | 3.4 | 1.7 | 1.7 |

The last row is outside the Sz = 0 solve, and it is the case `GF.SPIN_AVERAGE` exists for. The
full-group average is exact only on Sz = 0 members; that is the even-N production case.

## Evidence from production runs

Paths are relative to `/leonardo_work/Sis26_collura/dfloreza/g100_move/Leonardo_move/3band/singlesite/Doping/Irrep/Nb15/J_0/`.
All runs use `SYMMETRIZE_DETS = TRUE` (group order 6), `PARITY_SOLVE` on and `NROTS = 0`.

**"GF asym"** is `max|G − P(G)| / max|G|` for the orbital projection P of `inout.symmetrize_fct`
(band-diagonal average). It is evaluated on the GF that DMFT actually used (`It_N/GF.dat`, or
`GF_raw.dat` where DMFT projected it afterwards).

### A1: (e,e,e) winner, nothing averaged. `U_60.00`, It_19, job 59847301 (2026-10-10)

Now archived in `U_60.00/hide_20261010_polarized/It_19/`.

| fact | where |
|---|---|
| `counts_conserved`: "a count-sector trap is still possible and is not handled" | `ASCI/asci.out:9` |
| 4 parity sectors at E = −159.2500629047, all within 2.8e-14; winner (e,e,e) with `N_band=[8,6,6]` | `ASCI/parity_sectors.dat` |
| `max 1-RDM symmetry deviation = 1.9965` in the winning sector; 1e-16 in the other three | `ASCI/asci.out:198-199` |
| "GF_BAND_AVERAGE: parity sector (e,e,e) tied with (o,o,e) (o,e,o) (e,o,o) … the GF is not averaged over those sectors"; no averaging line follows | `ASCI/asci.out:1091` |
| Occs per band per spin = 0.9985 / 0.00025 / 0.00025 | `ASCI/asci.out:775` |
| **GF asym = 0.753** (It_18, 18-electron sector, symmetric: 0.000) | `It_19/GF.dat` |
| NewHyb orbital asymmetry 0.750; Irrep fit stopped at BFGS maxit without converging | `output_59847301.out` |
| It_20 µ search crashed: `could not match band 0's pole at eps = -10.9415 … cost = 9.425e-01, hard_tol = 1.0e-01` (`Solver.py:253`) | `error_59847301.err` |

**Cost:** It_19 took 8 h 41 min, then the job died.

The impurity RDM shows the manifold *is* the single 15-dim irrep the argument above needs. The raw
entropy is S = 0.032. After the U(6) twirl (`~/Code/Comp_rho_reduced/rdm_flavor_average.py`) it is
S = 2.734 (ln 15 = 2.708), with 99.6% of the weight in the (1,1,0,0,0,0) irrep.

### A2: odd winner, averaged over m = 3 partners, still asymmetric

| run / iteration | winner | MACIS log | GF asym |
|---|---|---|---:|
| `U_60.00/hide/It_19` (job 59623228) | (o,o,e) | `m = 3 degenerate partners; max \|G - G_partner\| = 1.665e-01 (max \|G\| = 1.474e-01)` | **0.604** |
| `U_60.00/hide/It_23` | (o,e,o) | m = 3 | 0.591 |
| `U_65.00/It_21`, `It_23` | (o,e,o), (e,o,o) | m = 3 | 0.574, 0.581 |
| `U_70.00/It_19`, `It_20` | (o,o,e), (o,e,o) | m = 3 | 0.537, 0.507 |
| `U_35.00/It_11`, `It_12` (`GF_raw.dat`) | (e,o,o), (o,o,e) | m = 3 | 0.643, 0.641 |

In `U_60.00/hide/It_19`, the "averaged" diagonal at the first Matsubara point is
**−0.0191, 0.0364, 0.0919** for bands 0, 1, 2. MACIS's `GF.dat` and DMFT's `GF.dat` are
identical, so nothing happens in between.

`U_35.00` It_11 and It_12 have nearly the same bath (dHyb = 0.0012) but different winners. In
It_11 band 0 is empty (Occs 0.001/0.499/0.499); in It_12 band 2 is (0.499/0.499/0.001). Yet both
outputs have the **same** diagonal at iω₀: −0.035, 0.044, 0.124. The asymmetry does not follow the
solved state. It comes from which group elements represent the cosets, as the toy ED predicts.

### The same thing, (e,e,e) winners

`U_50.00` It_15 and It_17 have winner (e,e,e) and `N_band=[8,6,6]`, with 1-RDM deviations of 1.995
and 1.69. GF asym is **0.700** and **0.613**.

## What happens downstream without the change

The band-polarized GF matches no ensemble and enters the lattice Dyson equation, then NewHyb, then
the bath fit. The runs above show four outcomes:

1. **Crash.** `U_60.00` It_20 (above): the asymmetric fit cannot pass
   `band_permutation_generators` on the next solve.
2. **No convergence.**
   - `U_60.00/hide`, It_19–28: dHyb 0.09–0.87 for 10 iterations, about 17 h of walltime (7 h 40 min
     plus 9 iterations of about 1 h). The empty band changed from iteration to iteration.
   - `U_50.00`, It_16–21: dHyb 0.07–0.18.
   - `U_65.00`, It_21–23: dHyb 0.27, 0.17, 0.11.
   - `U_70.00`, It_19–20: dHyb 0.47, 0.34.

   These runs had `impose_orb_symmetry = .true.`, which projects the hybridization *after* the
   non-linear Dyson step. As `Solver.py`'s own comment at the `symmetrize_solver_output` branch
   says, that leaves a fit target no impurity state reproduces.
3. **Guard aborts.** With the DMFT-side GF projection on (`symmetrize_solver_output`), the
   `symmetrize_warn_thresh = 0.5` guard raised on `U_35.00` in jobs 59623209 (discarded weight
   0.552) and 59079466 (0.712). That guard exists to catch solver failures, and here it fires on a
   degeneracy.
4. **Workaround with the guard disabled.** `U_35.00` (job 59701118) converged to dHyb = 0.0000 at
   It_12 while discarding 0.64 of the GF every iteration. `U_60.00` was restarted on 2026-10-10 with
   `symmetrize_warn_thresh = 1.0`. Both work, because the band average is exact here. But they make
   the projection carry most of the result, so the discarded weight can no longer tell a degeneracy
   apart from a broken solve.

**Prevalence.** 11 of the 39 U points in the 3-band `Nb15/J_0` scan have iterations with
`std(Occs) > 0.05`, counting archived attempts. In the 2-band `Ulysses_move/.../Nb14/Irrep/J_0` scan
it is 17 of 23. I checked the mechanism iteration by iteration only for the 3-band runs listed above.

## Why in MACIS and not only on the DMFT side

The DMFT-side `symmetrize_solver_output` does the same algebra in its orbital channel, and at J = 0
it is just as exact. It is the wrong place for the fix, for four reasons:

- **MACIS knows the symmetry is exact.** `prepare_det_symmetry` verifies that the group leaves T,
  Td and V of the Hamiltonian actually solved invariant to `SYM_TOL = 1e-8`. The DMFT projector
  *assumes* the symmetry and applies it blindly, including to genuine failures and crystal-field
  cases.
- **It is opt-in per calculation and off in most of the affected runs** (`U_50`, `U_65`, `U_70`).
  Enabling it at J = 0 also requires disabling its guard.
- **Other consumers read MACIS's GF directly.** `GF.dat` inside `ASCI.tar.gz` feeds the
  real-frequency pass and the analysis tools, and today it is polarized.
- **The diagnostic gets its meaning back.** After the change, the DMFT-side discarded weight drops
  to noise level whenever the group is exact, so `grep "Symmetry projection"` works again as a
  detector of solver failure.

## Secondary recommendations

1. **Reword the 1-RDM warning** (`impurity_solver.cpp:318`). When the winner tied with other
   sectors, or `counts_conserved` holds, a deviation of about 2 means a degenerate member, not a
   Davidson failure. Say so, and refer to GF_BAND_AVERAGE instead of "check Davidson convergence".
2. **Print a group-averaged occupation line next to `Occs:`**, leaving `Occs:` itself unchanged
   because DMFT parses it. One-body occupations obey the same Schur argument, so the S3-averaged
   values are exact. The `std(Occs)` column in `dHyb.dat` would then report real symmetry breaking
   instead of which member Davidson picked.
3. **Keep the log format.** `check_dmft_health.py` matches
   `^GF_BAND_AVERAGE parity sector .*?m = (\d+)`. Print `m = |G|` (6) in the same line.

## What this does not fix

- **Which degenerate vector Davidson returns.** That is `PLAN_symmetric_asci_search.md` Step 11.
  At J = 0 the count per band *and spin* is conserved. Distinct members of an orbit therefore have
  different count vectors, the cross terms between them vanish, and the GF of an S3-projected vector
  equals the group average of the GF of any member. So for the GF this
  proposal gives the same result as Step 11, without touching Davidson.
- **The RDM.** An S3 average is not enough there: at `U_60.00` It_19, SU(2) + S3 gives S = 1.130
  (≈ ln 3), against the 2.734 U(6) result. Keep `rho.bin` raw and twirl it afterwards with
  `rdm_flavor_average.py`.
- **Sz ≠ 0 members** (odd N). `GF.SPIN_AVERAGE` covers these; see the last row of the toy table.
- **Manifolds that are not a single irrep** (accidental degeneracies). The full-group average is
  then the orbit ensemble, not the manifold ensemble. That limitation also applies to the current
  code, so the change is no regression.
- **Finite J with pair hopping.** The Hund triplet M = 0 member is invariant up to phase under its
  stabilizer, so the coset average is already complete there, and the full-group average gives the
  identical result. No change in behaviour is expected.

## Can the DMFT-side symmetrization flags be dropped?

**Not all of them.** The full-group average is a projector. Whenever it runs, the GF that MACIS
writes is exactly invariant under S_nbands, whichever degenerate member or tied sector Davidson
returned. That group acts on band labels only, and the average runs only under conditions that the
DMFT flags set up or back up.

| flag | after the change | why |
|---|---|---|
| `asci_symmetrize_dets` | **must stay on** | It is the switch. MACIS does not detect a symmetry; it only checks the group it is given (`prepare_det_symmetry`, `impurity_solver.cpp:168`). `band_permutation_generators` (`Solver.py:375`) builds that group and also projects the bath exactly onto the symmetric subspace before the FCIDUMP is written. Without this flag there is no group, and nothing is averaged. |
| `impose_orb_symmetry` | can be dropped | Its target, NewHyb, is already symmetric, so it does nothing. |
| `symmetrize_solver_output` | keep, as a tripwire | Its orbital channel does nothing on an averaged GF (discarded weight around 1e-15). See below. |
| `impose_site_symmetry` | keep for clusters | The generators permute bands and carry the site index along. Permutations of cluster sites are not in the group. |
| `impose_ph_symmetry` | keep at half filling | Particle–hole symmetry is not a permutation of orbitals. |

**Why keep `symmetrize_solver_output`.** `band_orbit_average_gf` has exits that print a `WARNING`
to `asci.out` and return the polarized GF:

- `GF.BAND_AVERAGE = FALSE` (`impurity_solver.hpp:284`);
- an ORBS_COMP size mismatch (`:291`);
- a permutation that maps an entry outside ORBS_COMP (`:313`).

With every DMFT flag off, nothing downstream catches these. Keep the flag on and tighten
`symmetrize_warn_thresh` from 0.5 to about 1e-6. This applies to single-site runs, where the site
channel does nothing. The guard then raises only when MACIS skipped the average or the solve really
failed, which restores the detector described in "Why in MACIS and not only on the DMFT side". The
alternative is to make those exits fatal in MACIS whenever `SYMMETRIZE_DETS` is on.

**Symmetric does not mean correct.** The average always gives a symmetric GF. It gives the
T → 0⁺ ensemble GF only under the conditions in "What this does not fix":

- **The degenerate manifold must be a single orbit.** If a sector outside the orbit is accidentally
  tied with it (the warning at `impurity_solver.hpp:271`, for example two multiplets crossing near a
  transition), the result is symmetric but carries the wrong weights.
- **The solved member must have Sz = 0, or spin must be averaged.** Spin is a separate mechanism:
  `GF.SPIN_AVERAGE` in MACIS (on by default, `gf/gf.hpp:67`), not a DMFT flag. DMFT reads only the
  up-spin GF (`UPoComp`).

The DMFT-side projector has the same blind spots, so neither point is a regression.

**Larger groups.** Only S_n is averaged. For a single site with a band-diagonal bath, S_n plus
band-parity conservation already forces G ∝ 1 per spin. That is everything a continuous orbital
group (U(n) at J = 0, SO(3) for three-band Kanamori at finite J) could impose on a one-body matrix,
so the GF loses nothing. The RDM does lose something; see the S3 versus U(6) entropies above.

**Where the average cannot be used at all.**

- NROTS > 0.
- Baths with cross-band couplings: `band_permutation_generators` raises above `band_diag_tol`.
- Any band-dependent one-body term, such as a crystal field: `prepare_det_symmetry` throws.

In the first two cases the DMFT orbital flags remain the only tool. In the crystal-field case they
would be wrong anyway.

**Side effect.** The bath is re-projected every iteration and the GF is averaged, so the
self-consistency cannot leave the orbitally symmetric subspace. Real orbital order is erased by
construction. Turn the average off to look for nematic or orbitally ordered solutions.

## Tests

- **Extend `tests/gf_band_average.cxx`** with the 3-band J = 0 closed-shell case from the toy ED:
  a doublon member ((e,e,e), today not averaged) and a 0↑1↓ member ((o,o,e)). Assert a uniform
  diagonal, equal to the ensemble values the script prints, for every choice of representatives.
- **Regression.** In the existing 2-band test, the group is Z2, so for an odd key the cosets
  already exhaust the group and the result must not change. An invariant 2-band key ((e,e)) will
  now be averaged; update any assertion that expects it untouched.
