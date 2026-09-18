# hole_particle_mismatch — GF index remapping for the polarized limit

## Symptom

Three DMFT runs aborted at the same point with the same solver error:

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  In evaluate_GF: todelete_h != todelete_p, the particle and hole
           Green's functions dropped different orbitals and cannot be summed
```

Affected jobs (all J=0, deep in the orbitally polarized phase):

| Job | Dir | n₁ | n₂ |
|---|---|---|---|
| 56481901 | `delta_0.5/J_0/again/U_22.00` | 1.004 | 0.544 |
| 56473134 | `delta_1/J_0/again/U_15.00` | 0.991 | 0.0008 |
| 56473136 | `delta_1/J_0/again/U_20.00` | 1.001 | 0.0005 |

Each crashed at It_5 — the first iteration after polarization completed.

## Root cause

`evaluate_GF` (`include/macis/impurity_solver.hpp`) computes the particle and
hole Green's functions separately via `RunGFCalc`. Each sector drops orbitals
whose add/remove vector vanishes (`norm ≤ zero_thresh = 1e-7`, in
`BuildWfn4Lanczos`, `include/macis/gf/gf.hpp`):

- n₁ → 1 (orbital full): `a₁⁺|GS⟩ = 0` → orbital 1 dropped from the **particle** sector.
- n₂ → 0 (orbital empty): `a₂|GS⟩ = 0` → orbital 2 dropped from the **hole** sector.

So `todelete_p = {1}` while `todelete_h = {2}`. The guard at
`impurity_solver.hpp:163` threw on the mismatch, because the two reduced
matrices index *different* orbital subsets and cannot be summed directly.

This guard was added deliberately in commit `7880030` ("Added to_delete +
assertions") and is documented in `FORK_REVIEW.md` §1.6: before that commit the
code silently read out of bounds and produced mis-indexed GFs, so failing loudly
was the correct stopgap. The index remapping needed to support the polarized
case was never implemented — hence the crash.

## The fix

The physically correct full `n_imp × n_imp` GF in the polarized limit is:

```
G₁₁ = hole propagator of orbital 1        (hole sector only)
G₂₂ = particle propagator of orbital 2    (particle sector only)
G₁₂ = G₂₁ = 0                             (no inter-orbital charge fluctuation at T=0)
```

A dropped orbital contributes **exactly zero from the sector that dropped it**,
so the two sectors occupy disjoint diagonal blocks. The fix is to **pad each
sector's reduced GF back to the full `GF_orbs_comp` index space with zeros**,
then add elementwise — instead of shrinking and requiring matching drop lists.

### Changes

`include/macis/gf/gf.hpp`:
- `RunGFCalc` now pads the `nvecs × nvecs` GF back to `GF_orbs_comp.size()²`
  with zeros in the dropped rows/cols (scatter via the surviving-orbital list).
- New early return: when a sector drops every requested orbital (`nvecs == 0`),
  it returns an all-zero full-size GF instead of seeding the resolvent with
  empty vectors.
- Per-sector diagnostic files (`LanGFMatrix_ADD/SUB.dat`) are written without
  the drop list, since the matrix is now full-size.
- Docstrings updated; `#include <algorithm>` added for `std::find`.

`include/macis/impurity_solver.hpp` (`evaluate_GF`):
- Removed the `todelete_h != todelete_p` throw.
- Removed the `!todelete_p.empty()` throw.
- `sum_GFs` call replaced by a direct elementwise add of the two full-size GFs.
- `write_GF` now writes the full matrix (empty drop list).

`tests/test_driver.cxx`, `tests/test_driver_dop.cxx`:
- Same pattern applied (drop-list check removed, elementwise add).

`sum_GFs` (`include/macis/gf/gf.hpp`) is now unused by production and tests but
left in place.

### Correctness

- Diagonal elements: `G₁₁` comes only from the hole sector, `G₂₂` only from the
  particle sector — the two physical propagators.
- Off-diagonal elements: zero in both sectors → zero, the correct T=0
  single-reference result for a fully polarized state.
- Downstream `import_mat` (`PolClassy_DMFT/inout.py`) reads the full 2×2, so
  `CompSelfEn` produces the physically expected Σ (Mott-insulating Σ₁₁ from the
  hole GF, band-like Σ₂₂), consistent with the note in `delta_1/J_0/again/README.md`
  that Z₂ becomes meaningless once n₂ → 0.

## Status / not done

- **Implemented but not compiled.** No build or test run was performed.
- The general case where `GF_orbs_comp` is a proper subset of the impurity
  orbitals is still unsupported (the back-rotation assumes the full impurity
  set); the existing `G_n_orbs < n_imp` guard still covers it.
- Near-zero (but not exactly dropped) Lanczos seeds — orbitals with occupation
  just above the `1e-7` norm threshold — remain a separate numerical robustness
  concern, not addressed here.
