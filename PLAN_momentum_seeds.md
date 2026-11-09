# Momentum seeds on the rotated path (option 3) — branch `momentum_seeds`

> **STATUS (2026-11-09): built and validated on the dimer, uncommitted.**
> - Build: `MACIS_build_momentum_seeds`, source snapshot `MACIS_build_momentum_seeds_source.patch`.
> - Unit tests: 91/93 pass; the 2 failures are the known weak tests, and all 4 new tests pass.
> - Dimer (`/leonardo_work/Sis26_collura/dfloreza/momentum_seeds_dimer_20261109`, `compare_dimer.tsv`):
>   - χ0(q=π) is within 0.3-1.75% of the sector-enriched site-basis v3, where v2 was 31-48% low.
>   - q=0 matches the same run's Sz_delta to 1e-15; R(q0,qπ) = 0; nothing dropped as coupled.
>   - ~1 h and 8 GB per point.
> - Not yet run on the 2x2, where the X/Y gauge fix is actually exercised.

> **STATUS (2026-11-09): DRAFT.** Code written on top of `resolvent_testing` 1e79317 and
> uncommitted. The changed TUs pass `g++ -fsyntax-only` with no warnings: `impurity_solver.cpp`,
> `run_asci_impsolv_dop.cxx` and `tests/dynamical_properties.cxx`. Not compiled, not unit-tested,
> never run.

## Why

The option-B test (job 63871827, `N_20_symdets/J_0.333/U_15.00/Post/It_9/OrbMatrix_FullRot_test_20261108`)
died in `added_components`: the CSR over 59.66M grown determinants overflowed int32 (2.15e9 nnz
at row 23.26M). Two problems sat under the overflow.

1. **Wasted pairs.** 32 of the 64 rotated pairs cross the bands. Their images lie in odd band-parity
   sectors, and site Sz never uses them: the impurity rotation is band-block-diagonal.
2. **The (π,π) channel is never expanded.**
   - The degenerate X/Y natural orbitals came out mixed at 36° (occupations 0.164877 / 0.164873).
     The impurity and bath blocks are both mixed.
   - In that basis, total momentum is not a quantum number of the determinants.
   - Γ↔M and X′↔Y′ images stay H-connected to ψ0, so `added_components` drops them.

## What changes

| Piece | Where |
|---|---|
| `SignedPerm`, `signed_perm_matrix`, `to_rotated_basis`, `symmetry_adapted_block_orbitals`, `orbital_character_masks`; `covariant_natural_orbitals(..., adapt, adapt_leak)` | `include/macis/asci/covariant_orbitals.hpp` |
| `ASCISettings::rot_adapt` | `include/macis/asci/determinant_search.hpp` |
| `prepare_rot_adapt` (signed-perm integral check, involution, commutation with each other and with the SYMMETRIZE_DETS group); rotation loop takes the covariant path with adapted blocks; end-of-solve character check | `src/macis/impurity_solver.cpp` |
| `GFSettings::orb_sym_mask`; `grow_basis_by_singles` keeps character-conserving singles only | `gf/gf.hpp`, `gf/dynamical_properties.hpp` |
| `RunResolventBilinearSeeds` (seeds = any combination ΣC_ab O_ab); `RunResolventOrbitalMatrix` becomes a wrapper with unit C (bit-identical seeds); `bilinear_combination_image`, `apply_bilinear_combination`, `spin_weight_bilinear_coefficients` (C = ½ Uᵀ diag(w) U) | `gf/dynamical_properties.hpp` |
| `evaluate_resolvent_spin_weights`, `detail::impurity_rotation_block` (shared with the orbital matrix), `detail::write_seed_resolvent_matrix` | `include/macis/impurity_solver.hpp` |
| Keys `ASCI.ROT_ADAPT_NGEN/_PERM_g/_SIGN_g`, `GF.SPIN_WEIGHT_RESOLVENT/_N/_k`, `GF.ORB_SYM_FILTER` (default TRUE, only acts with ROT_ADAPT and NROTS > 0) | `main/run_asci_impsolv_dop.cxx` |

Adaptation diagonalizes the 1-RDM inside each joint ±1 eigenspace of the generators. That equals
averaging ρ over the abelian group, a choice of basis only; `adapt_leak` reports how far ρ is from
symmetric. The generators are given on the original orbitals and are expressed in each macro
iteration's basis as orb_rotᵀ M orb_rot.

## Input for the 2×2 two-band Irrep runs

Each generator below was verified against `OrbMatrix_FullRot_test_20261108/FCIDUMP.dat` with
deviation 0. Dropping the bath signs gives 1.08.

```
ASCI.ROT_ADAPT_NGEN   = 3
# Ta: sites (0 1)(2 3); bath X-like and M-like orbitals odd
ASCI.ROT_ADAPT_PERM_1 = 1 0 3 2 5 4 7 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
ASCI.ROT_ADAPT_SIGN_1 = 1 1 1 1 1 1 1 1 1 -1 -1 1 1 -1 -1 1 1 -1 -1 1 1 -1 -1 1
# Tb: sites (0 2)(1 3)
ASCI.ROT_ADAPT_PERM_2 = 2 3 0 1 6 7 4 5 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
ASCI.ROT_ADAPT_SIGN_2 = 1 1 1 1 1 1 1 1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1
# band-1 parity (keeps cross-band singles out of the growth)
ASCI.ROT_ADAPT_PERM_3 = 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
ASCI.ROT_ADAPT_SIGN_3 = -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 1 1 1 1

GF.SPIN_WEIGHT_RESOLVENT = TRUE
GF.SPIN_WEIGHT_N = 4
GF.SPIN_WEIGHT_1 = 1  1  1  1  1  1  1  1     # Gamma (= Sz_resolvent.dat)
GF.SPIN_WEIGHT_2 = 1 -1  1 -1  1 -1  1 -1     # X (odd under Ta)
GF.SPIN_WEIGHT_3 = 1  1 -1 -1  1  1 -1 -1     # Y (odd under Tb)
GF.SPIN_WEIGHT_4 = 1 -1 -1  1  1 -1 -1  1     # M
GF.ORB_EXPAND_BASIS = TRUE
```

These generators are not in the SYMMETRIZE_DETS group, so they need no extra det
symmetrization. The band swap must map each generator to ± itself: Ta and Tb commute with it,
and P1 goes to −P1. `prepare_rot_adapt` checks this, and the end-of-solve character check is
the backstop.

**Site matrix.** O_q = Σ_s c_q(s) Sz_s, with c_q = ±1, summed over both bands of each site. Then
χ_ij = (1/16) Σ_q c_q(i) c_q(j) R_qq, since the cross terms R_qq′ vanish by symmetry and the log
prints their max. On-site: (R_Γ + R_X + R_Y + R_M)/16.

## Built-in checks

- The Γ seed reproduces `Sz_resolvent.dat` when `GF.SZ_RESOLVENT` is on in the same run.
- R_XX = R_YY (C4).
- The off-diagonal R_qq′ vanish (printed).
- Capture of every q ≠ 0 seed is 1 after expansion. Nothing should be "dropped as coupled": a
  nonzero dropped count for a q ≠ 0 seed means the adaptation failed.

## Unit tests added (`tests/dynamical_properties.cxx`, not yet run)

- Combination seeds are linear in the pair matrix resolvent; a unit C is bit-identical to
  `apply_orbital_bilinear`.
- With U = 1, `spin_weight_bilinear_coefficients` gives the same operator and normalization as
  `RunResolventWeighted(Spin)`; a uniform weight gives C = ½·1 for any U.
- Character filter: all-zero masks leave the growth unchanged; the filtered growth stays in the
  seed's character.
- 2×2 plaquette with X/Y mixed at 36°: adapted orbitals are pure Γ/X/Y/M with the right
  occupations; plain gesvd stays mixed; a 4-cycle generator throws.

## Open

- **Ground state changes.** The adapted basis changes the ASCI truncation, so E0 must be
  re-checked against −14.798305 (REFINE_ETOL 1e-5).
- **Basis size per q.** Unknown. The filter removes cross-sector singles, so expect far below
  59.7M, but the overflow guard still applies. Start with `TOT_SD 0` (exact zeroth moment), then
  measure.
- **Γ channel.** It stays on the ASCI basis. `GF.ORB_EXPAND_SECTOR` enriches it; see
  `PLAN_capture_basis_expansion.md` §Sector enrichment for the dimer evidence.
- **Other generators.** Z2 only: real orbitals cannot carry complex characters, so translations
  of order > 2 are refused.
- **J = 0.** The rotation path never ran at J = 0 with symdets. ROT_ADAPT itself does not need
  SYMMETRIZE_DETS (empty group). Without PARITY_SOLVE, add band-parity generators so the impurity
  block is split per band.
