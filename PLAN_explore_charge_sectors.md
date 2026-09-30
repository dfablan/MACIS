# Plan: `explore_charge_sectors` — a MACIS main that finds the ground-state charge sector

## Context
DMFT runs fix `nup`/`ndo` (usually norbs//2 each), but the impurity ground state can live in another
charge sector. On the archived 3-band run (`Leonardo_move/3band/singlesite/Doping/J_.2/U_10.00`)
the Python scan showed (6,6) is probably not the ground-state sector, and that Ω(N) is non-convex
(local minima at N=8 and N=10), so a 3-point bracket is not enough. The Python tool
(`~/scripts/explore_charge_sectors.py`) needs one SLURM task and one cold ASCI start per sector.

This new executable does the whole exploration in one process, starting from an archived iteration
(`It_N/ASCI.tar.gz`, like the Python `--it N`). It prints the ground-state sector. With
`--warm-start` it seeds each new sector from its already-solved neighbour by adding or removing one
electron (c†/c on the neighbour's leading determinants), instead of a cold canonical-HF start.

Principle, as in the Python tool: the FCIDUMP carries −µ, so E(CI) *is* Ω(N). All sectors use the
same FCIDUMP at fixed µ (no µ search), so their energies compare directly.

## Decisions (from the user)
- `--warm-start` (default off):
  - **on:** NROTS is forced to 0 everywhere; each sector is seeded from its neighbour's wavefunction.
  - **off:** every sector is solved cold with the input's NROTS (`SolveImpurityASCI_rot`).
- Setup code is **copied** from `run_asci_impsolv_dop.cxx` into the new main; the production driver
  is not touched.
- The input can be the **archive directly**: `It_N/ASCI.tar.gz`, `It_N/`, an `ASCI/` dir, or a plain
  `input.in`.
- Search: `--search walk` (default, with `--margin M`, default 2) or `--search window`
  (with `--window W`, default 2).

## Files
- **New** `main/explore_charge_sectors.cxx`.
- **Edit** `main/CMakeLists.txt`: add the target `explore_charge_sectors.cxx ../tests/ini_input.cxx`,
  linked to `macis` (same pattern as `charge_sector_estimate`).
- No library changes. Everything used already exists and is instantiated for N=64.

## Design of `main/explore_charge_sectors.cxx`

### 1. Input resolution (reuse `charge_sector_estimate.cxx` helpers: `looks_like_ini`, `resolve`, `dir_of`)
- For `*.tar.gz` / `It_N/`: `std::system("tar -xzf <tar> -C <workdir>")` into
  `--workdir` (default `./sector_scan_It<N>/`), then use `<workdir>/ASCI/input.in`. A stale absolute
  `CI.FCIDUMP` path falls back to the file of the same name next to `input.in` (existing `resolve`).
- **Final µ:** if `locFCIDUMP.dat` sits next to `input.in`, read its one-body block with
  `macis::read_fcidump_1body` and overwrite `T[0:n_imp, 0:n_imp]` (and `Td` if `spin_dep`). Print
  the µ before and after.
- **Reference wavefunction:** use `wfn.out` next to `input.in` only if warm start is on, the
  archived `input.in` has `NROTS = 0` (otherwise `wfn.out` is in a natural-orbital basis), its header
  (`read_wavefunction`, `wavefunction_io.hpp:66`) matches `norb`, and its (nalpha, nbeta) is the
  reference sector. Otherwise the reference is solved from scratch. Say which case applied.

### 2. Setup — copied from `run_asci_impsolv_dop.cxx`
- Copy `:56-157` (FCIDUMP, active space, `CI.*`), `:160-224` (MCSCF/ASCI settings; keep the
  `WFN_FILE` block, which is used internally), and `:275-332` (print loggers, `orb_rot`,
  `active_hamiltonian`, `E_inactive`).
- Drop the doping, GF and observables branches. Say that `DOPING`/`GF` are ignored because the scan
  is at fixed µ.
- `MPI_Init`/`MPI_Finalize` (the production driver lacks `Finalize`). Rank 0 prints and writes files.
- Keep pristine copies of `T_active`, `V_active`, `Td_active`. `SolveImpurityASCI_rot` rotates them
  in place when NROTS > 0, so they are restored before every sector.
- `--scale x` multiplies `ntdets_max`/`ncdets_max`, for cheap screening.
- `--nalpha/--nbeta` or `--half` override the reference sector (default: `input.in`'s NALPHA/NBETA).

### 3. Solving one sector — `SectorResult solve_sector(p, pristine, target, seed_source)`
- Reset `p.T_active/V_active/Td_active` from the pristine copies, set `p.nalpha`, `p.nbeta`,
  clear `p.asci_wfn_fname`.
- **Cold:** call `macis::SolveImpurityASCI_rot<64>(p)` (`src/macis/impurity_solver.cpp:502`) with
  the input's nrots (0 in warm mode for the reference when there is no `wfn.out`).
- **Warm** (nrots forced to 0):
  1. **Seed.** `apply_ladder(parent dets, parent C, spin, create)` loops over the top
     `--seed-parents` parent determinants (default `ncdets_max`) and every active orbital i whose
     bit allows the operation. It flips the bit (alpha is the low 32 bits, beta the high 32).
     Each new determinant gets weight `sqrt(Σ C_parent²)`, accumulated in a
     `std::map<wfn_t, double, bitset_less_comparator>`. Only the ranking matters, because Davidson
     recomputes C, so signs are not needed. This avoids spurious cancellation between different i.
     Moves in both α and β (an Sz flip) compose two ladder steps.
  2. **Truncate** to the top `min(--seed-size, ntdets_max)` by weight (default `ntdets_max`).
     Assert every determinant has the target (na, nb) (`bitset_lo_word/hi_word`).
  3. **Diagonalize in the seed space.** `macis::selected_ci_diag` (`solvers/selected_ci_diag.hpp:171`)
     on a `SDBuildHamiltonianGenerator` built over `p.T_active/V_active`
     (+`ReadTdo`, `SetJustSingles`, `SetNimp`, as in `impurity_solver.cpp:425-436`). This gives a
     variational E_seed and proper C.
  4. **Hand the seed to the production path.** Rank 0 writes it with `macis::write_wavefunction`
     to `<workdir>/seed_Na<a>_Nb<b>.wfn`, followed by a barrier. Then set
     `p.asci_wfn_fname = that file`, `p.compute_asci_E0 = false`,
     `p.asci_E0 = E_seed + E_core + E_inactive`, and call `SolveImpurityASCI_rot<64>(p)`.
     Its `load_asci_guess` (`impurity_solver.cpp:23-91`) already checks the sector, requires
     `nrots == 0` and `MAX_REFINE_ITER > 0`, then runs `asci_grow` (skipped if the seed already
     has `ntdets_max` determinants) and `asci_refine`. Symmetry closure, occupations, etc. all come
     from the production code.
- Wrap each solve in `try`. `asci_refine` throws "did not converge", which records the sector as
  UNCONVERGED rather than aborting the scan.
- Record: E (total), impurity n/band = `2·mean(p.occs[0:n_imp])`, number of determinants, wall time,
  the seed parent sector (or "cold"/"wfn.out"), and E_seed. E_seed vs E shows how good the warm start
  was.
- After a warm solve, keep `p.dets`/`p.C` as the parent for the next step outward.

### 4. Sector bookkeeping and search
- SU(2): only the minimal-|S_z| sector per N is solved, `((N+1)/2, N/2)`. The step from one N to
  the next is a single ladder operator (e.g. (k,k)→(k+1,k) adds ↑; (k+1,k)→(k+1,k+1) adds ↓).
  A spin-dependent input (`spin_dep`) prints a note that ±S_z mirrors are not equivalent; only
  minimal |S_z| is scanned.
- **walk:** solve N0, N0−1, N0+1 (the neighbours are seeded from N0). Loop: m = argmin Ω. If fewer
  than `margin` solved sectors lie below m, extend `lo−1` (seeded from `lo`). Likewise above m,
  extend `hi+1`. Stop when both sides have `margin` higher sectors or hit 0 / 2·n_active. This
  handles the non-convex case (8/9/10).
- **window:** solve N0−W..N0+W outward from N0, each seeded from its inner neighbour.
- `--check-spin` (optional): at the final N, also solve the (a+1, b−1) sector, seeded by
  c†↑ c↓ from the ground state, to flag a high-spin ground state (degenerate within `--etol`).

### 5. Output
- A table on stdout and `<workdir>/sector_scan.dat` with columns N, NALPHA, NBETA, E(CI)=Ω, E−E_min,
  n/band, E_seed, ndets, time, seed source, status.
- Verdict lines: the minimum; whether it is bracketed; the gaps Ω(N±1)−Ω(N); whether Ω has more than
  one local minimum; gaps below `--etol` (default 1e-4); whether the reference sector is the ground
  state.
- A greppable final line: `GROUND_SECTOR NALPHA = a NBETA = b N = n E = e`.

## Verification
1. **Exact test** (6-orbital `Ulysses_move/.../2bands/Doping/Nb4/J_0.1/U_10.00`, FCIDUMP patched to
   the final µ, `NTDETS_MAX` ≥ full space). The exact energies from the earlier ED are:

   | N | Exact E |
   |---|---|
   | 3 | −3.447724060 |
   | 4 | −3.793691151 |
   | 5 | −3.625266736 |
   | 6 | −3.192387002 |
   | 7 | −2.463327603 |

   Run from reference (3,3) in both modes. Both must end at `GROUND_SECTOR ... N = 4`, with every
   per-N energy matching to 1e-8.
2. **Warm vs cold** on the 3-band `It_7` archive at `--scale 0.01` on a scratch copy. Both modes
   should find the same minimum. Warm should show E_seed close to E and less wall time per sector.
   Report both.
3. **Input paths:** `.tar.gz`, `It_N/`, and plain `input.in`; the locFCIDUMP µ overlay is printed;
   the `wfn.out` reuse/skip message is correct for NROTS = 0 and NROTS > 0 archives.
4. **Build:** hand-compile a scratch binary with the flags and link line of the existing
   `run_asci_impsolv_dop` target. Do **not** run `make`: `CMakeLists.txt` changed, so `make` would
   reconfigure and poison the tree. The real build is `sbatch send_compile.job`, submitted by the
   user.
5. No jobs are submitted and no calculation folders are written; all tests run on scratch copies.

## Implementation status (done)

### What was implemented
- **New** `main/explore_charge_sectors.cxx`, plus the `explore_charge_sectors` target in
  `main/CMakeLists.txt` (same pattern as `charge_sector_estimate`). No library files changed.
- Everything in the design above is in place: the four input forms (`.tar.gz`, `It_N/`, `ASCI/`,
  `input.in`), the locFCIDUMP µ overlay, `wfn.out` reuse (warm mode only, with the stated checks and
  a message saying which case applied), cold and warm solves, `walk`/`window` search, `--check-spin`,
  `--scale`, `--seed-parents`, `--seed-size`, `--etol`, the table on stdout and in
  `<workdir>/sector_scan.dat`, the verdict lines and the `GROUND_SECTOR` line.
- Choices the plan did not cover:
  - The program `chdir`s into `--workdir` before solving, because `SolveImpurityASCI_rot` writes
    `active_ordm.dat` and `rot_matrix*.dat` into the current directory. Archive folders stay untouched.
  - With `spin_dep`, the µ overlay changes only the impurity **diagonal** of `Td`, as the µ search
    does (`set_impurity_diagonal` in `fix_mu.cpp`). `T`'s whole impurity block is overwritten.
    The overlay is refused when `NINACTIVE != 0`, like the µ search.
  - In warm mode, a seeded solve that fails for any reason other than refinement not converging is
    retried cold. The table shows it as `cold(seed failed)`. A refinement that does not converge is
    recorded as `UNCONVERGED`, as planned.
  - When a sector's parent failed, the sector is solved cold (`cold(parent failed)`).
  - To save memory, the program keeps wavefunctions only for the two ends of the solved range and
    the current minimum.

### How it was verified
Done in a cloud container, not on the cluster: the ULYSSES/LEONARDO archives were not available.
- **Build:** scratch CMake build outside the repo (MPI on, Release), no warnings from the new file.
  The real build is still `sbatch send_compile.job`.
- **Exact test (substitute for verification step 1):** a synthetic 6-orbital, 2-band Kanamori model
  (U = 6, J = 0.8, 4 bath sites with inter-band hybridization). Reference energies came from
  `charge_sector_estimate --exact`.
  - µ = 2, reference (3,3), true minimum N = 5. Cold, warm, and warm with 20-determinant seeds all
    give `GROUND_SECTOR ... N = 5`. The energy of every sector matches ED to 1e-10.
  - µ = 6, NROTS = 2, Hund's triplet ground state at N = 6. Cold and warm both match ED, and
    `--check-spin` flags (4,2) as degenerate with (3,3) ("high-spin").
- **Input paths:** `.tar.gz` (including a stale absolute `CI.FCIDUMP` path), `It_N/` (default workdir
  `sector_scan_It<N>`), `ASCI/`, and `input.in`. The µ overlay (FCIDUMP at µ = 4 + locFCIDUMP at
  µ = 2) reproduces the µ = 2 run exactly. `wfn.out` written by `run_asci_impsolv_dop` is reused for
  NROTS = 0 and skipped with the correct message for NROTS > 0 and for a mismatched sector.
- **Not verified:** multi-rank MPI (see problem 4); steps 1 and 2 of the plan on the real data.
- **Unit tests** (`macis_test`, single rank): 29 of 30 pass. The failing one, `ASCI Symmetric
  Search` (`tests/determinant_symmetry.cxx:318`, |E_sym − E_unsym| = 0.286 > 0.1), fails the same
  way on the base commit and has nothing to do with these changes.

### Problems found in existing code (1 fixed, 2–4 still open)
1. **FIXED — `read_fcidump_1body(fname, T, LDT)` ignored LDT** (`src/macis/fcidump.cxx`), and
   `read_fcidump_2body(fname, V, LDV)` had the same bug. Both built a strided `submdspan` and passed
   it as a `layout_left` span, so the leading dimension was lost when it differed from the file's
   norb, and the values landed in the wrong elements. Reading the n_imp-orbital locFCIDUMP into the
   norb×norb `T` filled bath hoppings with garbage. Fix: the span code is now a template shared by
   both overloads, and the pointer overloads pass the strided view unchanged. An LD smaller than
   norb now throws. Callers with LD == norb (all production calls) get the same values as before.
   The new "Leading Dimension" section in `tests/fcidump.cxx` fails on the old code and passes on the
   new. `explore_charge_sectors` now calls `read_fcidump_1body(loc, T, norb)` directly (it no longer
   needs the buffer workaround). Its µ-overlay energies are unchanged.
2. **A conserved parity traps Davidson in the wrong block.** With a band-diagonal bath and pair
   hopping, each band's electron-number parity is conserved. `selected_ci_diag` starts from the
   lowest diagonal element (`p_diagonal_guess`) and never leaves that parity block, so ASCI (and even
   a full-space diagonalization) can return an excited state. In the first test, N = 3, 4, 7 were off by
   0.3–0.5 Ha until an inter-band hybridization was added. Production 3-band Kanamori runs with
   band-diagonal baths have the same structure. This should be checked there.
3. **`asci_refine` requires a fixed size** ("Wavefunction size can't change in refinement",
   `include/macis/asci/refine.hpp`). A guess that already holds `NTDETS_MAX` determinants skips
   `asci_grow`. If the ASCI search then cannot return that many determinants, refinement throws.
   Seen with `--scale 0.3` on the small test, where the search could return fewer than 300 determinants. A
   production `ASCI.WFN_FILE` restart can hit the same check. Covered here by the cold retry.
4. **Multi-rank runs hang in this container.** With `mpirun -np 2`, the first Davidson stalls, and it
   does so for the unmodified `run_asci_impsolv_dop` too. The problem is either the environment or
   the library, not the new main. The multi-rank code in the new main (eigenvector gather copied
   from `asci_iter`, rank-0 file writes followed by a barrier) is untested. Do one short multi-rank
   run on the cluster.

### Still to do (on the cluster)
- `sbatch send_compile.job`.
- Verification step 1 on the real 6-orbital `Ulysses_move/.../2bands/Doping/Nb4/J_0.1/U_10.00`
  case, and step 2 (warm vs cold, 3-band `It_7`, `--scale 0.01`), including a multi-rank run.

## Warm start with NROTS > 0 ("option A", done)

### What changed (`main/explore_charge_sectors.cxx` only; no library changes)
- `--warm-start` no longer forces NROTS = 0 everywhere. A sector that starts **cold** (the
  reference, or a fallback) uses the input's NROTS and ends in its own natural-orbital (NO) basis. A
  **seeded** sector is solved in its parent's basis:
  - before seeding, its active integrals are rotated in place by the parent's cumulative rotation
    (`rotate_hamiltonian_rotmat_imp_bath`, T ← UᵀTU for T, Td and all four indices of V);
  - it then runs with NROTS = 0 in that basis (`load_asci_guess` requires this anyway).

  So every seeded sector ends up in the NO basis of the cold sector it descends from.
- Each sector records the basis its determinants are in (`SectorResult::U`: the parent's basis for a
  seeded sector, the solver's `orb_rot` for a cold sector with NROTS > 0, empty for the original
  orbitals). That basis is passed on to its children and to `--check-spin`.
- **Singles-only must be turned off after the rotation**, in `p` itself:
  `SolveImpurityASCI_rot` re-applies `p.just_singles` to its own generator, and a rotated
  density-density interaction has double excitations. `rotate_active` sets `p.just_singles = false`,
  and `restore` puts the pristine value back for every sector.
- n/band needs no back-rotation: all these rotations are block-diagonal in (impurity, bath), so the
  impurity trace is invariant.
- **Reference from the archive with NROTS > 0:** `wfn.out` is reused together with the
  `rot_matrix.dat` next to it (both come from the solver's last call). The rotation matrix is checked
  to be orthogonal (to 1e-8) and block-diagonal in (impurity, bath) (to 1e-10); if either check fails,
  or `rot_matrix.dat` is missing, the reference is solved from scratch and the message says why.
- `--warm-nrots0` (implies `--warm-start`) keeps the previous behaviour: NROTS = 0 for every sector,
  all in the original orbitals.
- Known bias, printed as a note at startup: the inherited basis is optimal only for the sector it
  came from, which lowers that sector's E slightly relative to its descendants. At production size
  this should be around the NROTS gain (~1e-5 Ha in the U_1.00 log). Confirm near-degenerate sectors
  (`--etol` warning) with a cold run.

### Verification (cloud container, single rank)
- **Full space, NROTS = 2** (energy is basis-independent, so this checks the rotation and seeding):
  matches exact ED to 1e-10 in every sector for (a) the Hund's-coupled model (J = 0.8, µ = 6, plus
  `--check-spin` on the triplet), (b) a density-density model (J = 0, U' = 4.4 ≠ U, impurity
  orbitals mixed ~45° by the NO rotation), and (c) a spin-dependent input (`CI.FCIDUMP_DO` with a
  Zeeman field on the impurity; Td is rotated).
- **Negative control:** without `p.just_singles = false`, case (b) is off by up to 1.3e-2 Ha in the
  inherited sectors, so the test catches that mistake.
- **Truncated space** (10 orbitals, NTDETS_MAX = 800 of up to 63504 determinants, NROTS = 2,
  window 2 around N = 10; exact ground state N = 9). Errors vs exact ED:

  | N | cold (NROTS = 2) | warm, option A | `--warm-nrots0` |
  |---|---|---|---|
  | 8 | UNCONVERGED | 3.1e-3 | UNCONVERGED |
  | 9 | 2.1e-4 | 6.7e-4 | 4.9e-3 |
  | 10 (reference) | 3.6e-4 | 3.6e-4 | 8.2e-3 |
  | 11 | 4.1e-4 | 1.0e-3 | 6.6e-3 |
  | 12 | 1.0e-1 (stuck from the HF start) | 4.9e-3 | 1.1e-2 |

  All three find N = 9. Option A converges every sector. It is 2–10× more accurate than
  `--warm-nrots0`, and 2–5× less accurate than each sector's own NO basis: that is the inherited-basis
  bias, which is large here because the budget is tiny.
  (With REFINE_ETOL = 1e-8 instead of 1e-6, cold refinement flipped between two determinant sets at
  dE = ±1.4e-6 and never converged, which also made the cold run report N = 10. This is a tolerance
  choice, not a code problem; production uses 1e-4.)
- **Archive reuse:** `run_asci_impsolv_dop` with NROTS = 2 wrote `wfn.out` and `rot_matrix.dat`. The
  warm scan seeded from them reproduces the production E(CI) exactly (E_seed = E = −12.9801768906),
  and its neighbours match the cold-reference warm run. A non-orthogonal matrix, an impurity–bath
  mixing matrix, a missing `rot_matrix.dat`, and `--warm-nrots0` are each rejected with the right
  message.
- **Regression:** cold runs (NROTS = 0 and 2) and warm runs with NROTS = 0 in the input give tables
  identical to before the change.

## Charge-sector search inside the doping mu search (done)

### What it does
With `CI.DOPING = true`, `run_asci_impsolv_dop` now calls `macis::Fix_Mu_sectors`
(`src/macis/doping/charge_sectors.cpp`) instead of `Fix_Mu_der/noder` directly.
It is an outer loop around the existing mu search:
1. mu search in the current sector N (unchanged code).
2. At the converged mu, scan the neighbours (`SectorScan::walk`, margin `DOP.SECTOR_MARGIN`,
   default 2), seeded from the solved sector with the same ladder seeds as `--warm-start`.
3. **Accept N** if no scanned converged sector lies more than `DOP.SECTOR_ETOL` (1e-4) below it.
   Sectors within etol give a warning. Unconverged neighbours are warned about and left out.
4. Otherwise **switch** to the lowest sector and go to 1, starting from the same mu. The new search
   starts from the scan's wavefunction when NROTS = 0 (original basis), otherwise cold; if it fails it is redone cold.
5. **Error** (the run stops) if a sector would be searched twice (the target filling lies in a jump of
   the ground-state filling: no ground state has it) or after `DOP.SECTOR_MAX_SWITCH` (4) switches.
The target is the filling of the lowest E_N(mu) over N. Only minimal-|S_z| sectors are scanned; an
input sector with |NALPHA - NBETA| > 1 skips the search with a warning.
The accepted sector, mu and the final scan go to `GS_charge_sector.dat` (also written, with a note, when
`DOP.SECTOR_SEARCH = FALSE`), and to stdout as `GROUND_SECTOR NALPHA = a NBETA = b N = n E = e MU = x`.
The DMFT script reads that file to update NALPHA/NBETA of the next iteration's input.in; the solver does not.
Keys: `DOP.SECTOR_SEARCH, _MARGIN, _ETOL, _WARM, _MAX_SWITCH, _DIR` (see README).

### Code
- New `include/macis/doping/charge_sectors.hpp`, `src/macis/doping/charge_sectors.cpp`: the seeding,
  solving and scan code moved out of `main/explore_charge_sectors.cxx` (unchanged behaviour), plus
  `Fix_Mu_sectors` and `write_ground_sector_file`. Neighbours are solved with ED when `CI.EXPANSION = CAS`.
  In cheap mode the current sector is re-solved with full ASCI for the comparison.
- `main/run_asci_impsolv_dop.cxx`: reads the keys and calls `Fix_Mu_sectors`.
- `main/explore_charge_sectors.cxx`: uses the library.
- New unit test `tests/charge_sectors.cxx`.

### Verification (cloud container, single rank)
- `explore_charge_sectors` output (tables, verdicts, GROUND_SECTOR) is identical before and after the
  refactor for cold, warm, `--warm-nrots0` and `--check-spin` runs on the synthetic models.
- Synthetic 6-orbital, 2-band Kanamori model with an independent Python ED reference. Target 0.6
  electrons/orbital (exact: N = 5 at x = -3.157801413, E = -8.8789574814): starting from the wrong sector
  (3,3) the run switches to (3,2) and ends at x = -3.15780143, E = -8.87895750, for ASCI with NROTS = 0 and 2,
  warm and cold neighbours, CAS, cheap mode, margin 1, and the secant solver (derivative method).
  Target 0.9 (exact N = 6, x = -4.04132321): stays in or switches to (3,3) from either side.
  Target 0.75 (a jump: N = 5 at x = -4.117, N = 6 at x = -3.030, each lower at the other's mu): error.
- With `DOP.SECTOR_SEARCH = FALSE` the driver output equals the unmodified driver's (timing digits aside).
- `macis_test`: 30 of 31 cases pass; the failing one is the known `ASCI Symmetric Search`.
- Observed, not new: with NROTS = 2 the bracketing mu search in a sector can stop on a discontinuity of
  n(mu) (two states of different character at the same mu, e.g. inside the S_z = 0 sector). The sector
  search now warns when the final filling misses the target by more than 1e-3.
- Not verified: multi-rank MPI (same container problem as above). Run one short multi-rank job on the cluster.
