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

### Problems found (existing code, not fixed here)
1. **`read_fcidump_1body(fname, T, LDT)` ignores LDT** (`src/macis/fcidump.cxx:224`). It builds a
   strided `submdspan` and passes it as a `layout_left` span, so the leading dimension is lost when
   LDT differs from the file's norb, and the values land in the wrong elements. Reading the
   n_imp-orbital locFCIDUMP into the norb×norb `T` filled bath hoppings with garbage and made Davidson
   break down. The new main works around it by reading into an n_imp×n_imp buffer. Production calls
   it with LDT == norb and is not affected, but the overload should be fixed.
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
