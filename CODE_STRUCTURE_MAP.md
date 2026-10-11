# CODE_STRUCTURE_MAP — where things live in the MACIS fork

A navigation reference for agents. It tells you **where a feature already lives** and
**how one solver run flows**, so you can find existing code before you write new code.
It does not replace reading the source. Line anchors were taken against the working tree
of commit `50136d6`, plus the uncommitted edits present on 2026-10-11. They drift, so
`grep` the symbol name before you trust a line number.

Build, cluster and DMFT-interface notes are in `../CLAUDE.md` (the fork root) and are not
repeated here. Read that file before you build anything.

---

## 0. Orientation

- **What this is.** This is a fork of upstream MACIS (LBNL: ASCI selected-CI + CASSCF). It
  adds a **T = 0 Anderson-impurity solver** for DMFT. Several layers sit on top of upstream:
  an impurity driver, a µ search (doping), the Green's function via band Lanczos,
  impurity observables and RDMs, charge-sector search, band-parity sector solve, and
  orbital-permutation symmetrization.
- **Worktrees.** `MACIS_fork/MACIS` (branch `feature/spin_dep`) is the checkout that
  `MACIS_build/` and `send_compile.job` compile, so it is **the production binary**.
  `MACIS_claude`, `MACIS_experimental` and `MACIS_resolvent_testing` are git worktrees of the
  same repo. An edit made in a worktree does not reach the production binary until it is
  merged into `feature/spin_dep`.
- **Language and layout.** C++17. The code is mostly header-only templates in `include/macis/`.
  Explicit `<64>` instantiations live in `src/macis/*.cpp|.cxx`, and those files are compiled
  into `libmacis`. The determinant type is `wfn_t<N> = std::bitset<N>` (`types.hpp:43`), with
  **alpha in the low N/2 bits and beta in the high N/2 bits**. Production uses N = 64, so at
  most **32 active orbitals**. The driver throws `"Not Enough Bits"` above that.
- **Dependencies.** Eigen, blaspp/lapackpp, GSL (the µ root-finders), spdlog, mdspan,
  OpenMP and MPI (optional). There are two vendored sub-libraries: `src/sparsexx`
  (sparse/distributed matrices) and `src/lobpcgxx`.
- **Ignore `attic/`.** It holds the dead upstream prototype code: `cmz_ed`, `dbwy`, old tests
  and inputs. Nothing builds it.

---

## 1. Top-level layout

| Path | Contents |
|---|---|
| `main/` | Executables (§2). **`run_asci_impsolv_dop.cxx` is the production driver.** |
| `include/macis/` | Almost all algorithmic code, as headers (§4) |
| `src/macis/` | Non-template code and the `<64>` instantiations; `CMakeLists.txt` lists what goes into `libmacis` |
| `src/sparsexx/`, `src/lobpcgxx/` | Vendored linear-algebra libraries (CSR/COO, distributed SpMV, MM/RB file IO) |
| `tests/` | Catch2 unit tests (`macis_test`), the INI parser `ini_input.{hpp,cxx}`, legacy test drivers, `ref_data/` |
| `cmake/` | FetchContent helpers (spdlog, mdspan, catch2, ips4o) |
| `*.md` at root | Fork design notes and reviews (§8) |
| `symmetry_sector_files/` | Superseded parity-sector drafts and the assessment script |
| `attic/` | Dead code. Do not read it as current |

---

## 2. Executables (`main/CMakeLists.txt`)

| Binary | Source | Purpose |
|---|---|---|
| `run_asci_impsolv_dop` | `main/run_asci_impsolv_dop.cxx` | **The production impurity solver called by PolClassy_DMFT.** Run as `run_asci_impsolv_dop input.in` |
| `explore_charge_sectors` | `main/explore_charge_sectors.cxx` | Offline scan of charge sectors N at **fixed µ** from an archived `It_N/ASCI.tar.gz`, an `ASCI/` dir or an `input.in`. Uses `SectorScan` (§4.6) |
| `charge_sector_estimate` | `main/charge_sector_estimate.cxx` | Cheap estimate of the ground-state (NALPHA, NBETA): the U=0 and V=0 limits plus a heuristic. `--exact` does full ED |
| `run_asci_impsolv_mu_vs_n` | `main/run_asci_impsolv_mu_vs_n.cxx` | Scans n(µ) over `DOP.INIT_MU..FINAL_MU` (`MU_NPT` points) and writes `mu_vs_n.dat`. Uses `Mu_vs_n` → `SolveImpurityASCI` (the non-rotating path) |
| `test_lapack_convention` | `main/test_lapack_convention.cxx` | Sanity check that blas/lapack are column-major |
| — | `main/read_rho_blocks.py` (untracked) | Python reader for `rho_blocks.bin` |

`tests/` also builds `standalone_driver`, `test_driver` and `test_driver_dop`. These are older
driver copies from upstream and early fork work, and are **not** what DMFT calls.

---

## 3. Workflow of one production run (`run_asci_impsolv_dop`)

DMFT calls the binary **once per DMFT iteration**, with `cwd = It_N/ASCI/`. The binary reads
`input.in` and `FCIDUMP.dat` and writes everything into the cwd.

```
main()                                         main/run_asci_impsolv_dop.cxx
 ├─ INIFile input(argv[1])                     tests/ini_input.hpp  (keys "SECTION.KEY")
 ├─ read FCIDUMP → params.T, V, E_core         util/fcidump.hpp, src/macis/fcidump.cxx
 │    CI.FCIDUMP_DO → params.Td (spin-down one-body; sets spin_dep)
 │    is_2body_diagonal → params.just_singles  (Hubbard–Kanamori has diagonal V)
 ├─ parse optional keys via OPT_KEYWORD        (§5 has the full key list)
 │    ASCI.SYMMETRIZE_DETS → sym_group generators (expanded later)
 │    ASCI.WFN_FILE guard checks (needs E0_WFN, NROTS=0, MAX_REFINE_ITER>0)
 ├─ setup_parity_sectors (if ASCI.PARITY_SOLVE, default ON)   :319
 │    → build_parity_labels: detects band groups, zeroes parity-breaking ints ≤ PARITY_TOL
 ├─ active_hamiltonian(...) → T_active, V_active, F_inactive  util/fock_matrices.hpp:124
 │  inactive_energy → E_inactive
 │
 ├─ CI.DOPING = TRUE ? ─────────────────────────────────────────────────────────┐
 │   yes: µ search (doping/)                                                    │
 │     DOP.SECTOR_SEARCH ? Fix_Mu_sectors (µ search + charge-sector walk)       │
 │                       : Fix_Mu_der / Fix_Mu_noder (GSL root find)            │
 │       each trial µ → Mu_Cost_f: set_impurity_diagonal(T[,Td], µ, δ_CFS)      │
 │                       → rebuild active ints → SolveImpurity{ED|ASCI_rot|Cheap}│
 │                       → err = n_imp(µ) − DOP.NELECTRONS                      │
 │     final re-solve at the root µ; write GS_charge_sector.dat, locFCIDUMP.dat │
 │   no: SolveImpurityED (CAS) | SolveImpurityASCI_rot (ASCI) | CheapASCI       │
 │ ◄────────────────────────────────────────────────────────────────────────────┘
 ├─ print E(CI) and "Occs:" (per orbital per spin, ORIGINAL basis)  ← DMFT parses these
 ├─ observables (CI.COMP_*)                    CompObservables, comp_observables.cpp
 │    double occupancy, <Sz Sz>, <τz τz>, charge-charge, impurity RDM → rho.bin[.txt], rho_blocks.bin
 └─ GF (CI.GF) and/or Sz resolvent (GF.SZ_RESOLVENT)
      rebuild active ints, SDBuildHamiltonianGenerator, rotate into orb_rot basis
      evaluate_GF  (impurity_solver.hpp:349)
        RunGFCalc particle + hole (band Lanczos, 64-bit CSR)   gf/gf.hpp:562
        spin average (NALPHA≠NBETA) → SECTOR_CHECK line → back-rotate to original basis
        band_orbit_average_gf (parity sector not invariant under sym group)
        write_GF → GF.dat
      evaluate_resolvent_sz → Sz_resolvent.dat (bosonic Matsubara, starts at ν₁)
```

### 3.1 Inside one impurity solve (`src/macis/impurity_solver.cpp`)

The public entry points are at `:1330–1440`. Each wraps a private `*_one` function through
`dispatch_solve` (`:1297`). When parity labels exist and no target is set yet, the wrapper
runs **every band-parity sector** (`solve_parity_sectors`, `:1155`), keeps the lowest, and
records ties in `params.parity_tied_keys`.

| Entry | Inner | What it does |
|---|---|---|
| `SolveImpurityED` | `solve_ed_one` `:448` | Full Hilbert space (`generate_hilbert_space`), filtered to the parity sector if one is set, then `selected_ci_diag` |
| `SolveImpurityASCI` | `solve_asci_one` `:571` | ASCI grow + refine, no orbital rotation. Called only by `Mu_vs_n` |
| `SolveImpurityASCI_rot` | `solve_asci_rot_one` `:696` | **Production ASCI path.** Runs `NROTS+1` macro iterations. In each: seed → `asci_grow` → `asci_refine`. If more remain: 1-RDM → `rotate_hamiltonian_ordm_imp_bath` (natural orbitals, impurity and bath kept separate; per band group under parity), or under `SYMMETRIZE_DETS` `covariant_natural_orbitals` + `rotate_hamiltonian_rotmat_imp_bath` and a re-check of the integral symmetry → new HF det `hf_determinant_byocc` / `parity_restart`. Afterwards writes `active_ordm.dat` and `rot_matrix.dat` (`write_rot_outputs`). `rot_matrix_<i>.dat` is written per macro iteration |
| `SolveImpurityCheapASCI` | inline `:1370` | Re-diagonalizes the existing `params.dets` without a search. Used by `DOP.CHEAP_MODE` after the first µ step |

Helpers, all in the anonymous namespace of `impurity_solver.cpp`:

- `load_asci_guess` `:29` — the ASCI.WFN_FILE warm start.
- `asci_reference_determinant` `:130` — the HF seed, energy-ordered unless HF_BY_ENERGY=FALSE.
- `prepare_det_symmetry` `:168` — validates and expands the permutation group.
- `close_guess_under_group` `:273`, `check_rdm_symmetry` `:304`.
- `sector_seed` `:340`, `require_in_sector` `:362`, `parity_restart` `:392`.
- `split_guess_by_sector` `:1007`, `report_parity_sectors` `:1080`.

### 3.2 Inside ASCI (`include/macis/asci/`)

```
asci_grow (grow.hpp)        while ndets < NTDETS_MAX: ndets ← min(max(NTDETS_MIN, ndets·GROW_FACTOR), MAX)
   └─ asci_iter (iteration.hpp)    sort by |C|; take NCDETS_MAX seeds (closed under sym group)
        ├─ asci_search (determinant_search.hpp:455)
        │     world_size==1 → asci_contributions_standard   :94
        │     MPI           → asci_contributions_constraint :202 (mask_constraints.hpp)
        │     → sort/accumulate pairs (determinant_sort.hpp) → top-k
        │     → symmetric_orbit_select if SYMMETRIZE_DETS (determinant_symmetry.hpp:148)
        │     → drop off-sector candidates if parity_target (logged count must be 0)
        └─ selected_ci_diag (solvers/selected_ci_diag.hpp:171) → CSR H → Davidson
asci_refine (refine.hpp)    fixed size, repeat asci_iter until |ΔE| < REFINE_ETOL or MAX_REFINE_ITER
                            (throws); a period-2 cycle (det set of iterate k == that of k-2) stops on the lower state
   (optional) GROW_WITH_ROT: grow.hpp rotates over the FULL active space; incompatible with PARITY_SOLVE
```

---

## 4. Feature index — "does this already exist?"

### 4.1 Input, output, integrals
| Feature | Location |
|---|---|
| INI parser (`getData<T>`, `containsData`, vectors) | `tests/ini_input.hpp:23`, `tests/ini_input.cxx` |
| FCIDUMP read (norb, core, 1-body, 2-body), `is_2body_diagonal`, write | `include/macis/util/fcidump.hpp`, `src/macis/fcidump.cxx` |
| Active-space Hamiltonian, inactive Fock and energy, generalized Fock | `include/macis/util/fock_matrices.hpp`, `src/macis/fock_matrices.cxx` |
| Wavefunction file read/write (ASCII dets + C) | `include/macis/wavefunction_io.hpp` (`read_wavefunction :67`, `write_wavefunction :172`) |
| Write matrix/vector to text | `include/macis/util/general_io.hpp` (`util::write_matrix`) |
| Integral basis transform `Y = Cᵀ X C` | `include/macis/util/transform.hpp`, `src/macis/transform.cxx` |
| 1D Hubbard model integrals (tests) | `include/macis/model/hubbard.hpp` |

### 4.2 Determinants and bit operations
| Feature | Location |
|---|---|
| HF determinant (raw, energy-ordered, by occupation) | `sd_operations.hpp:32,57,74` |
| Singles/doubles generators, CIS/CISD/full Hilbert spaces, combinations | `sd_operations.hpp:148–580` |
| Fermionic excitation signs | `sd_operations.hpp:594–645` |
| Determinant ↔ string (`to_canonical_string`, `from_canonical_string`) | `sd_operations.hpp:687,706` |
| Bitset helpers (lo/hi word, ffs, clz, masks) | `bitset_operations.hpp` |
| Orbital permutation of a det, group expansion, orbit representatives | `asci/determinant_symmetry.hpp` |
| Group-covariant natural orbitals (SYMMETRIZE_DETS with NROTS > 0), natural-orbital blocks | `asci/covariant_orbitals.hpp` |

### 4.3 Hamiltonian, matrix elements, RDMs, rotations
| Feature | Location |
|---|---|
| `HamiltonianGenerator<N>` base (Tu/Td/V, intermediates, matrix elements) | `hamiltonian_generator.hpp`, `hamiltonian_generator/{impl,matrix_elements,fast_diagonals}.hpp` |
| Spin-dependent one-body (`ReadTdo`) | `hamiltonian_generator/impl.hpp:27` |
| **`SDBuildHamiltonianGenerator`** (production; `SetJustSingles`, `SetNimp`) | `hamiltonian_generator/sd_build.hpp` |
| `DoubleLoopHamiltonianGenerator` (reference implementation used by tests) | `hamiltonian_generator/double_loop.hpp` |
| CSR Hamiltonian build (32/64-bit index) and index-overflow guard | `csr_hamiltonian.hpp`, `util/csr_index.hpp` |
| 1-/2-RDMs from a CI vector (`form_rdms`, spin-resolved overload) | `sd_build.hpp:177,263`; kernels in `util/rdms.hpp` |
| Natural-orbital rotation: full space / impurity+bath blocks / given matrix | `hamiltonian_generator/rdms.hpp:23` (`rotate_hamiltonian_ordm`), `:122` (`_imp_bath`, per-group option), `rotate_hamiltonian_rotmat_imp_bath` |
| 1-RDM rotated back to the original basis | `evaluate_ordm`, `impurity_solver.hpp:681` |

### 4.4 Eigensolvers
| Feature | Location |
|---|---|
| Davidson (serial `davidson :134`, distributed `p_davidson :435`) | `solvers/davidson.hpp` |
| Selected-CI diagonalization driver (builds H, calls Davidson) | `solvers/selected_ci_diag.hpp:171` |
| LOBPCG wrapper | `solvers/lobpcg.hpp`, `src/lobpcgxx/` |
| Lanczos (single band, GS energy/vector, matrix-op wrappers incl. distributed sparse) | `gf/lanczos.hpp` |
| Band Lanczos and band resolvent, lowest Ritz value | `gf/bandlan.hpp`, `src/macis/gf/bandlan.cxx` |
| Tridiagonal/dense LAPACK eigen wrappers | `gf/eigsolver.hpp`, `src/macis/gf/eigsolver.cxx` |
| CASSCF / MCSCF (upstream; DIIS, BFGS, orbital gradient/Hessian) | `util/mcscf*.hpp`, `util/orbital_*.hpp`, `util/diis.hpp`, `src/macis/bfgs/`, `src/macis/{casscf,orbital_*}.cxx` |
| MP2 natural-orbital guess (upstream) | `util/moller_plesset.hpp`, `src/macis/moller_plesset.cxx` |

### 4.5 ASCI
| Feature | Location |
|---|---|
| `ASCISettings` (all ASCI knobs, symmetry, parity target) | `asci/determinant_search.hpp:37` |
| Grow / iterate / refine | `asci/grow.hpp`, `asci/iteration.hpp`, `asci/refine.hpp` |
| Candidate scoring (singles, same-spin and opposite-spin doubles) | `asci/determinant_contributions.hpp` |
| MPI constraint-partitioned search (triplets/quads, histograms) | `asci/mask_constraints.hpp`, `util/dist_quickselect.hpp` |
| Symmetric orbit selection (SYMMETRIZE_DETS) | `asci/determinant_symmetry.hpp:148` |
| Parity labels, masks, keys, `ParityTarget` | `asci/parity_labels.hpp` |

### 4.6 Impurity-specific layers (fork additions)
| Feature | Location |
|---|---|
| `impurity_params<N>` — **the state struct passed everywhere** | `impurity_solver.hpp:47` |
| Solver entry points, parity dispatch, guess loading, HF seed | `src/macis/impurity_solver.cpp` (§3.1) |
| Permutation-symmetry validation and expansion | `prepare_det_symmetry`, `impurity_solver.cpp:168` |
| **µ search**: cost functions, GSL brent/bisection/falsepos (no derivative) and newton/secant/steffenson (with derivative), auto-bracketing | `doping/fix_mu.hpp`, `src/macis/doping/fix_mu.cpp` (`Mu_Cost_f :177`, `Fix_Mu_der :470`, `Fix_Mu_noder :598`) |
| Writing µ (and crystal-field splitting) onto the impurity diagonal | `set_impurity_diagonal`, `fix_mu.cpp:24` |
| **Charge-sector search** (`ChargeSectorSettings`, seeding by c†/c, `SectorScan` walk/window, `Fix_Mu_sectors`) | `doping/charge_sectors.hpp`, `src/macis/doping/charge_sectors.cpp` (`Fix_Mu_sectors :540`) |
| `GS_charge_sector.dat` writer | `write_ground_sector_file`, `charge_sectors.cpp:518` |
| **Band-parity sectors** (label detection, integral cleaning, key enumeration, stabilizer, seed moves) | `parity_sectors.hpp`, `src/macis/parity_sectors.cpp`; driver `setup_parity_sectors`, `impurity_solver.cpp:1351` |

### 4.7 Dynamical quantities
| Feature | Location |
|---|---|
| `GFSettings` (all GF.* keys) | `gf/gf.hpp:41` |
| GF driver: particle+hole, spin average, sector check, back-rotation, band average | `evaluate_GF`, `impurity_solver.hpp:349` |
| GF basis build (add/remove one electron in the active space), Lanczos seeds | `gf/gf.hpp:236` (`get_GF_basis_AS_1El`), `:412` (`BuildWfn4Lanczos`) |
| Core GF calculation (one sector; pads dropped orbitals with zeros; returns E(N±1) bound) | `RunGFCalc`, `gf/gf.hpp:562` |
| GF frequency grid and file writers (`GF.dat`, `GFMatrix_*`) | `src/macis/gf/gf.cxx` |
| `SECTOR_CHECK` (sign of E(N±1)−E(N) bounds) | `report_sector_check`, `impurity_solver.hpp:122` |
| GF spin-flip average | inside `evaluate_GF`, `impurity_solver.hpp:435` |
| GF band-orbit average under the sym group | `band_orbit_average_gf`, `impurity_solver.hpp:189` |
| Ground-state resolvent, diagonal-operator application, impurity Sz resolvent | `gf/dynamical_properties.hpp` (`RunResolventGS :70`, `RunResolventSz :197`); driver `evaluate_resolvent_sz`, `impurity_solver.hpp:615` |

### 4.8 Static observables
| Feature | Location |
|---|---|
| `CompObservables<N>` (builds spin-resolved 1-/2-RDMs in the original basis) | `comp_observables.hpp`, `src/macis/comp_observables.cpp` |
| Double occupancy (total and per-band matrix) | `comp_observables.cpp:208,227` |
| ⟨Sz_i Sz_j⟩, ⟨τz_i τz_j⟩ (per site), charge–charge (per orbital) | `comp_observables.cpp:272,307,352` |
| Impurity many-body RDM ρ = Tr_bath\|Ψ⟩⟨Ψ\| — dense reference | `observables/impurity_rdm.hpp:332` (`compute_impurity_rdm_from_state`) |
| Impurity RDM — block-sparse by (N_imp, Sz) sector, writers `rho.bin` / `.txt` / `rho_blocks.bin` | `observables/impurity_rdm.hpp:369+` (`ImpRdmBlocks`, `compute_impurity_rdm_blocks_from_state :803`) |

### 4.9 Parallel and utility code
| Feature | Location |
|---|---|
| MPI helpers (`comm_rank`, datatypes, `MACIS_MPI_CODE` macro) | `util/mpi.hpp` |
| Memory size helpers (`to_gib`) | `util/memory.hpp` |
| Distributed quickselect (top-k) | `util/dist_quickselect.hpp` |
| Sparse matrix types, distributed SpMV, MatrixMarket IO | `src/sparsexx/include/sparsexx/` |

---

## 5. `input.in` keys (`run_asci_impsolv_dop`)

**The authoritative list is the `getData`/`OPT_KEYWORD` calls in `main/run_asci_impsolv_dop.cxx`.**
An unknown key is ignored silently. Format: `[SECTION]` then `KEY = value`. The code
addresses each key as `SECTION.KEY`.

| Section | Keys (→ field) |
|---|---|
| `CI` | **`FCIDUMP`, `NALPHA`, `NBETA`** (required); `EXPANSION` = `CAS`\|`ASCI`\|`ASCI_cheap`; `FCIDUMP_DO`; `JUST_SINGLES`; `NINACTIVE`; `NACTIVE`; `NIMP`; `NBANDS`; `DOPING`; `GF`; `COMP_DB_OCCS`; `COMP_SZ_I_SZ_J`; `COMP_TAUZ_I_TAUZ_J`; `COMP_CHARGE_CHARGE`; `COMP_IMP_RDM`; `WRITE_RHO_BIN`; `WRITE_RHO_TXT`; `WRITE_RHO_BLOCKS`; `RHO_BLOCK_THRESH`; `RDM_DENSE_REF`; `RDMFILE`; `FCIDUMP_OUT` |
| `ASCI` | `NTDETS_MAX`, `NTDETS_MIN`, `NCDETS_MAX`, `HAM_EL_TOL`, `RV_PRUNE_TOL`, `PAIR_MAX_LIM`, `GROW_FACTOR`, `MAX_REFINE_ITER`, `REFINE_ETOL`, `GROW_WITH_ROT`, `NROTS`, `ROT_SIZE_START`, `CONSTRAINT_LVL`, `HF_BY_ENERGY`, `SYMMETRIZE_DETS`, `SYM_TOL`, `SYM_NPERM`, `SYM_PERM_1..n`, `WFN_FILE`, `WFN_OUT_FILE`, `E0_WFN`, `PARITY_SOLVE` (default **TRUE**), `PARITY_TOL`, `PARITY_ETOL`, `PARITY_ONLY` |
| `DOP` | `NELECTRONS` (electrons per impurity orbital, summed over spin), `INIT_MU`, `DERIV`, `METHOD`, `ABS_TOL`, `MAXITER`, `DSTEP`, `INIT_SHIFT`, `PRINT_DOPING`, `DELTA_CFS`, `CHEAP_MODE`, `SECTOR_SEARCH`, `SECTOR_MARGIN`, `SECTOR_ETOL`, `SECTOR_WARM`, `SECTOR_MAX_SWITCH`, `SECTOR_DIR` |
| `GF` | `NORBS`, `TRUNC_SIZE`, `TOT_SD`, `GFSEEDTHRES`, `ASTHRES`, `USE_BANDLAN`, `NLANITS`, `WRITE`, `PRINT`, `SAVEGFMATS`, `ORBS_BASIS`, `IS_UP_BASIS`, `ORBS_COMP`, `IS_UP_COMP`, `WMIN`, `WMAX`, `NWS`, `ETA`, `BETA`, `IMAG_FREQ`, `SPIN_AVERAGE`, `BAND_AVERAGE`, `SZ_RESOLVENT` |
| `MCSCF` | `MAX_MACRO_ITER`, `MAX_ORB_STEP`, `MCSCF_ORB_TOL`, `ENABLE_DIIS`, `DIIS_START_ITER`, `DIIS_NKEEP`, `CI_RES_TOL`, `CI_MAX_SUB`, `CI_MATEL_TOL`, `CI_NSTATES`, `MP2_GUESS`. The `CI_*` keys are the **Davidson settings used by ASCI and ED**, not only by MCSCF |
| `PRINT` | `DAVIDSON`, `CI`, `MCSCF`, `DIIS`, `ASCI_SEARCH`, `DETERMINANTS`, `DETERMINANTS_THRES`. `OFF` nulls the matching spdlog logger |

Option interactions enforced in code. Search the error string to find each check:

- `WFN_FILE` requires `E0_WFN`, `NROTS = 0` and `MAX_REFINE_ITER > 0`.
- `SYMMETRIZE_DETS` requires `SYM_NPERM ≥ 1`, and it rejects `GROW_WITH_ROT`. With `NROTS > 0`
  the rotations are group-covariant (`covariant_orbitals.hpp`); a group that permutes orbitals
  inside an (imp|bath) × band block throws.
- `PARITY_SOLVE` requires `GROW_WITH_ROT = FALSE`, `NINACTIVE = 0` and `NACTIVE = NORB`.
- The µ search requires `NINACTIVE = 0`.
- A nonzero `DELTA_CFS` is supported only for `NBANDS` = 2 or 3. For any other value it is reset to 0.

## 6. Output files (cwd of the run)

| File | Written by |
|---|---|
| stdout (`asci.out` on the DMFT side): `E(CI) = …`, `Occs: …`, `SECTOR_CHECK …`, `GF_SPIN_AVERAGE …`, `GF_BAND_AVERAGE …`, `PARITY_*` | driver, solver and GF code |
| `GF.dat` | `write_GF` (`src/macis/gf/gf.cxx:123`) when `GF.WRITE` |
| `GFMatrix_*.dat`, `GFMatrix_OrbitalIndices_{ADD,SUB}.dat` | `GF.SAVEGFMATS` |
| `Sz_resolvent.dat` | `evaluate_resolvent_sz` |
| `locFCIDUMP.dat` | the doping branch of the driver: impurity block at the converged µ. **DMFT reads µ back from it** |
| `GS_charge_sector.dat` | the doping branch (`write_ground_sector_file`) |
| `active_ordm.dat`, `rot_matrix.dat`, `rot_matrix_<i>.dat` | `SolveImpurityASCI_rot` |
| `rho.bin`, `rho.bin.txt`, `rho_blocks.bin` | `CI.COMP_IMP_RDM` |
| `db_occs_matrix.dat`, `sz_sz.dat`, `tauz_tauz.dat`, `charge_charge.dat` | the matching `CI.COMP_*` flags |
| `ASCI.WFN_OUT_FILE` | `write_wavefunction` |
| `charge_sectors/` (`DOP.SECTOR_DIR`) | the sector search's scratch files and seeds |

---

## 7. Conventions that bite

- **µ sign.** The FCIDUMP impurity diagonal holds **−µ_phys**. The root-search variable
  (`DOP.INIT_MU`, `Mu_Cost_f`'s `x`) is the **value written onto that diagonal**. So the
  solved H is already H − µN, and E(CI) of different sectors compares directly (it is the
  grand potential).
- **`set_impurity_diagonal` overwrites** the impurity diagonal with µ, or with µ ± a
  `DELTA_CFS` pattern. It does **not** add to it. Any on-site energy in the impurity block
  of the FCIDUMP diagonal is replaced during a µ search. The function writes T and Td
  identically.
- **Occupations.** `params.occs[i]` is **per spin** (½ of the 1-RDM diagonal) and printed in
  the **original basis**. `DOP.NELECTRONS` is compared against `2·mean(occs[0:n_imp])`.
- **Orbital layout.** Impurity orbitals come first (`0..n_imp−1`), arranged **band-major**:
  orbital i belongs to band `i / (n_imp/nbands)`. Bath orbitals follow. Parity grouping
  attaches each bath orbital to the band it hybridizes with.
- **Storage.** Matrices are column-major (`T[p + q·n]`).
  `V[p + q n + r n² + s n³] = (pq|rs)`.
- **`orb_rot`** is the cumulative rotation (n_active², column-major). The GF back-rotation
  uses only its impurity block, and throws if that block is not unitary.
- **HF seed.** The seed is energy-ordered by default (`HF_BY_ENERGY`). The determinants
  inherit every quantity H conserves, parity included, from the seed.
- **Logging.** spdlog loggers are named `davidson`, `ci_solver`, `asci_search`,
  `asci_refine`, `mcscf`, `diis`. Other ranks get null loggers.

## 8. Design notes and reviews in this repo (read before changing a layer)

| File | Covers | Status |
|---|---|---|
| `FORK_REVIEW.md` | Severity-ranked review of the fork against upstream, with FIXED/open markers | living reference |
| `PLAN_symmetric_asci_search.md` | SYMMETRIZE_DETS design; §12 G-covariant rotations for `NROTS > 0` | Steps 1–10 done, Step 11 open; §12 **implemented** 2026-10-11 (as built: §12.9), toy tests pass, no production run |
| `parity-sector-solve-simple.md` | Band-parity sector solve as built (§11–12 list what was built and where) | Phases 1–2 implemented |
| `symmetry-sector-solve.md` | Consolidated plan for symmetry sectors beyond parity | Partly implemented |
| `PLAN_explore_charge_sectors.md` | `explore_charge_sectors` and the charge-sector search | implemented |
| `gf_band_average_full_group.md` (+ `gf_degenerate_average_toy_ed.py`) | Proposal: average the GF over the whole group | **not implemented** |
| `gf_csr_int32_overflow.md` | Why the GF path uses 64-bit CSR | fixed |
| `hole_particle_mismatch.md` | GF padding of dropped orbitals | fixed |
| `symmetry_sector_files/` | Older parity drafts and the independent assessment | superseded |

## 9. Tests (`tests/`, Catch2 → `macis_test`)

Each file is named after the layer it tests:

- **Upstream core:** `bitset_operations`, `sd_operations`, `fcidump`, `read_wavefunction`,
  `double_loop`, `sd_build`, `csr_hamiltonian`, `davidson`, `transform`, `fock_matrices`,
  `mcscf`, `asci`, `dist_quickselect`, `hubbard`.
- **Fork additions:** `determinant_symmetry`, `impurity_rdm`, `dynamical_properties`,
  `charge_sectors`, `impurity_seed`, `parity_sectors`, `gf_spin_average`, `gf_band_average`,
  `gf`, `ini_input`.

`ini_input.cxx` is a test file, and it also compiles the INI parser into every executable.
Reference data is in `tests/ref_data/`. To type-check a header change without a full build,
use the recipe in `../CLAUDE.md`.

## 10. Adding something — where it goes

- **New input key.** Add an `OPT_KEYWORD` in `main/run_asci_impsolv_dop.cxx`. Add the field
  to `impurity_params`, `ASCISettings`, `GFSettings` or `ChargeSectorSettings`. Mirror it in
  `explore_charge_sectors.cxx` if that driver needs it. Add the key to §5 above.
- **New static observable.** Add a method to `CompObservables` (header, plus `.cpp` with the
  `<64>` instantiation). Then add a `CI.COMP_*` flag and its output block in the driver.
- **New dynamical quantity.** Put the kernel in `gf/dynamical_properties.hpp` and the driver
  function next to `evaluate_resolvent_sz` in `impurity_solver.hpp`. Gate it with a `GF.*`
  key.
- **A change to the solve itself.** Modify `solve_*_one` in `impurity_solver.cpp`. The parity
  wrapper and the µ / sector loops all reach it through `SolveImpurity*`, so one change
  covers every caller.
- **Templates.** Anything templated on `N` and defined in a `.cpp` needs an explicit `<64>`
  instantiation at the bottom of that file.
