# gf_csr_int32_overflow: 64-bit CSR indices for the GF Hamiltonians

## Symptom

`SOLVER_TESTS/GFbasis_all/U_4.0` (job 58891771) segfaulted in the hole sector,
at the first step of band Lanczos:

```
---> FINAL ADD BASIS HAS 49594822 ELEMENTS
Time to build hole Hamiltonian: 79m 2.657e+01s
ORBITALS WITH NO CORRESPONING ADD-VECTOR: []
RESOLVENT ROUTINE: QR DECOMPOSITION ...DONE! BAND LANCZOS ...[...] UCX  WARN  ucs_debug_disable_signal: signal 8 was not set in ucs
```

The job used `ORBS_BASIS = 0…23`, `ASTHRES = 1e-6`, `TRUNC_SIZE = 1e8`, 1.5M
ground-state determinants, N = 14, U = 4 and NROTS = 3. This is the §6.4 test
in `SOLVER_HEALTH.md`.

- The particle sector (30,240,304 rows) completed.
- The hole sector (49,594,822 rows) built its Hamiltonian and then crashed in
  the first sparse matrix-vector product.
- Peak RSS was 55 GB of the 400 GB allocated, so this was not an OOM.
- addr2line placed the crash at `V_k[Aci_st[j]]` in the sparsexx SpMV
  (`src/sparsexx/include/sparsexx/spblas/spmbv.hpp:68`).

## Root cause

The GF Hamiltonian was stored as `csr_matrix<double, int32_t>`, and its row
pointer overflowed.

- `RunGFCalc` has `template <size_t nbits, typename index_t = int32_t>`
  (`include/macis/gf/gf.hpp`). `evaluate_GF` (`include/macis/impurity_solver.hpp`)
  called `RunGFCalc<N>`, so the N±1 Hamiltonian was built with 32-bit indices.
- `sparsexx::csr_matrix` uses one `index_t` for both `rowptr` and `colind`,
  while `nnz_` is `int64_t`, taken from `nzval.size()`.
- The builders (`include/macis/hamiltonian_generator/sd_build.hpp` and
  `double_loop.hpp`) did `rowptr[i + 1] = rowptr[i] + nrow;` with no check.
  Once the running nonzero count passes `INT32_MAX` = 2,147,483,647, this is
  signed overflow: undefined behaviour, which in practice wraps negative.
- For 49,594,822 rows, overflow needs only
  2,147,483,647 / 49,594,822 = **43.3 nonzeros per row**. The ASCI space already
  has 31.8 per row (47,743,424 nnz over 1.5M dets). A single-and-double GF space
  built from all 24 orbitals is more closed under single excitations, so it is
  denser.
- The particle sector, at 30.2M rows, would need more than 71 per row to
  overflow. It survived, which fits.
- In the SpMV, the loop `for j in [rowptr[i], rowptr[i+1])` then reads `colind`
  through corrupt offsets, which segfaults. Any memory error is fatal, so
  OpenMP is not the cause. At most it changes when the crash happens.
- Only the row pointer overflows. Column indices (< 49.6M) and the `int j` /
  `uint32_t id` determinant positions in `sd_build.hpp` still fit.

**Confidence:** high but indirect. The nnz of the hole matrix was never
printed, so the overflow is inferred from the row count, the density, the
crash site and the code path. The guard added below turns any future case into
an exception that states the nnz, so it can be confirmed directly.

## Fix (plan)

Goals:
- Build the GF Hamiltonians with 64-bit indices.
- Keep the ASCI and Davidson Hamiltonians at 32 bits to save memory.
- Turn any future overflow into a clear exception instead of undefined
  behaviour.

Parts already templated on the index type and reused as they are:
- `HamiltonianGenerator::make_csr_hamiltonian_block<index_t>`, which already
  dispatches to `make_csr_hamiltonian_block_64bit_`.
- `make_dist_csr_hamiltonian<index_t>`.
- `SparseMatrixOperator<SpMatType>` (`solvers/davidson.hpp`).
- `sparsexx::spblas::pgespmv` / `spmv_info<index_type>`.
- `RunGFCalc`, `BuildWfn4Lanczos` and `RunResolventGS`.

Only two wrappers were hard-coded to `int32_t`, so changing `RunGFCalc` alone
would not have compiled: `BandResolvent` and `SparsexDistSpMatOp`.

Steps:
1. Add a checked row-pointer increment and a dimension check to both CSR
   builders.
2. Template `BandResolvent` and `SparsexDistSpMatOp` on the index type.
3. Call `RunGFCalc<N, int64_t>` from `evaluate_GF` and from the test drivers.
4. Add tests.

### Out of scope (follow-ups)

- **Dense Lanczos vectors indexed with `int`.** `BandResolvent`, `BandLan`
  and `BuildWfn4Lanczos` index `vecs[j * len_vec + i]` and take
  `int len_vec` / `int nvecs`. That overflows once `nvecs × nterms > 2^31`:
  8 orbitals × 268M dets, or 24 orbitals × 89M dets. The current run is at
  8 × 49.6M ≈ 4.0e8, so it is safe. It becomes the next limit if `ORBS_COMP`
  grows or the GF space passes about 268M determinants.
- **`int j = it->id` in `sd_build.hpp`,** with `det_pos::id` as `uint32_t`.
  This caps a matrix at 2^31 rows. Also not reached here.
- **Other resolvent paths stay 32-bit:** `RunResolventSz` / `RunResolventGS`,
  and the ASCI Davidson in `selected_ci_diag.hpp`. They act on the ground-state
  space, which is much smaller. With the guard they now fail with a clear
  exception rather than silently.
- **Memory.** 64-bit indices widen `colind` too (sparsexx has a single
  `index_t`), so they cost +4 bytes per nonzero. That is about +33 % of CSR
  storage: at least +8.6 GB for a matrix at the 2^31 boundary, about 12 GB at
  3e9 nnz. Transient growth of `std::vector` during the build can add more.
  The ASCI Hamiltonian is unaffected.

## Implementation

| File | Change |
|---|---|
| `include/macis/util/csr_index.hpp` (new) | `check_csr_dims<index_t>(nrows, ncols)` and `checked_rowptr_next<index_t>(prev, nrow, row, nrows)`. Both throw `std::overflow_error`; the message names the row, the running nnz and the index width, and suggests `index_t = int64_t`. |
| `include/macis/hamiltonian_generator.hpp` | Includes `csr_index.hpp`. |
| `include/macis/hamiltonian_generator/sd_build.hpp`, `double_loop.hpp` | Check dimensions before the build. `rowptr[i + 1] = checked_rowptr_next(rowptr[i], nrow, i, nbra_dets)`. The cost is one compare per row. |
| `include/macis/gf/lanczos.hpp` | `SparsexDistSpMatOp` becomes `template <typename index_t = int32_t>`, holding `spmv_info<index_t>`. |
| `include/macis/gf/bandlan.hpp`, `src/macis/gf/bandlan.cxx` | `BandResolvent` becomes `template <typename index_t>`. The definition stays in the `.cxx`, with explicit instantiations for `int32_t` and `int64_t`. The body is unchanged: `SparseMatrixOperator Hop(H)` already deduces the type. |
| `include/macis/gf/gf.hpp` | The non-band Lanczos path uses `SparsexDistSpMatOp<index_t>`. |
| `include/macis/gf/dynamical_properties.hpp` | `RunResolventGS` uses `SparsexDistSpMatOp<index_t>`. |
| `src/macis/gf/gf.cxx` | Explicit instantiations of both `RunGFCalc<64, int64_t>` overloads. |
| `include/macis/impurity_solver.hpp` | `evaluate_GF` calls `RunGFCalc<N, int64_t>` for both sectors. |
| `tests/test_driver.cxx`, `test_driver_dop.cxx`, `standalone_driver.cxx` | `RunGFCalc<nwfn_bits, int64_t>`, so every entry point matches. |

The ASCI and Davidson path (`selected_ci_diag.hpp`) is untouched and stays
at `int32_t`.

## Tests

Added to `macis_test`:

- **`tests/csr_hamiltonian.cxx`: "CSR index overflow guard"**
  - `checked_rowptr_next<int16_t>` reaches exactly 32767 without throwing and
    throws one past it.
  - `2^31` nonzeros throw for `int32_t` and pass for `int64_t`.
  - `check_csr_dims<int16_t>` throws when either dimension exceeds 32767.
- **`tests/csr_hamiltonian.cxx`: "CSR Hamiltonian 64-bit indices"**
  - Builds the water cc-pVDZ CISD Hamiltonian with `int32_t` and with `int64_t`.
  - Requires identical dimensions, nnz, `rowptr`, `colind` and `nzval`.
- **`tests/gf.cxx` (new, added to `tests/CMakeLists.txt`): "RunGFCalc - 32-bit
  and 64-bit sparse indices agree"**
  - System: 4-orbital Hubbard-like model, full 2α2β sector, exact ground state.
  - Runs `RunGFCalc<64, int32_t>` and `RunGFCalc<64, int64_t>` for particle and
    hole, with band Lanczos and with regular Lanczos. This covers the templated
    `BandResolvent` and `SparsexDistSpMatOp<int64_t>`.
  - Requires the same `todelete` and the same GF to 1e-12, and a nonzero GF so
    that two all-zero results can't pass.

Results (1 MPI rank, Release build):

- The three new test cases and the existing "CSR Hamiltonian" test pass.
- Full `macis_test`: 32 of 33 test cases pass. The one failure is
  `ASCI Symmetric Search` (`tests/determinant_symmetry.cxx:318`, energy
  difference 0.286 against a 0.1 limit). It fails identically on the unmodified
  `HEAD`, so it predates this change.
- All executables build: the three test drivers, `run_asci_impsolv_dop`,
  `run_asci_impsolv_mu_vs_n` and `charge_sector_estimate`.

Not covered by unit tests:
- The builder guard firing inside a real Hamiltonian build. The helper is
  tested directly; forcing a real overflow needs more than 2^31 nonzeros.
- Multi-rank MPI runs.

## Verification still to do on the cluster

1. Rebuild the production binary, then rerun a known causal case, for example
   `1×2 Irrep U=4 It_5`. `GF.dat` must match the old binary to round-off.
2. Rerun `SOLVER_TESTS/GFbasis_all/U_4.0`.
   - Both sectors should finish band Lanczos.
   - Peak RSS in `error_asci.err` should come out at about 55 GB plus roughly
     10 GB.
   - Then read T4 with `test_solver_health.py`, against +0.599.
