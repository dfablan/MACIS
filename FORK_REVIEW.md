# Code review — `dfablan/MACIS` fork vs. upstream `wavefunction91/MACIS`

**Baseline:** upstream `feature/spin_dep` @ `984aa834` (which already contains upstream
`feature/gf` @ `8930b91`).
**Reviewed:** `feature/spin_dep` @ `cd23952` (local HEAD).
**Diff:** 37 files, +6266 / −72.
**Re-reviewed 2026-08-24:** the fix round sitting unstaged on top of `b64251d`
(16 files, +520 / −289) was verified against each FIXED claim below. Verdicts are
annotated in place; new findings from the re-review are in the
"Re-review of the fix round" section, and the severity-ranked list of what
remains open is at the end.

```
git diff 984aa8345c5ed60a38556ed5dfaada4d7519ef63..HEAD
```

Everything below was checked by reading the changed ranges and the surrounding
upstream code. Findings are ordered by severity. Purely stylistic points are
collapsed into the last section.

---

## 1. Correctness bugs that silently produce wrong numbers

### 1.1 `Mu_vs_n` reads an empty vector and never applies `mu` — `src/macis/doping/fix_mu.cpp:32,118-124` - FIXED

```cpp
std::vector<double> occs;              // line 32 — local, EMPTY, shadows p->occs
...
E = SolveImpurityASCI<N>(*p);          // fills p->occs, not this one
...
for (int i = 0; i < n_imp; i++)
    std::cout << "occs[" << i << "] = " << occs[i] << std::endl;   // OOB read
curr_nel_per_spin = std::accumulate(occs.begin(), occs.begin()+n_imp, 0.0);
```

`occs` is a local empty vector. `occs[i]` and `occs.begin()+n_imp` are undefined
behaviour (heap over-read past `end()`), and the returned electron count is
whatever garbage happens to be there.

Compounding this, `Mu_vs_n` writes the chemical potential into `p->T` but — unlike
its near-twin `Mu_Cost_f`, which calls `macis::active_hamiltonian(...)` at line 228 —
**never propagates `T` into `T_active`**. `T_active` is the array the solvers actually
consume, so even with the `occs` bug fixed the routine solves the *same* Hamiltonian
at every `mu`.

**Why it matters:** `Mu_vs_n` is the entire payload of the `run_asci_impsolv_mu_vs_n`
driver. Every point on the μ-vs-n curve it produces is meaningless. The two functions
share ~90 lines of duplicated crystal-field-splitting logic, which is exactly why the
fix landed in one and not the other — they should be one function with the solve
call parameterised.

### 1.2 ASCI ground state is built in the wrong spin sector — `src/macis/impurity_solver.cpp:169` - FIXED

```cpp
dets = {macis::canonical_hf_determinant<N>(nalpha, nalpha)};   // SolveImpurityASCI
```

versus, 115 lines later in `SolveImpurityASCI_rot`:

```cpp
macis::wfn_t<N> hf_det = macis::canonical_hf_determinant<N>(nalpha, nbeta);  // :284
```

The signature is `canonical_hf_determinant(nalpha, nbeta, ...)`. `SolveImpurityASCI`
passes `nalpha` twice, so for any `nalpha != nbeta` the reference determinant has the
wrong number of β electrons. ASCI only ever generates single and double excitations,
which conserve per-spin particle number, so the *entire* expansion stays in the wrong
(N, S_z) sector.

**Why it matters:** no error is raised — you get a plausible-looking energy for the
wrong electron count. `Mu_vs_n` (§1.1) dispatches to exactly this function, so a
spin-polarised μ search converges to the filling of a sector that was never requested.

### 1.3 Chemical potential is applied to spin-up only — `src/macis/doping/fix_mu.cpp:96-98,166-224,228` - FIXED

`Mu_Cost_f` writes `mu` into `T.at(i*norb+i)` and then rebuilds `T_active` via
`active_hamiltonian`. It never touches `p->Td` and never rebuilds `p->Td_active`.

In `main/run_asci_impsolv_dop.cxx:99-104` a second FCIDUMP sets `params.Td` and
`params.spin_dep = true`; `Td_active` is computed once at driver setup (lines 265-269)
and never again. `SolveImpurity*` then calls `ham_gen.ReadTdo(Td_active)`.

**Why it matters:** in every spin-dependent run the μ shift is applied to the ↑ channel
and not the ↓ channel. That is not a chemical potential, it is a Zeeman field, and the
root finder converges to a filling that has no relation to the intended one.

*Re-review:* verified fixed. The duplicated CFS logic is now one helper
(`set_impurity_diagonal`), applied to `T` and, when `spin_dep`, to `Td`; both
`T_active` and `Td_active` are rebuilt via `active_hamiltonian` with exactly the
argument pattern the driver uses at setup (`run_asci_impsolv_dop.cxx:256-267`), and
`Mu_vs_n` now does the same (fixing the missing-`T_active` half of §1.1 as well).
The `run_asci_impsolv_mu_vs_n` driver allocates every array the new code touches.

Related, still open: `T.at(i*norb+i) = mu` **replaces** the diagonal rather than
shifting it, and the original on-site energies are never saved. After the first call
they are gone. If the input FCIDUMP carries non-zero impurity on-site energies, they
are silently discarded. If this is the intended convention (impurity on-site energies
live in `delta_CFS`, the FCIDUMP diagonal is ignorable), say so in a comment; today a
reader cannot tell the discard is deliberate.

### 1.4 `Comp_db_occs` reports half the true double occupancy — `src/macis/comp_observables.cpp:89` - FIXED

`form_rdms` returns `ordm` and `trdm` in **different normalizations by construction**.
Every two-body kernel in `include/macis/util/rdms.hpp` folds in a factor of 1/2
(`:20`, `:37`, `:53`, `:79`), applied *after* the one-body `ordm` accumulation in the
same functions. Contracting the diagonal contribution for a single determinant with
N_up, N_down electrons gives `tr(ordm_u) = N_up` but
`sum_pq trdm_uu(p,p,q,q) = (N_up^2 - N_up)/2` (the p=q exchange term cancels the
direct one exactly) and `sum_pq trdm_ud(p,p,q,q) = N_up*N_down/2`. Summed over all
four spin blocks that is `N(N-1)/2`, i.e. `trdm = Gamma/2` where `Gamma` is the
standard 2-RDM normalized to `N(N-1)`.

So the `// Possible bug fix` doubling at `:186-193` is **correct**: it restores the
normalization consistent with `tr(ordm) = N`. Two things in the code confirm it:

* `Transform_2RDMs` (`:6`) adds the *undoubled* `ordm` onto `trdm_uu/dd` to convert
  `<c+_a c+_c c_d c_b>` into `<c+_a c_b c+_c c_d>`. That identity only holds if
  `trdm` is already in standard normalization — and the call at `:195` is correctly
  placed *after* the doubling. Check: one up-electron in orbital 0 gives
  `trdm_uu(0,0,0,0) = 0` from the kernel, plus `ordm_u(0,0) = 1`, equals `<n_0up> = 1`.
* `compute_charge_charge_correlations` (`:333-370`) is consistent, not broken: after
  the doubling `mean_nn` is `<n_i n_j>` and `mean_n_mean_n` is `<n_i><n_j>`, both in
  the same normalization. They would only be mismatched *without* the factor of 2.

The actual defect is that the same correction is **missing** at `:89`, in the
free function `Comp_db_occs`:

```cpp
orb_db_occs += trdm_ud[a + a * n_active + a * n_active2 + a * n_active3];
```

This value is printed at `:93` as "Double Occupancies (from 2-RDM)" directly
alongside "Double Occupancies (from WF)" at `:122`, which sums `|C|^2` over
determinants carrying a `'2'` at site `a` — the true double occupancy. The 2-RDM
line therefore prints exactly **half** the WF line every run.

**Fixed.** `:89` now takes `2.0 * trdm_ud[...]`, so both printed double occupancies
agree. The convention is documented at its origin (`include/macis/util/rdms.hpp`,
above the kernels), at the doubling loop in the `CompObservables` constructor
(replacing `// Possible bug fix`), and as a precondition on `Transform_2RDMs`, which
adds an undoubled `ordm` and therefore must run after the doubling.

The four `form_rdms` call sites in `src/macis/impurity_solver.cpp` were checked and
need no change: they read observables only from `active_ordm`, and pass
`active_trdm` straight to energy/CASSCF contractions, where the raw `Gamma/2`
normalization is the correct one.

### 1.5 FCIDUMP integral-first format is silently misparsed — `src/macis/fcidump.cxx:44-90` - FIXED

The `if(idx_first) {...} else {...}` dispatch was replaced by "try layout A, catch,
try layout B". `idx_first` is still computed at line 45 but is now used only for the
`idx_first and int_first` sanity throw — the layout decision it used to drive is gone.

The fallback never fires, because `std::stoi` does not throw on a leading numeric
prefix. For an integral-first line `0.5 1 1 1 1`:

- `std::stoi("0.5")` → `0` (stops at `.`, no exception)
- `p,q,r,s = 0,1,1,1`, all non-negative → no throw
- `std::stod("1")` → `1.0`
- returns `(0, 1, 1, 1, 1.0)` — **wrong indices, wrong integral, no error**

**Scope.** This is a lost capability, not an active corruption of the fork's own runs.
Two layouts are accepted: `<p> <q> <r> <s> <integral>` — what `write_fcidump` emits,
what every fixture in `tests/ref_data` uses, and what is used in production here — and
`<integral> <p> <q> <r> <s>`, the Molpro/PySCF convention that `pyscf.tools.fcidump`
writes. Index-first lines take the first `try` block, which is the correct branch, so
they parse identically before and after this fix. What the fork broke is the
integral-first layout, which went from supported to silently misparsed. No test covered
it — every fixture is index-first — so CI could not catch it.

Measured against the pre-fix parser on an integral-first file:

| line | pre-fix result | correct |
|---|---|---|
| `5.0000000000000000E-01 1 1 1 1` | `5 1 1 1 → +1.0` | `1 1 1 1 → +0.5` |
| `-1.2500000000000000E+00 1 1 0 0` | `1 1 0 0 → -1.25` | same |
| `1.2500000000000000E-01 0 0 0 0` | `1 0 0 0 → +0.0` | `0 0 0 0 → +0.125` |

Note the asymmetry: `std::stoi("-1.25…")` yields `-1`, which trips the negative-index
throw and therefore *does* reach the fallback, so **negative** integrals parsed
correctly while **positive** ones silently corrupted. That mix is why the failure was
easy to miss. Worse than wrong values, the core line degrades to `p q r s = 1 0 0 0`,
which `line_classification` calls `OneBody`; `read_fcidump_1body` then does `p--; q--`
and writes `T(0, -1)` — an out-of-bounds write. The old path was memory-unsafe on
integral-first input, not merely inaccurate.

Two more defects in the same block:
- `catch(const std::exception& e) { ... throw e; }` at `:86-90` **slices** the
  exception: it throws a copy of the static type `std::exception`, discarding the
  derived type and the message. This one affects index-first users too — it is the
  path that reports any malformed line.
- The first `catch(...)` swallows the deliberate `runtime_error("Invalid Orb Idx")`
  raised for negative indices, retrying a genuinely malformed line as the other layout
  instead of failing.

**Fixed.** The layout is now decided by *full-token consumption* rather than by parse
failure: `is_integer_token` requires `std::stoi` to consume the entire token
(`pos == str.size()`), which is the exact discriminator the old `is_float` heuristic
(`isalpha or '.'`) was reaching for. Indices and integrals go through strict
`parse_index` / `parse_integral` helpers that reject trailing garbage, and the dead
second `try` is gone. The single remaining `catch` rethrows a `runtime_error` carrying
both the underlying message and the offending line, replacing the slicing `throw e;`.

The one genuinely ambiguous case — all five tokens bare integers, e.g. `4 1 1 1 1` from
`%.16g` on an exactly-integral value — cannot be decided from a single line. It is
resolved to integral-first, which is both the FCIDUMP standard and the previous
upstream behaviour, and the tie-break is documented at the dispatch. `write_fcidump`
uses `{:25.16e}`, so files this code produces are never ambiguous.

Covered by a new `"Column Ordering"` section in `tests/fcidump.cxx`, with two
2-orbital fixtures holding identical integrals in the two layouts
(`tests/ref_data/layout.index_first.fci.dat` and
`tests/ref_data/layout.integral_first.fci.dat`, the latter carrying a Molpro `&FCI`
header to exercise header skipping). It anchors the index-first read against known
values, then requires `T` and `V` to agree element-wise across layouts. Verified to
fail against the pre-fix parser. Full suite green: 7,055,816 assertions in 24 test
cases.

### 1.6 GF back-rotation reads out of bounds; the guard against it can never fire — `include/macis/impurity_solver.hpp` — FIXED

`RunGFCalc` declared `todelete` as a local and dropped it, so the caller's
`todelete_p` / `todelete_h` were always empty. That made the
`todelete_h != todelete_p` consistency check dead, and made `sum_GFs` compute
`GFmat_size = GF_orbs_comp.size() - 0` — i.e. assume nothing had been dropped —
so it read past the end of every row whenever an orbital actually was dropped.
The same latent out-of-bounds existed in `tests/test_driver.cxx` and
`tests/test_driver_dop.cxx`.

**What changed:**

* `RunGFCalc` (`include/macis/gf/gf.hpp`) now takes `std::vector<int> &todelete`
  as a documented out-parameter, and fills the caller's vector instead of a
  discarded local. A thin overload with the old signature is kept for callers
  that genuinely do not need the list (`tests/standalone_driver.cxx`), with a
  comment saying when it is safe. Both are explicitly instantiated in
  `src/macis/gf/gf.cxx`.
* `sum_GFs` validates that the incoming matrices are actually
  `(GF_orbs.size() - todelete.size())^2` and throws a diagnostic naming the
  likely cause, instead of running off the end. It also rejects the degenerate
  case where every orbital was dropped.
* `evaluate_GF` forwards `todelete_p` / `todelete_h` to the two sector calls, so
  the `todelete_h != todelete_p` check is live. It is now a `throw` rather than a
  `std::cout`, since the two matrices index different orbital subsets in that
  case and summing them is meaningless.
* The back-rotation is guarded three ways: `G_n_orbs >= n_imp` before indexing
  `GF[iw][j + k*G_n_orbs]`; a rejection of the case where any orbital was dropped
  (GF row `j` no longer maps to impurity orbital `j`, and nothing in the code
  remaps it); and an explicit unitarity test on `rotMat`, which catches a
  non-block-diagonal `orb_rot` from the full `rotate_hamiltonian_ordm`
  (`grow_with_rot`) path directly, without needing to know which routine produced
  it.
* The declarations of `todelete_p` / `todelete_h` were missing from the working
  tree, leaving them referenced-but-undeclared at the `sum_GFs` and `write_GF`
  call sites; they are restored with a comment on why they must be forwarded.

**Behavioural note:** runs where an orbital's add/remove vector vanishes now abort
with a clear message instead of silently producing a mis-indexed GF. The index
remapping that would be needed to support that case is not implemented anywhere,
so failing loudly is the only correct option short of writing it.

### 1.7 `rot_matrix.dat` is written with mismatched bounds and stride — `src/macis/impurity_solver.cpp:462-469` - FIXED

```cpp
for (int i = 0; i < norb; i++)
  for (int j = 0; j < norb; j++)
    ofile_rot << orb_rot[i + j * n_active] << " ";
```

`orb_rot` has `n_active * n_active` elements. `norb` is the *total* orbital count
(active + inactive). Whenever `norb > n_active` this reads well past the end of the
vector; whenever `norb < n_active` it silently truncates the matrix. Both loop bounds
should be `n_active`.

### 1.8 τ_z–τ_z correlator is only valid for two bands — `src/macis/comp_observables.cpp:298-330` - FIXED

```cpp
double sign = (band_i == band_j) ? 1.0 : -1.0;
```

An orbital-isospin τ_z with ±1 eigenvalues exists only for a two-band manifold. The
fork explicitly added three-band support (`nbands == 3` in `fix_mu.cpp:58-86`, commit
`e1e342c`), and with `nbands == 3` this sign assignment is not τ_z for anything.
There is no guard and no comment stating the two-band precondition — the function just
returns numbers.

*Re-review:* the guard added at `comp_observables.cpp:313` was
`assert(n_bands_ == 2)`. `assert` is compiled out under `NDEBUG`, and this project is
built `CMAKE_BUILD_TYPE=Release` (see `MACIS_build/CMakeCache.txt`), which defines
`NDEBUG` — so in the builds actually being run the guard did not exist and a
three-band run still silently returned non-τ_z numbers.

**Fixed (2026-08-24).** Replaced with `if(n_bands_ != 2) throw std::runtime_error(...)`,
matching how every other precondition in this fork is enforced.

---

## 2. Regressions against upstream behaviour

### 2.1 `no_constraint_search` default flipped — breaks MPI ASCI — `include/macis/asci/determinant_search.hpp:47` - FIXED

```cpp
-  bool no_constraint_search = false;
+  bool no_constraint_search = true;
```

At `:472` this selects `asci_contributions_standard` over
`asci_contributions_constraint` for **every** rank count, not just `world_size == 1`.
The standard routine is serial: each rank walks all `ncdets` and builds an identical,
fully replicated pair list. Downstream:

- `:523` — `if(world_size == 1) sort_and_accumulate_asci_pairs(asci_pairs);`
  The comment above it reads *"MPI + Constraint Search already does S&A"*. With the new
  default that premise is false, so under MPI the pairs are never deduplicated or
  accumulated.
- `:600-636` — the distributed top-K runs `dist_quickselect` over `world_size` copies
  of every score (so the k-th ranked value is wrong by roughly that factor) and then
  `MPI_Allgatherv`s the kept strings, producing `world_size` duplicates of each
  determinant before truncating to `top_k_elements`.

**Why it matters:** this is a shared default in a public header. Any MPI run, including
upstream's own `standalone_driver`, now selects a determinant set that is both
mis-ranked and largely duplicated. If the fork's workflows are single-rank, set the
flag in the fork's own drivers rather than changing the library default.

**Fixed (2026-08-24).** Reverted the default to `false`, matching upstream. Confirmed
no driver or test in the fork sets `no_constraint_search` explicitly, so this only
restores the constraint-search path for `world_size > 1` and does not change behaviour
for any single-rank run. A run that genuinely wants the standard path under MPI can
still opt in per-run via `ASCISettings::no_constraint_search`.

### 2.2 `p_davidson` deadlocks on a rank-dependent collective — `include/macis/solvers/davidson.hpp:550-555` - FIXED

```cpp
size_t total_degenerate_count = 0;
if(degenerate_count > 0) {                                   // rank-local predicate
  total_degenerate_count = allreduce(degenerate_count, MPI_SUM, comm);   // COLLECTIVE
  logger->warn(...);
}
```

`allreduce` is collective over `comm` but is called only on ranks where
`degenerate_count > 0`. As soon as one rank has a near-degenerate diagonal element and
another does not, the run hangs. The `allreduce` must be unconditional, with the
logging gated on the result.

**Fixed (2026-08-24).** `allreduce(degenerate_count, MPI_SUM, comm)` now runs
unconditionally every iteration; the `logger->warn` is gated on the reduced
`total_degenerate_count > 0` instead. Exactly the fix the review proposed.

### 2.3 `p_davidson` now rejects ranks that own zero rows — `include/macis/solvers/davidson.hpp:386`

```cpp
if(N_local <= 0) throw std::runtime_error("Davidson: Invalid Matrix Size");
```

Upstream's guard was `if(N_local and !X_local) throw ...` — deliberately written so that
a rank with `N_local == 0` and `X_local == nullptr` is legal. `selected_ci_diag` passes
`H.local_row_extent()`, which is zero on trailing ranks whenever the CI dimension is
smaller than the rank count (routine in the early ASCI growth iterations). This turns a
supported configuration into a hard abort.

### 2.4 Davidson now reports unconverged results as converged — `davidson.hpp:250-270,304,506-524,563` - FIXED

Three new paths set `converged = true` without meeting `tol`:

- linear dependence detected mid-iteration → `converged = true; break;` (`:304`, `:563`)
- stagnation with `res_nrm < 100 * tol` → `converged = true` (`:268`, `:524`)

`converged` is the sole gate on `if(!converged) throw std::runtime_error("Davidson Did
Not Converge!")` (`:310`, `:569`). Upstream's contract was binary: reach `tol`, or throw.
Now the solver prints `"Davidson Converged!"` and returns an eigenvalue that may be 100×
looser than requested, or one from a broken-down Krylov space.

**Why it matters:** `asci_refine` compares successive energies against
`refine_energy_tol = 1e-6` and `Fix_Mu_*` root-finds on a filling derived from that
wavefunction. Silently accepting `100 * tol` breaks the error budget both loops assume.
Report the achieved residual and let the caller decide.

**Fixed (2026-08-24).** Both branches, in both the serial `davidson` and the parallel
`p_davidson`, now `throw std::runtime_error(...)` immediately instead of setting
`converged = true; break;`. The exception message carries the achieved residual norm
and the requested tolerance (plus the stagnant-iteration count, for that branch), so
the caller gets an honest, actionable diagnostic instead of a silently loosened
result. No code anywhere catches these exceptions around a `davidson`/`p_davidson`
call (checked `asci_refine.hpp`, `asci/grow.hpp`, `selected_ci_diag.hpp`), so this
restores the exact binary "reach tol, or throw" contract upstream had; nothing was
relying on the loosened behaviour to avoid a crash. In `p_davidson`, both the
stagnation decision (via `res_nrm`, itself already an `allreduce`d value) and the
`p_gram_schmidt` linear-dependence result are already identical across ranks before
this point, so every rank throws consistently — this does not introduce a new
divergent-collective risk on top of the §2.2 fix above.

### 2.5 `write_fcidump` can shrink `norb` on round-trip — `src/macis/fcidump.cxx:250-263`

Zero-valued integrals are now skipped. `read_fcidump_norb` (`:104-121`) derives `norb`
from the **maximum orbital index it sees in the file**. If the highest-index orbital
has all-zero integrals — entirely plausible for a decoupled bath orbital, and guaranteed
for the impurity diagonal when `mu == 0` and `delta_CFS == 0` — that orbital never
appears in the file and the dimension is silently lost on read-back.
`main/run_asci_impsolv_dop.cxx:363` writes `locFCIDUMP.dat` through exactly this path.

### 2.6 Rotating the Hamiltonian silently disables a user setting — `include/macis/hamiltonian_generator/rdms.hpp:116` and `src/macis/impurity_solver.cpp:351`

`rotate_hamiltonian_ordm` (an upstream function) now ends with `SetJustSingles(false)`,
as do both new rotation variants. Then:

```cpp
ham_gen.rotate_hamiltonian_ordm_imp_bath(active_ordm.data(), n_imp, tmp_rot.data(), p.spin_dep);
asci_settings.just_singles = ham_gen.just_singles;   // now always false
```

A user who asked for `CI.JUST_SINGLES` gets singles-only for macro-iteration 1 and full
singles+doubles from iteration 2 onward, with no warning. If the rotation genuinely
invalidates a singles-only expansion, say so and fail loudly rather than rewriting the
user's setting.

### 2.7 New unconditional hard dependency on GSL — `src/macis/CMakeLists.txt:70-76`

```cmake
find_package(GSL REQUIRED)
target_link_libraries(macis PUBLIC GSL::gsl GSL::gslcblas)
```

Everyone building `macis` now needs GSL, even though only `src/macis/doping/` uses it.
Worse, `GSL::gslcblas` is linked publicly alongside whatever BLAS the project already
uses; duplicate CBLAS symbols across gslcblas and MKL/OpenBLAS are a well-known source
of silent mis-linking. Gate this behind an option (`MACIS_ENABLE_DOPING`) and drop
`gslcblas` unless a GSL routine actually needs it.

---

## 3. Missing error handling and unsafe resource use

### 3.1 C++ exceptions thrown through GSL C frames — `src/macis/doping/fix_mu.cpp:255-262,528,626`

`Mu_Cost_f` is installed as a GSL callback (`f.function = &Mu_Cost_f<N>` at `:576`) and
throws `std::runtime_error` on invalid `nbands` / invalid `ci_exp`. `SolveImpurity*` can
also throw (`"Davidson Did Not Converge!"`). GSL is C, compiled without unwind tables;
propagating a C++ exception through `gsl_root_fsolver_iterate`'s frame is undefined.
Even when it happens to work, it leaks the solver: `gsl_root_fsolver_free(s)` at `:637`
is never reached. Catch inside the callback and signal failure by returning `GSL_NAN`.

*Re-review:* still open. The new `GSLErrorHandlerGuard` addresses a *different*
problem (GSL's default handler aborting the process on a GSL-detected error) and does
it correctly, but `Mu_Cost_f` and the solvers it calls still `throw` from inside the
GSL callback, through GSL's C frames. The new `throw`s added in the `Fix_Mu_*` loops
themselves are fine — those run outside GSL frames.

### 3.2 GSL iterate status is discarded, and non-convergence is reported as success — `fix_mu.cpp:528-531,626-629`

```cpp
status = gsl_root_fdfsolver_iterate(s);
mu_prev = mu;
mu      = gsl_root_fdfsolver_root(s);
status  = gsl_root_test_delta(mu, mu_prev, abs_tol, 1.E-3);   // overwrites status
```

`gsl_root_fdfsolver_iterate` returns `GSL_EZERODIV` when the derivative is zero and
`GSL_EBADFUNC` when the function returns Inf/NaN — both immediately overwritten and lost.

At T = 0 the impurity filling is a **step function of μ**, so the forward-difference
derivative in `Mu_Cost_df` is *exactly zero* on any plateau. The sequence is then:
`iterate` returns `GSL_EZERODIV` (discarded) → μ does not move → `gsl_root_test_delta`
sees a zero step → returns `GSL_SUCCESS` → the code prints `"Converged!"` while sitting
on a plateau at the wrong filling. This is the most likely real-world failure mode of
the whole doping module and it is indistinguishable from success.

Neither `Fix_Mu_der` nor `Fix_Mu_noder` checks for `iter >= maxiter` after the loop —
both just return `gsl_root_*_root(s)` as if converged.

*Re-review:* verified fixed, and well. `iterate`'s status is checked before the
convergence test overwrites it; `GSL_EZERODIV` gets a diagnostic explaining the
plateau failure mode and pointing at `DOP.DSTEP`; exhausting `maxiter` now throws in
both variants instead of returning the last trial as converged; and `Fix_Mu_noder`
checks `gsl_root_fsolver_set`, turning an invalid initial bracket (`GSL_EINVAL`) into
an actionable error. Two small residuals: in `Fix_Mu_der`,
`gsl_root_fdfsolver_set` (`:488`) is *not* checked, and the `GSLErrorHandlerGuard` is
installed only after it — so a failure inside the derivative solver's setup still hits
GSL's aborting default handler. Mirror the noder version.

### 3.3 Reported observables do not correspond to the reported μ — `fix_mu.cpp:540-545,636-638` and `main/run_asci_impsolv_dop.cxx:331-422` - FIXED

`Fix_Mu_*` returns `res_mu` but never re-solves the impurity problem there. The driver
then does:

```cpp
mu_fixed = macis::Fix_Mu_der<nwfn_bits>(method_name, init_mu, &params);
std::cout << "Mu has been fixed to " << mu_fixed << std::endl;
...
for(const auto oc : params.occs) std::cout << oc << ", ";   // stale
E0 = params.E;                                              // stale
macis::CompObservables<nwfn_bits> obs(params);              // stale dets/C
```

`params.E`, `params.dets`, `params.C`, `params.occs` are whatever the *last* GSL trial
evaluation left behind — for a bracketing solver, typically a bracket endpoint, not the
root. Every occupation, energy, observable and Green's function printed under the
heading "Mu has been fixed to …" belongs to a different μ. Add a final
`Mu_Cost_f(res_mu, params)` before returning.

*Re-review:* verified fixed — both `Fix_Mu_der` and `Fix_Mu_noder` now re-solve at
`res_mu` before returning. Two residuals worth knowing about: (a) with
`cheap_mode` on, `mu_cost_counter > 1` at that point, so the final re-solve — the one
whose state all reported observables come from — is done by `SolveImpurityCheapASCI`,
not the full solver (see §3.4; a fix for §3.4 should also force the final call to the
full solver); (b) the re-solve runs before `gsl_root_*_free(s)`, so if it throws, the
solver leaks — free first, or use a guard like the error-handler one.

### 3.4 Finite-difference derivative compares two different solvers — `fix_mu.cpp:277-296` with `:236-238`

```cpp
if(cheap_mode && mu_cost_counter > 1)
  ci_exp = CIExpansion::ASCI_cheap;      // ci_exp is a reference into *p — permanent
```

`Mu_Cost_df` calls `Mu_Cost_f(mu + dstep)` then `Mu_Cost_f(mu)`. With `cheap_mode` on,
the counter crosses 1 between those two calls, so the pair is evaluated with *different
solvers* (`SolveImpurityASCI_rot` then `SolveImpurityCheapASCI`) and the difference is
dominated by the method change, not by `dstep`. The derivative is meaningless.

The mutation is also permanent and visible outside the module: `p->ci_exp` never returns
to the user's choice, so e.g. `tests/test_driver_dop.cxx:346`
(`if(params.ci_exp == CIExpansion::ASCI && asci_wfn_out_fname.size())`) silently stops
writing the wavefunction. Use a local variable for the per-iteration dispatch.

`dstep` is also never validated; `dstep == 0` gives a division by zero.

*Re-review:* still open, unchanged (`fix_mu.cpp:219-221`), and its blast radius has
grown: it now also silently downgrades the final re-solve added for §3.3, so in
cheap mode every "final" reported observable comes from the cheap solver.

### 3.5 No file-open error checking anywhere in the new I/O — `include/macis/util/general_io.hpp:36,85,106`, `src/macis/gf/gf.cxx:102-150`, `src/macis/impurity_solver.cpp:410,451,462`

```cpp
std::ofstream output_file(filename);
output_file.precision(...);
// ... writes ...   never checked
```

If the path is not writable, the disk is full, or the directory does not exist, every
write is discarded and the function returns normally. These are the only persistence
paths for the rotation matrices, 1-RDMs, correlators and Green's functions.

Related, in the same code:
- Filenames are hard-coded into the current working directory (`"GF.dat"`,
  `"rot_matrix.dat"`, `"active_ordm.dat"`, `"Sz_resolvent.dat"`,
  `"rot_matrix_<i>.dat"`). Under MPI **every rank** writes the same path concurrently —
  interleaved, corrupt output. None of the new code has a `world_rank` guard.
- `src/macis/impurity_solver.cpp:449` hard-codes `bool print_ordm = true;` — the write
  cannot be turned off.
- Inside `Fix_Mu_*`, `SolveImpurityASCI_rot` runs once per root-finder evaluation, so
  `rot_matrix_0.dat` is silently overwritten on every μ iteration.

### 3.6 `impurity_params` has ~30 uninitialised members — `include/macis/impurity_solver.hpp:40-86`

```cpp
template <size_t N>
struct impurity_params {
  size_t n_active;   size_t nbeta;    size_t nalpha;
  size_t n_imp;      size_t nbands;   size_t mu_cost_counter;
  double delta_CFS;  double dstep;    bool spin_dep;   // ... none initialised
};
```

Several are used unguarded on paths where a driver may not have set them:

- `CompObservables` ctor (`src/macis/comp_observables.cpp:147`) computes
  `n_sites_ = n_imp_ / n_bands_` — **division by zero** on an unset `nbands`.
- `fix_mu.cpp:140` reads `p->mu_cost_counter` before any driver necessarily sets it.
- `p.orb_rot` is read by `evaluate_GF` (`:179`) and `compute_impurity_rdm`, but only
  `SolveImpurityASCI_rot` populates it.

Give every member a default member initialiser, and validate the derived quantities
(`n_bands != 0`, `n_imp % n_bands == 0`, `n_imp <= n_active`, `n_active <= N/2`) in one
place.

### 3.7 Missing bounds validation in the bitset packing — `include/macis/sd_operations.hpp:73-88`, `include/macis/observables/impurity_rdm.hpp:35-61`

`hf_determinant_byocc` does `alpha.flip(idx[i])` and `beta.flip(idx[i] + N/2)` with no
check that `orb_occs.size() <= N/2` or that `nalpha <= orb_occs.size()`. If
`n_active > N/2`, an α flip silently lands in the β half of the bitset — a corrupt
determinant with no diagnostic.

`decompose_det` computes `n_active - n_imp` on `size_t`; `n_imp > n_active` underflows
to a huge loop bound and indexes the bitset out of range. `1ULL << p` is also UB once
`n_active - n_imp >= 64`, while the only guard anywhere
(`compute_impurity_rdm_from_state:336`) checks `n_imp > 32`.

---

## 4. Code that cannot compile, or is unreachable

### 4.1 `asci_grow_with_rot_legacy` will not compile if anyone calls it — `include/macis/asci/grow.hpp:180-254` - SOLVED

```cpp
ham_gen.form_rdms(wfn.begin(), wfn.end(), wfn.begin(), wfn.end(),
                  X_local.data(), ordm.data());        // :228 — 6 arguments
```

Both `form_rdms` overloads take 7 or 11 arguments, and `matrix_span_t` is a
`Kokkos::mdspan` with two dynamic extents — not constructible from a bare `double*`.
This compiles today only because it is a function template that is never instantiated
(`grow_with_rot_legacy` in the drivers routes to `SolveImpurityASCI_rot` instead).

The body has three further problems that make it worth deleting rather than fixing:
- `MPI_COMM_WORLD` is hard-coded in the `selected_ci_diag` call (`:242`) while
  `asci_iter` on the line above correctly receives `comm`.
- Unlike `asci_grow`, it never gathers the distributed `X_local` back into a full
  vector before the next `form_rdms`, so under MPI it would form RDMs from one rank's
  chunk treated as the whole wavefunction.
- `std::max(100ul, ...)` at `:219` hard-codes 100 where `asci_grow` uses
  `asci_settings.ntdets_min`, silently ignoring the user's setting.
- `if(its > 0 && its < nrots)` — `its > 0` is always true after the increment.

### 4.2 `write_matrix` vector overload calls a function that does not exist — `include/macis/util/general_io.hpp:70`

```cpp
template <typename T>
void write_matrix(const std::vector<T>& mat, ...) {
  print_matrix(mat.data(), rows, cols, filename, is_column_major, width);
}
```

There is no `print_matrix` in `macis::util` or any header. The only `print_matrix` in
the tree is a file-local helper in `main/test_lapack_convention.cxx`. The overload is
uninstantiable; it compiles because nothing calls it.

The same header uses `std::numeric_limits` at `:37`, `:86` and `:107` without including
`<limits>`, and its Doxygen blocks document a `name` parameter and an `output` stream
parameter — neither of which exists — while leaving the actual `filename` parameter
undocumented.

### 4.3 Two declared functions with no definition — `include/macis/doping/fix_mu.hpp:72-73,100-101`

```cpp
void print_state_fix_mu_noder(std::ostream&, size_t, const gsl_root_fdfsolver*, double mu0);
void print_state_fix_mu_noder(std::ostream&, size_t, const gsl_root_fsolver*,   double mu0);
```

Neither exists. The only definition (`fix_mu.cpp:325`) takes three arguments and is not
declared in the header. `print_header_fix_mu_noder` is declared twice, identically
(`:57` and `:84`).

The commented-out Doxygen blocks in this header describe an entirely different API
(`GhostGutzwiller &latt`, `Vs`, `lambda_cs`, `Chis`, `imp_1rdms`, `spsp_cfs`, `Eimps`)
copy-pasted from another project. That is worse than no documentation — a reader will
trust it.

### 4.4 Unqualified `abs()` on doubles — `src/macis/doping/fix_mu.cpp:404-405`, `src/macis/comp_observables.cpp:111`

```cpp
double delta_x = abs(x_hi - x_lo);      // fix_mu.cpp:404
if( abs(f_hi) < abs(f_lo) )             // fix_mu.cpp:405
```

Whether these resolve to `::abs(double)` or the integer `::abs(int)` from `<cstdlib>`
depends on which headers happen to be pulled in — it is not guaranteed. If the integer
overload wins, `abs(0.2)` is `0`, so `ProposeInitBracket_MuED` steps by zero: it
re-solves the impurity problem at the same μ ten times and then throws
"failed to find a valid bracket". The rest of the codebase uses `std::abs`
consistently, including `Fix_Mu_noder:583` twenty lines away.

*Re-review:* the `comp_observables.cpp` occurrence is gone (the string-sort that used
it was deleted, see §5.6). The `fix_mu.cpp` occurrences (`:383-384`,
`ProposeInitBracket_MuED`) are **fixed (2026-08-24)**: qualified to `std::abs`,
consistent with every other call in the file.

---

## 5. Efficiency and structure worth fixing

### 5.1 `build_reduced_density_matrix` is quadratic in the Fock-space dimension — `include/macis/observables/impurity_rdm.hpp:249-277`

Inside the loop over bath keys:

```cpp
std::vector<double> coeffs_vector(basis_size, 0.0);                 // 4^n_imp
std::vector<std::array<int,4>> occs_vector(basis_size, {0,0,0,0});
...
for(size_t i = 0; i < basis_size; ++i)
  for(size_t j = 0; j < basis_size; ++j) { ... }                    // 16^n_imp
```

`basis_size = 4^n_imp`. The inner double loop runs `16^n_imp` times per bath key even
though only `group_indices.size()` entries are non-zero — typically a handful. For
`n_imp = 5` that is ~10^6 iterations per bath key instead of ~10, and there can be
millions of bath keys. Iterating over `group_indices` directly makes this exact and
fast. The two `std::vector` allocations per iteration should also be hoisted.

Memory is the harder limit: `rho_local` is `16^n_imp` doubles **per thread**
(134 MB/thread at `n_imp = 6`), and `compute_overlap_matrix` builds a dense matrix of
the same size. The only guard is `n_imp > 32` (`:336`), which permits `basis_size = 2^64`.
The real limit is closer to `n_imp <= 7`.

### 5.2 Rotation-matrix convention is inverted relative to the rest of the codebase — `impurity_rdm.hpp:161,169`

```cpp
rot_mat[(size_t)m * (size_t)n_active_orbitals + (size_t)n]     // row-major
```

`orb_rot` is built and consumed as **column-major** everywhere else
(`impurity_solver.cpp:399` `orb_rot[i + i*n_active]`, all the `blas::gemm` calls with
`Layout::ColMajor`, `comp_observables.cpp` `orb_rot_[i + a*n_active_]`). Indexing it
row-major here selects the transposed sub-block, which flips `O[i][j]` to `O[j][i]` and
turns `rho' = O rho O^T` into `O^T rho O`. Given the repository history around this
exact question (`main/test_lapack_convention.cxx`, commits "Fixed ColMaj convention",
"Now seems fully consistent with ColMaj convention"), either fix it or add a comment
deriving why the transpose is intentional.

### 5.3 `compute_fermionic_sign` contains a term that cannot affect the result — `impurity_rdm.hpp:94-102`

```cpp
phase += 1LL * n_bath_up * (n_imp_up_bra + 2 * n_imp_down_ket + n_imp_up_ket);
```

The result is used only as `phase & 1`. `2 * n_imp_down_ket` is always even, so that
term is dead. Either the coefficient should be 1 (and the sign is currently wrong), or
the term does not belong. This function is the entire fermionic bookkeeping of the
impurity RDM and it needs a derivation comment plus a test that actually reaches it
(see §6.1).

### 5.4 Duplicated 4-index transform, three times — `include/macis/hamiltonian_generator/rdms.hpp:20-315`

`rotate_hamiltonian_ordm`, `rotate_hamiltonian_ordm_imp_bath` and
`rotate_hamiltonian_rotmat_imp_bath` each carry their own copy of the same ~70-line
quarter-transform. `rotate_hamiltonian_ordm_imp_bath` should build the rotation and
delegate to `rotate_hamiltonian_rotmat_imp_bath`. Three copies of a subtle contraction
means a fix to one will not reach the others.

Two smaller issues in the same file:
- `rotate_hamiltonian_rotmat_imp_bath:236` uses `assert(rot_mat != nullptr)`, compiled
  out under `NDEBUG`, while its sibling at `:126` uses `throw`. In a release build a
  null pointer becomes `std::copy(nullptr, nullptr + norb2_, ...)`.
- `const int nbaths = norb_ - nimps;` (`:127`) — `size_t` subtraction narrowed to `int`.
  `nimps == 0` is checked; `nimps > norb_` is not, and underflows.
- `std::cout << " Spin-dependent is set to TRUE..."` (`:157`) is a debug print in a
  library routine called once per ASCI macro-iteration, on every rank, bypassing the
  spdlog loggers used everywhere else.

### 5.5 Natural orbitals are derived twice, by two different LAPACK routines — `src/macis/impurity_solver.cpp:352,367-395` - FIXED

`rotate_hamiltonian_ordm_imp_bath` obtains the rotation from `lapack::gesvd` on the
imp/bath blocks. Immediately afterwards the caller re-derives the *same* blocks with
`lapack::syev` to get occupations for `hf_determinant_byocc`. For a symmetric PSD 1-RDM
the eigenvalues and singular values agree, but the **column ordering and signs need
not**, so `orb_occs` may not describe the basis the Hamiltonian was actually rotated
into — and `hf_determinant_byocc` picks the reference determinant from that list. Have
the rotation routine return the occupations it used.

*Re-review:* verified fixed exactly as suggested.
`rotate_hamiltonian_ordm_imp_bath` gained an optional `occs_out` parameter filled from
the *same* `gesvd` factorizations that build the rotation (imp block then bath block,
each descending — the same ordering the deleted `syev` code produced), the copy sits
before the spin-dependent branch so both spin cases get it, and the ~45-line duplicate
diagonalization in `SolveImpurityASCI_rot` is deleted. `orb_occs` is sized `n_active`
at the call site and the generator's `norb_` is the active dimension, so the bounds
match. The default argument keeps every other caller source-compatible.

### 5.6 Double occupancy computed by scanning formatted strings — `src/macis/comp_observables.cpp:97-122`

```cpp
std::vector<wf_pair> pairs;                    // one 32-char string per determinant
...
std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b){ return abs(a.coeff) > abs(b.coeff); });
for(int idet = 0; idet < pairs.size(); ++idet)
  for(size_t i = 0; i < n_imp; ++i)
    if(pairs[idet].str[i] == '2') orb_db_occs_bm += pairs[idet].coeff * pairs[idet].coeff;
```

The accumulation is order-independent, so the `std::sort` is pure dead work — and it
sorts with the unqualified `abs` from §4.4. Building `n_dets` strings of length `N/2`
costs ~32 MB for a 10^6-determinant wavefunction. `(alpha & beta & imp_mask).count()`
on the bitsets gives the same number directly.

*Re-review:* the string/sort machinery is fixed — the WF-based value is now computed
directly from `bitset_lo_word`/`bitset_hi_word` (correct halves: α low, β high), with
a good derivation comment tying it to the 2-RDM value. The three sub-items below are
still open.

Two more in the same function:
- When `asci_settings.nrots != 0` the whole body is skipped and the function returns
  `0.0` with **no warning** (`:73`, `:128`) — a rotated run silently reports zero double
  occupancy.
- `orb_db_occs` (the 2-RDM value) is computed, printed, and then discarded; the
  string-derived `orb_db_occs_bm` is what gets returned.
- The function rebuilds `active_hamiltonian` and a whole `SDBuildHamiltonianGenerator`
  (`:51-64`) — allocating `n_active^4` for `V_active` — even though `form_rdms` uses
  only the determinants and coefficients. It also omits the `ReadTdo` call that the
  solvers make, which makes the setup look meaningful when it is not.

### 5.7 `CompObservables` stores references into a mutable parameter struct — `include/macis/comp_observables.hpp:27-51`

```cpp
size_t& norb_;  size_t& n_imp_;  size_t& n_bands_;
std::vector<macis::wfn_t<N>>& dets_;  std::vector<double>& C_;  std::vector<double>& orb_rot_;
```

`dets_`, `C_` and `orb_rot_` alias the exact vectors that `SolveImpurity*` clears and
reassigns. Any solve while a `CompObservables` is alive silently changes what it
observes, and the object dangles if `p` goes out of scope first. The scalar references
buy nothing over copies. Take the values the class needs by value in the constructor.

### 5.8 Repeated `O(n_imp^6)` contractions — `src/macis/comp_observables.cpp:198-372`

`compute_double_occupancies`, `compute_db_occs_matrix`, `compute_sz_sz_correlations`,
`compute_tz_tz_correlations` and `compute_charge_charge_correlations` each redo the same
`orb_rot`-by-2-RDM contraction from scratch. Transform the 2-RDM into the site basis
once in the constructor.

Their normalisations also disagree: `compute_double_occupancies` divides by `n_imp_`
while `compute_db_occs_matrix[0]` computes the same sum and does not, so the two
"double occupancy" numbers the drivers print differ by a factor of `n_imp`. And
`sz_sz`/`tz_tz` return band-summed `n_sites × n_sites` matrices while
`charge_charge` returns a band-resolved `n_imp × n_imp` one — undocumented.

### 5.9 `sum_GFs` — `include/macis/gf/gf.hpp:698-726`

*Re-review:* mostly addressed as part of the §1.6 fix — element-count validation
against `(GF_orbs.size() - todelete.size())^2`, a frequency-count check, and a
rejection of the all-orbitals-dropped case now precede the loops, with an error
message naming the likely cause. The `int` narrowing, the redundant
`GF_orbs.size() > 1` branch and the `const` by-value return remain, but they are now
behind the validation and harmless in practice.

- No size validation on `GF1`, `GF2` or `ws`; mismatched inputs read out of bounds.
- `int GFmat_size = GF_orbs.size() - todelete.size();` — unsigned subtraction narrowed
  to `int`. If `todelete` is larger, the negative value converts back to `size_t` in
  `std::vector(GFmat_size * GFmat_size)` and requests an astronomically large allocation.
- If `GF_orbs.size() == 1` and `todelete` is non-empty, `GFmat_size == 0` and the
  `else` branch writes `GF[iii][0]` into a zero-length inner vector.
- The `if(GF_orbs.size() > 1)` branch is redundant — the general loop already handles
  the 1×1 case.
- `const std::vector<...>` return by value inhibits move on assignment.

### 5.10 `evaluate_GF` / `evaluate_resolvent_sz` — `include/macis/impurity_solver.hpp:116-128,232-243`

The twelve-line frequency-grid construction is duplicated verbatim between the two
functions and duplicates a third implementation, `GetGFFreqGrid` (`src/macis/gf/gf.cxx:13`)
— except that `GetGFFreqGrid` honours `settings.w_scale` ("lin"/"log") and
`settings.real_g` while these two silently ignore both. There is no check that
`gf_settings.beta > 0` (Matsubara branch divides by it) or that `nws > 1` (the real-axis
branch divides by `nws - 1`).

Both also allocate `active_trdm` of `n_active^4` doubles (`:96`, `:277`) purely to
satisfy the `form_rdms` signature and never read it — 104 MB at `n_active = 60`. The
`// TODO Make 1RDM-only work` in `asci_grow.hpp:80` is the same wish; a 1-RDM-only
overload would pay for itself here.

`size_t G_n_orbs = sqrt(GF[0].size())` (`:174`) and `size_t n_active = sqrt(orb_rot.size())`
(`:270`) recover a dimension by floating-point square root and truncation. Use
`std::llround`, or better, pass the dimension explicitly — it is available at both call
sites.

### 5.11 Global-scope pollution in public headers — `include/macis/impurity_solver.hpp:24-32`, `include/macis/comp_observables.hpp:5-11`

```cpp
using macis::NumActive;
using macis::NumOrbital;
// ... seven of these, at global scope, in a header
enum class CIExpansion { CAS, ASCI, ASCI_cheap };   // also global scope
```

Every translation unit that includes these headers inherits the using-declarations.
`CIExpansion` is a live conflict: `tests/standalone_driver.cxx:46` defines its own
`enum class CIExpansion { CAS, ASCI }` at global scope. The two only coexist today
because that driver does not include `impurity_solver.hpp`. Move both inside
`namespace macis`.

---

## 6. Testing and documentation gaps introduced by the change

### 6.1 `tests/impurity_rdm.cxx` does not exercise anything risky

The single test case uses one determinant, an identity rotation, `n_imp = 2`, and
**an empty bath**. Consequences:

- `compute_fermionic_sign` always receives `n_bath_up = n_bath_dn = 0`, so `phase == 0`
  and the sign is always `+1`. The entire fermionic bookkeeping (§5.3) is untested.
- `compute_overlap_matrix` is called with the identity, so the row-major/column-major
  question (§5.2) — the one thing this code most needs pinned down — cannot fail the test.
- No multi-determinant state, no non-trivial rotation, no spin-down occupation.

The `REQUIRE(nonzero == 1)` and `rho^2 == rho` checks would pass for a completely broken
sign convention. At minimum: a two-determinant state with an occupied bath, and a known
non-identity rotation with a hand-computed `rho`.

Also `std::bitset<dim>(i)` at `:46-47` uses the Hilbert dimension (16) as the bitset
*width*, where `2*n_imp` (4) was meant.

### 6.2 `tests/dynamical_properties.cxx` deadlocks under MPI — `tests/dynamical_properties.cxx:94,196`

`ROOT_ONLY(MPI_COMM_WORLD)` (`tests/ut_common.hpp.in:15-18`) returns on all ranks > 0.
The test then calls `RunResolventSz` → `make_dist_csr_hamiltonian(MPI_COMM_WORLD, ...)`,
which is **collective**. Rank 0 blocks inside it while every other rank has already
returned. Every other `ROOT_ONLY` test in the suite is purely local; these two are not.
Run the test on `MPI_COMM_SELF`, or drop `ROOT_ONLY` and make it genuinely collective.

That said, this file is the strongest testing work in the fork — an exact Lehmann-sum
comparison, a spectral-positivity check and a vanishing-response edge case. It is the
model the other new modules should follow.

### 6.3 Modules with no test coverage at all

`src/macis/doping/fix_mu.cpp` (654 lines), `src/macis/impurity_solver.cpp` (562),
`src/macis/comp_observables.cpp` (398) and `include/macis/util/general_io.hpp` are added
with zero tests. Several findings above are things a single test would have caught:

- §1.1 (empty `occs`) — any call to `Mu_vs_n` at all.
- §1.2 (`nalpha, nalpha`) — one `SolveImpurityASCI` run with `nalpha != nbeta`,
  asserting the electron count of the resulting determinants.
- §1.4 (factor of 2) — a 2-site Hubbard dimer at half filling, where the double
  occupancy is analytic.
- §4.2 (`print_matrix`) — one instantiation of the `std::vector` overload.

`tests/test_driver.cxx` and `tests/test_driver_dop.cxx` are added to
`tests/CMakeLists.txt:34-38` as executables but are **not** registered with
`add_test`, so `ctest` never runs them. They are drivers, not tests.

### 6.4 Documentation defects

- `include/macis/doping/fix_mu.hpp` — the commented-out Doxygen describes a different
  project's API (see §4.3).
- `include/macis/util/general_io.hpp` — documents parameters that do not exist and omits
  the one that does (see §4.2).
- `include/macis/gf/dynamical_properties.hpp:135-138` states a real precondition —
  "Sz_imp is invariant under the block-diagonal, spin-conserving natural-orbital
  rotations used here" — that is **false** for the full `rotate_hamiltonian_ordm`
  rotation reachable via `asci_settings.grow_with_rot`. Nothing enforces it.
- `GFSettings` defaults changed silently: `nws` 2001 → 1001 and `beta` `bool` → `double`
  (`include/macis/gf/gf.hpp:57,62`). The `beta` change is a genuine upstream bug fix and
  should be flagged to upstream; the `nws` change alters output for existing inputs and
  is not mentioned anywhere.
- `src/macis/impurity_solver.cpp`, `src/macis/comp_observables.cpp` and
  `src/macis/doping/fix_mu.cpp` are not clang-formatted (4-space indent, literal tabs at
  `comp_observables.cpp:225-250`, trailing whitespace) despite `.clang-format` at the
  repository root and a history of "Committing clang-format changes" commits. They also
  use `.cpp` where the rest of `src/macis/` uses `.cxx`, and four new files omit the LBNL
  copyright header that every other file carries.

---

## 7. Smaller items, grouped

- **`std::cout` throughout the library.** Upstream consistently uses spdlog with
  `world_rank ? null_logger_mt(...) : stdout_color_mt(...)`. Every new file prints
  directly to `std::cout`/`std::cerr` with no rank guard, so an N-rank job produces N
  copies of every line. `impurity_rdm.hpp:116` still carries
  `std::cout << "Basis size: " << basis_size << "\n";  // Remove later`.
- **Signed/unsigned loop counters.** `for(int i = 0; i < n_active4_; i++)`
  (`comp_observables.cpp:187`) iterates to `n_active^4`; `int norbs` in
  `Transform_2RDMs` computes `o1*norbs*norbs*norbs` in `int` arithmetic. Both are
  unreachable at realistic sizes but the codebase uses `size_t` for these extents
  everywhere else. Same pattern in `evaluate_GF`, `Mu_Cost_f` and the CFS loops.
- **`Mu_Cost_f`'s `delta_CFS` reset.** `fix_mu.cpp:150-153` prints an error and then
  sets `delta_CFS = 0.0` on the caller's struct, silently changing the physics, while
  the structurally identical branches at `:212-216` throw. Pick one.
- **Dead locals.** `double step = 0.1` in `ProposeInitBracket_MuED` (`:401`), unused
  `nel_target` in `Mu_vs_n`, unused `trdm_ud`/`trdm_du` parameters in `Transform_2RDMs`,
  the unused `N` template parameter on `ImpBathDecomp` (`impurity_rdm.hpp:26`), a dozen
  unused reference bindings at the top of each `SolveImpurity*`.
- **`ProposeInitBracket_MuED` leaves `f_lo`/`f_hi` stale.** On success the first branch
  sets `x_hi = x_curr` without updating `f_hi` (and symmetrically in the second), so the
  returned function values no longer match the returned bracket.
- **`SolveImpurityCheapASCI` has an undocumented, unchecked precondition.**
  `src/macis/impurity_solver.cpp:499,536` uses `p.dets` without populating it — it
  relies on a previous solve having filled it. `if(dets.empty()) throw` would make the
  contract explicit.
- **Explicit instantiation for `<64>` only** (`impurity_solver.cpp:557-560`,
  `comp_observables.cpp:395-396`, `fix_mu.cpp:645-651`, `gf.cxx:151-159`) means the
  templated API is a link error at any other width. The `RunGFCalc<64>` instantiation in
  `gf.cxx` is also redundant — the template is defined in `gf.hpp` — while forcing the
  `MPI_COMM_WORLD` code path into the core library object.
- **`main/CMakeLists.txt`** compiles `../tests/ini_input.cxx` from a sibling directory and
  is added unconditionally from the top-level `CMakeLists.txt:25`, so the drivers build
  even with `BUILD_TESTING=OFF` while depending on a file under `tests/`. No trailing
  newline.
- **`include/macis/types.hpp`** drops the public `namespace KokkosEx` alias. Nothing in
  this repository still uses it, so the removal is safe here, but it is a breaking change
  for any downstream consumer.
- **Pruning diagnostics commented out rather than downgraded** —
  `determinant_search.hpp:150,155,157`. `logger->debug(...)` keeps them available.

---

## Re-review of the fix round (unstaged changes on top of `b64251d`)

Every FIXED/PARTIALLY FIXED verdict annotated above was checked by reading the new
code against its callers and the surrounding library; compilation was confirmed
separately (Release build, `MACIS_build`). The fixes for §1.1–§1.7, §3.2, §3.3 and
§5.5 are genuine and carefully done — in particular the fcidump parser rewrite
(§1.5, with regression fixtures in both layouts), the `todelete` plumbing (§1.6,
now validated at three levels), and the `occs_out` change (§5.5, which removes the
degenerate-subspace inconsistency rather than papering over it). Points verified
specifically because they could have broken something:

- `Mu_vs_n` / `Mu_Cost_f` now need `Td`, `Fd_inactive`, `Td_active`, `F_inactive`,
  `T_active`, `V_active`, `n_inactive` — all three drivers
  (`run_asci_impsolv_mu_vs_n`, `run_asci_impsolv_dop`, `tests/test_driver_dop`)
  allocate them, and the new `active_hamiltonian` calls mirror the drivers' setup
  calls argument-for-argument.
- The solvers fill `p.occs` with `n_active ≥ n_imp` entries, so `Mu_vs_n`'s reads
  are in bounds after the §1.1 fix.
- `evaluate_GF`'s new unitarity guard cannot false-fire on unrotated runs: every
  driver initializes `orb_rot` to the identity. It also enforces, for the GF path,
  the block-diagonality precondition that §6.4 noted was stated but unenforced
  (`evaluate_resolvent_sz` still has no equivalent guard).
- `bitset_lo_word`/`bitset_hi_word` return `std::bitset<N/2>` with α in the low
  half — the new `Comp_db_occs` reference value indexes the correct spin halves.
- Both `RunGFCalc` overloads are explicitly instantiated for `<64>`, so the drivers
  and `standalone_driver` all still link.

### New findings from the re-review

**R.1 The doping module silently assumed `n_inactive == 0`** —
`set_impurity_diagonal` writes μ onto diagonal entries `[0, n_imp)` of the *full*
orbital set, but `active_hamiltonian`'s documented convention is that the **inactive
orbitals are the leading indices**. With `n_inactive > 0` the μ shift would land on
inactive orbitals, the impurity orbitals inside the active block would never be
shifted, and `E_inactive` (computed once at driver setup) would go stale. This is
pre-existing behaviour, not introduced by the fix round.

**Fixed (2026-08-24).** `Mu_vs_n` and `Mu_Cost_f` now `throw std::runtime_error` if
`p->n_inactive != 0`, before touching `T`/`Td`. Both doping drivers default
`n_inactive` to 0 and only change it via the optional `CI.NINACTIVE` keyword, so this
does not affect any currently-working configuration — it turns a silent physics error
into an explicit one for the case that would have hit it. Actually supporting
`n_inactive != 0` (indexing the impurity orbitals at their offset within the active
block rather than at `[0, n_imp)`) is out of scope for this fix and remains
unimplemented.

**R.2 `Fix_Mu_der` setup is outside the new error handling** — `gsl_root_fdfsolver_set`
(`fix_mu.cpp:488`) has its status discarded and runs before the
`GSLErrorHandlerGuard` is installed; `Fix_Mu_noder` handles the equivalent call
correctly. (Detailed under §3.2.)

**R.3 Cheap mode degrades the final reported state** — the §3.3 final re-solve
inherits `ci_exp = ASCI_cheap` from the §3.4 mutation, so the observables reported
"at the fixed μ" come from the cheap solver. (Detailed under §3.3/§3.4.)

**R.4 Per-iteration `V_active` recopy** — `Mu_Cost_f` and `Mu_vs_n` now rebuild the
full active Hamiltonian on every μ evaluation, which copies the `n_active^4`
two-electron block once (twice when `spin_dep`) even though only the one-body part
changes with μ. Correct, and small next to the CI solve that follows, but a
one-body-only variant of `active_hamiltonian` would remove it.

---

## Remaining open issues, by severity

**High — wrong numbers, a hang, or an abort with no diagnostic:**

| # | Issue |
|---|-------|
| 3.4 | With `cheap_mode`, `Mu_Cost_df` differences two *different solvers* (meaningless derivative), permanently rewrites `p->ci_exp`, and now also downgrades the final §3.3 re-solve; `dstep == 0` still divides by zero |

2.1, 2.2 and 2.4 (all above this line as of the last pass) were fixed on 2026-08-24 —
see their entries above for what changed.

**Medium — wrong numbers or crashes on paths that are one setting away:**

| # | Issue |
|---|-------|
| 3.1 | C++ exceptions still propagate through GSL C frames from `Mu_Cost_f` and the solvers it calls (UB; leaks the GSL solver when it "works") |
| 2.5 | `write_fcidump` zero-skip can shrink `norb` on round-trip (`locFCIDUMP.dat` path) |
| 2.3 | `p_davidson` aborts on ranks owning zero rows — a configuration upstream deliberately supported |
| 3.6 | `impurity_params` members uninitialised; `CompObservables` divides by an unset `nbands`, `evaluate_GF` reads `orb_rot` that only some drivers/solvers populate |
| 3.5 | No file-open error checking in any new I/O; hard-coded filenames written by every MPI rank concurrently |
| 2.6 | Hamiltonian rotation silently disables the user's `CI.JUST_SINGLES` setting |
| 5.2 | `impurity_rdm.hpp` indexes `rot_mat` row-major where the rest of the codebase is column-major — transposed rotation of the impurity RDM (flagged for double-check; still unverified either way, and §6.1's identity-rotation test cannot catch it) |
| 5.3 | `compute_fermionic_sign` carries a term that cannot affect the parity — either the coefficient is wrong or the term is dead (flagged for double-check; untested, see §6.1) |

**Low — latent, inefficient, or hygiene:**

| # | Issue |
|---|-------|
| R.2 | `gsl_root_fdfsolver_set` status unchecked; error-handler guard installed too late in `Fix_Mu_der` |
| R.3/3.3 | GSL solver leaks if the final re-solve throws; final re-solve uses cheap solver in cheap mode |
| 3.7 | No bounds validation in `hf_determinant_byocc` / `decompose_det` (underflow, shift UB) |
| 4.2 | `write_matrix` vector overload calls nonexistent `print_matrix`; missing `<limits>` |
| 4.3 | Declared-but-undefined `print_state_fix_mu_noder` overloads; copy-pasted Doxygen from another project |
| 2.7 | Unconditional public GSL + gslcblas link for the whole library |
| 5.1 | `build_reduced_density_matrix` quadratic in Fock-space dimension; per-thread `16^n_imp` memory with only an `n_imp > 32` guard |
| 5.4 | Triplicated 4-index transform; `assert` vs `throw` inconsistency; `std::cout` debug print |
| 5.6 | `Comp_db_occs` returns silent `0.0` when `nrots != 0`; computes and discards the 2-RDM value; rebuilds a Hamiltonian generator it barely uses |
| 5.7 | `CompObservables` stores references into the mutable parameter struct |
| 5.8 | Repeated `O(n_imp^6)` contractions; inconsistent normalisations between the two double-occupancy accessors |
| 5.10 | Duplicated frequency-grid code ignoring `w_scale`/`real_g`; no `beta > 0` / `nws > 1` checks; dimensions recovered via `sqrt`; `n_active^4` scratch allocated and never read |
| 5.11 | Global-scope `using` declarations and `CIExpansion` in public headers |
| R.4 | Full `V_active` recopy per μ evaluation |
| 6.1–6.4 | Test/doc gaps: `impurity_rdm` test exercises nothing risky, `dynamical_properties` deadlocks under MPI, doping/impurity/observables modules still untested, `test_driver*` not registered with `ctest` |

## Summary

The largest and most consequential items:

| # | Finding | File |
|---|---------|------|
| 1.1 | `Mu_vs_n` reads an empty vector and never applies μ | `doping/fix_mu.cpp:32,118-124` | FIXED
| 1.2 | ASCI HF guess uses `(nalpha, nalpha)` — wrong spin sector | `impurity_solver.cpp:169` | FIXED
| 1.3 | μ applied to spin-up channel only | `doping/fix_mu.cpp:96,228` | FIXED
| 1.5 | Integral-first FCIDUMP silently misparsed | `fcidump.cxx:44-90` | FIXED
| 2.1 | `no_constraint_search = true` breaks MPI ASCI selection | `asci/determinant_search.hpp:47` | FIXED
| 2.2 | Rank-conditional `allreduce` deadlocks `p_davidson` | `solvers/davidson.hpp:552` | FIXED
| 2.4 | Davidson reports unconverged results as converged | `solvers/davidson.hpp:268,304,524,563` | FIXED
| 3.2 | GSL status discarded; plateau reported as convergence | `doping/fix_mu.cpp:528,626` | FIXED
| 3.3 | Observables reported for a different μ than the one printed | `doping/fix_mu.cpp:540` | FIXED

All FIXED markers above were re-verified on 2026-08-24 (see "Re-review of the fix
round"); 2.1, 2.2 and 2.4 were fixed and verified in that same pass. None of these
are covered by a regression test except 1.5. The current severity-ranked list of
everything still open is the "Remaining open issues" section above; §3.4 is now the
top item.

Very important according to me
5.5 - FIXED (verified in re-review)


To ask for a double check 
5.3 - still open, untouched this round
5.2 - still open, untouched this round

Interesting improvements
5.8
5.4
5.1

5.7
5.11
