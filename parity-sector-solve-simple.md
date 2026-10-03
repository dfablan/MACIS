# Band-parity sector solve: minimal implementation proposal

> **STATUS: PHASE 1 IMPLEMENTED (2026-10-03), plus per-sector failure handling from §10.** Phase 2
> (NROTS > 0, §9) is not implemented. §11 lists what was built, where it differs from this
> proposal, and what is still unverified. Line numbers in §1–§9 refer to `4bb55e2`.
>
> This is a reduced first step of `symmetry-sector-solve.md`. It implements only the band-parity
> part of §3.3 and §3.5–3.7 of that plan, under the restrictions of §0 below. Everything else in
> that plan is listed in §10. §9 extends this to NROTS > 0.

---

## 0. Scope and assumptions

| Assumption | Consequence |
|---|---|
| **Band-diagonal bath:** each bath orbital couples to one band only. | Each band (its impurity orbitals plus its bath) is a separate orbital group. The band parity (−1)^{N_g} is an exact label of every determinant. |
| **Single site, or 1×2.** | No K labels. On the 1×2, the site swap is a permutation symmetry *inside* each band group, so it never changes a parity label. |
| **Kanamori with pair hopping.** | Pair hopping moves two electrons between bands, so the label is the parity, not the count. With J_P = 0 the counts are conserved too. That case is detected and warned about, not solved (§2.3). |
| **`ASCI.NROTS = 0`, `GROW_WITH_ROT = FALSE` (phase 1).** | The orbital basis is the FCIDUMP basis for the whole solve, so the labels are fixed once. There are no natural-orbital rotations to mix bands (plan F4) and no macro-iteration restarts (plan F2). Other settings are refused. §9 describes what NROTS > 0 needs. |
| **`n_inactive = 0`, `n_active = norb`.** | The doping path already requires `n_inactive = 0`, so this adds nothing new. |
| **Bands stored band-major** (confirmed as always the case): impurity orbitals `[b·nsites, (b+1)·nsites)` are band b. | This is the convention of `set_impurity_diagonal` (`fix_mu.cpp:24`) and `CompObservables`. No override key is needed. Bath orbitals get their band from T (§2.2). |

**The problem being solved.** H conserves every band parity. ASCI starts from one determinant,
and Davidson starts from the lowest diagonal element of the space (plan F1). So the solve can never
leave the parity sector of its seed. The legacy seed is closed-shell at Nα = Nβ, which puts it in
the all-even sector. **The fix: solve every parity sector separately, and keep the lowest.**

**Toy check** (`parity-sector-toy-ed.py`): the model of `tests/charge_sectors.cxx` with the
cross-band coupling set to 0, U = 6, J = 0.8, ε_d = −3. H between different sectors is exactly 0.

| (Nα, Nβ) | Sector | dim | E_min | Legacy seed |
|---|---|---|---|---|
| (3,3) | (e,e) | 200 | −8.512442 | ← lands here |
| (3,3) | (o,o) | 200 | **−8.551448** (GS) | |
| (3,2) | (e,o) | 150 | −8.700141 | |
| (3,2) | (o,e) | 150 | **−8.865079** (GS) | ← lands here |

So at (3,3) today's solver returns a state 0.039 Ha too high, at any determinant budget. At (3,2) it
happens to be right. This model becomes the unit test (§7).

**Number of sectors.** With B bands and N fixed, only the parity vectors whose sum matches N mod 2
can occur, so there are 2^(B−1) sectors: 2 for two bands and 4 for three. Cost in this version: one
full ASCI solve per sector, run serially.

---

## 1. Overview of the changes

| # | Piece | Where | Size (est.) |
|---|---|---|---|
| 1 | `ParityLabels`: detect, verify and clean the labels once at setup | new `include/macis/parity_sectors.hpp` + `src/macis/parity_sectors.cpp` | ~200 lines |
| 2 | Sector filter in the ASCI search (a hard guarantee) | `asci/determinant_search.hpp` (`ASCISettings`, around :574) | ~30 lines |
| 3 | Seed for each sector | `parity_sectors.hpp` | ~80 lines |
| 4 | Sector wrapper, dispatched inside `SolveImpurityASCI_rot`, `SolveImpurityASCI` and `SolveImpurityED` | `impurity_solver.cpp` | ~200 lines |
| 5 | Input keys, refusals and output | `run_asci_impsolv_dop.cxx`, `impurity_params` | ~40 lines |
| 6 | Tests | new `tests/parity_sectors.cxx` | ~200 lines |

The wrapper is entered from **inside** the solver functions, so no caller changes:

- the driver (`run_asci_impsolv_dop.cxx:472`);
- the µ search (`fix_mu.cpp:155, 257, 261`);
- the charge-sector search (`charge_sectors.cpp:284-285`).

Each of them gets the parity loop automatically.

---

## 2. Labels: `ParityLabels` (piece 1)

### 2.1 Data

```cpp
// include/macis/parity_sectors.hpp
namespace macis {

/// Band-parity labels of the active orbitals. Built once per FCIDUMP and shared.
struct ParityLabels {
  size_t ngroups = 0;
  std::vector<int> group_of;                     // active orbital -> group, -1 = decoupled
  std::vector<std::vector<uint32_t>> group_orbs; // group -> its active orbitals
  double max_discarded = 0.;                     // largest integral zeroed by the cleaning
  bool counts_conserved = false;                 // no term moves electrons between groups
};

/// Determinant -> parity key, bit g = (N_g mod 2), alpha and beta summed.
template <size_t N>
struct ParityMasks {
  std::vector<wfn_t<N>> mask;  // per group: alpha bits | beta bits << N/2
  explicit ParityMasks(const ParityLabels&);
  uint32_t key(const wfn_t<N>& d) const {
    uint32_t k = 0;
    for(size_t g = 0; g < mask.size(); ++g)
      k |= uint32_t((d & mask[g]).count() & 1u) << g;
    return k;
  }
};

/// What the ASCI search needs. ASCISettings is not templated on N, so it stores
/// orbital lists, and the search builds ParityMasks<N> from them on entry.
struct ParityTarget {
  std::shared_ptr<const ParityLabels> labels;
  uint32_t key;
};
}
```

### 2.2 Detection (`build_parity_labels(p, tol)`, called once in the driver)

1. **Seed the bands.** Impurity orbital `i < n_imp` gets band `i / (n_imp / nbands)` (band-major,
   always the layout of these inputs).
2. **Join orbitals by one-body coupling.** Run a union-find over the pairs with `|T_pq| > tol` (and
   `|Td_pq| > tol` if spin-dependent), plus the impurity orbitals of each band.
3. **Classify each connected component:**
   - exactly one band → that band's group;
   - two or more bands → **throw**. The message names the orbitals and the largest `|T_pq|` that
     joins the bands ("the bath is not band-diagonal");
   - no band at all (a "dark" bath orbital with zero hybridization) → `group_of = -1`, with a
     warning. Its occupation is conserved on its own and is fixed by the seed (§4). It is
     excluded from the sector enumeration.

### 2.3 Verification and cleaning (same function, on `p.T`, `p.Td`, `p.V`)

- **One-body.** Every `|T_pq|` with `group_of[p] != group_of[q]` must be ≤ `tol`.
- **Two-body.** For every `(pq|rs)`, sum the group memberships of all four indices; for each group
  g the sum `[p∈g] + [q∈g] + [r∈g] + [s∈g]` must be even. **This parity test does not depend on the
  index convention.** A term that breaks it must have `|V| ≤ tol`.
- **Cleaning.** Every violating element ≤ `tol` is set to exactly 0, and the largest one is kept as
  `max_discarded`. Anything above `tol` throws, quoting the element. Cleaning `p.T`, `p.Td` and `p.V`
  (not the `_active` copies) keeps everything downstream consistent: µ updates rebuild `T_active`
  from `p.T` (`fix_mu.cpp:127, 228`), and the GF path rebuilds the Hamiltonian from `p.T`/`p.V`
  (`run_asci_impsolv_dop.cxx:586`).
- **Count check.** If no term with `|V| > tol` moves electrons between groups (chemist convention
  `a†p a†r a_s a_q`), set `counts_conserved = true` and warn: *"no pair hopping, so the band counts are
  conserved, which is finer than parity. A count-sector trap is still possible, and this version
  does not handle it."*
- **Cost.** One O(norb⁴) sweep at setup, the same as `prepare_det_symmetry`'s check.

Greppable output: `PARITY_LABELS groups = [[0,2,4,..],[1,3,5,..]] decoupled = [] max_discarded = 0.0e+00 counts_conserved = F`

---

## 3. Sector filter in the ASCI search (piece 2)

`ASCISettings` gets one more member, passed by value like `sym_group`:

```cpp
// Parity-sector constraint (set only by the parity wrapper): asci_search drops
// every candidate whose band-parity key differs from target.
std::shared_ptr<const ParityTarget> parity_target;
```

In `asci_search`, right after `// Finalize scores` (`determinant_search.hpp:574`) and before the
seed determinants are re-inserted:

```cpp
if(asci_settings.parity_target) {
  const ParityMasks<N> pm(*asci_settings.parity_target->labels);
  const auto target = asci_settings.parity_target->key;
  const auto n0 = asci_pairs.size();
  asci_pairs.erase(std::remove_if(asci_pairs.begin(), asci_pairs.end(),
                     [&](const auto& x) { return pm.key(x.state) != target; }),
                   asci_pairs.end());
  logger->info("  * PARITY FILTER: dropped {} candidates", n0 - asci_pairs.size());
}
```

- **With cleaned integrals this drops nothing.** An out-of-sector determinant has zero coupling, so
  it is never generated. The filter is a cheap hard guarantee, and a nonzero count in the log
  points to a bug.
- **It covers grow and refine,** because both go through `asci_search`.
- **It is rank-local and runs before the top-k,** so under MPI it needs no communication.

---

## 4. Seed for each sector (piece 3)

```cpp
template <size_t N>
wfn_t<N> parity_seed(const wfn_t<N>& base, uint32_t target, const ParityMasks<N>& pm,
                     const ParityLabels& L, HamiltonianGenerator<N>& H, size_t norb);
```

- **Base.** `base = asci_reference_determinant(p)`, the legacy seed (`impurity_solver.cpp:577`).
  - **The seed is now energy-ordered for every run.** The `SYMMETRIZE_DETS` gate was removed
    (`ASCI.HF_BY_ENERGY`, default `TRUE`), so the "home" sector no longer depends on the order in
    which the bath fit emitted its poles. `tests/impurity_seed.cxx` shows the effect on the parity
    toy with the bath listed out of energy order.
  - Either way, the base is only a one-body criterion. At large U both orderings put the deepest
    impurity levels first, doubly occupied, at a cost of about U each. That is why the descent
    below uses the full diagonal energy ⟨D|H|D⟩.
- **Home sector** (`key(base) == target`): return `base` unchanged. The sector that today's solver
  reaches therefore runs **bit-for-bit as today**. This is the regression anchor (test T2).
- **Any other sector:**
  1. **Repair.** While `key(D) != target`: among the single moves (one spin, occupied i → empty a)
     whose groups g(i) ≠ g(a) are both in `key(D) ^ target`, apply the one with the lowest diagonal
     energy `H.matrix_element(D', D')`. For two bands this is a single move. For three bands it is
     also one move, since the two flipped groups are fixed. For four bands and more it can take two.
  2. **Descend.** Repeatedly apply the sector-preserving single or double move that lowers
     `⟨D|H|D⟩` the most, until none does (capped at, say, 50 steps). Moves are excluded on decoupled
     orbitals.
     - This matters at large U: one-body order is a poor guide there. The 1×2 U = 70 odd-sector
       seed `u00d2222` is a one-electron-per-band impurity, and the diagonal energy (which includes
       U, U′ and J) prefers it.
     - Cost: (n_occ·n_vir)² diagonal evaluations of O(n²) per step, well under a second at
       n_active ≈ 24–32.
- **Dark orbitals** keep their `base` occupation (no move touches them).

**Why the lowest diagonal energy:** Davidson starts from the lowest-diagonal determinant of the
space anyway (F1), so this is the start the solver would choose itself if it could see the sector.

**Multiple starts per sector** (plan §3.5) are left out. Its natural extension is
`ASCI.PARITY_STARTS = k`: keep the k best distinct descent minima, and add the descended base as a
second start in the home sector. They are worth adding only if T4/T5 (§7) show traps inside a
sector.

---

## 5. Sector wrapper (piece 4)

### 5.1 Dispatch

```cpp
template <size_t N>
double SolveImpurityASCI_rot(impurity_params<N>& p) {
  if(!p.parity_labels or p.asci_settings.parity_target)  // off, or already inside a sector
    return solve_asci_rot_one(p);                        // today's body, renamed
  return solve_parity_sectors(p, &solve_asci_rot_one<N>);
}
```

`SolveImpurityASCI` (used by `Mu_vs_n`) and `SolveImpurityED` (used by the CAS paths) get the same
three-line dispatch. `SolveImpurityCheapASCI` re-diagonalizes the previous call's determinant space,
which is already parity-pure, so in cheap mode the µ search stays in the winner's sector. This is
documented, not changed.

### 5.2 Changes inside `solve_asci_rot_one` (today's body)

- **Seed.** When `parity_target` is set, `hf_det = parity_seed(asci_reference_determinant(p), ...)`
  instead of the plain reference (`:577`).
- **Output files.** `active_ordm.dat` and `rot_matrix.dat` (`:736-755`) are written by the caller
  (the wrapper or the dispatch), not inside each sector solve. Otherwise every sector overwrites
  them and the last sector wins (plan F5). The solve returns its `active_ordm` alongside E.
- Nothing else changes: with `NROTS = 0` the macro loop runs once.

### 5.3 `solve_parity_sectors`

```text
keys    = all parity keys consistent with N (minus the dark-orbital electrons of base),
          or just ASCI.PARITY_ONLY if given
guesses = split_guess_by_key(p)          // §5.4
for k in keys:
    ps = p                                // a full copy: isolates T_active, V_active, dets, C, occs
    ps.asci_settings.parity_target = {labels, k}
    ps.asci_settings.sym_group     = stabilizer(sym_group, k)   // §5.5, if SYMMETRIZE_DETS
    ps.asci_wfn_fname / asci_E0    = guesses[k] or cold
    r[k] = {E, ndets, seed det, <seed|H|seed>, band occupations N_g, active_ordm}
winner  = argmin E
copy back to p: dets, C, occs, orb_rot, E, T_active, Td_active, V_active, asci_settings.just_singles
write active_ordm.dat and rot_matrix.dat for the winner; print table; write parity_sectors.dat
```

- **Copy cost.** Copying `impurity_params` copies `T`, `V` and `V_active`: 2 × norb⁴ × 8 B, about
  17 MB at norb = 32. That is negligible.
- **Outputs.**
  - stdout: one line per sector:
    `PARITY_SECTOR key=(o,o) E=… dE=… ndets=… seed=<det> E_seed=… N_band=[…] [WINNER]`
  - `parity_sectors.dat`: the same table. It is overwritten on every call, so after a µ search it
    holds the final µ.
- **Near-ties.** If another sector lies within `ASCI.PARITY_ETOL` of the winner, print
  `PARITY_TIE`. With degenerate bands at odd N, the (o,e) and (e,o) sectors are exact partners
  under band swap. The winner then breaks band symmetry, so G_AA ≠ G_BB.
  - Today's solver has the same issue: it lands in one of the two.
  - The DMFT side should average the bands (the existing symmetrization, plan §3.9).
  - We solve both partners anyway. This version doesn't exploit band swap to skip one, and
    agreement within tolerance is a free consistency check.

### 5.4 Guesses (`ASCI.WFN_FILE`, charge-sector warm starts)

These are allowed under `NROTS = 0`. Read the guess once and group its determinants by key:

- **All in one sector** (any guess written by a parity solve): that sector loads the file
  unchanged, with the supplied `E0_WFN`. The other sectors start cold.
- **Spread over several sectors:** for each sector, diagonalize its slice, write
  `<fname>.par<key>`, and pass it on with its own E0, exactly as `solve_from_seed` does
  (`charge_sectors.cpp:184`).
  - This is the normal case for the charge-sector N±1 seeds: c†_A ψ and c†_B ψ lie in different
    parity sectors. Without the split they would mix in one solve and trap (F1).

### 5.5 `SYMMETRIZE_DETS`

At odd N a band-swap permutation maps (o,e) to (e,o). Closing a sector's space under it would put
both sectors into one solve. Restrict the group to the **stabilizer of the target key**: keep g only
if it maps every group onto a group with the same target parity bit. These elements of the already
expanded group form a subgroup, so no re-expansion is needed. Site swaps (1×2) stay inside a band
and are always kept.

### 5.6 ED path

In parity mode `SolveImpurityED` filters the full Hilbert space by key, then runs
`selected_ci_diag` and `form_rdms`. This is about 30 lines, and it is what makes the T1 references
and the CAS µ search parity-correct.

---

## 6. Keys, refusals, setup (piece 5)

| Key (`[ASCI]`) | Default | Meaning |
|---|---|---|
| `PARITY_SOLVE` | `FALSE` | enables everything here |
| `PARITY_TOL` | `1e-10` | label-verification tolerance; violations at or below it are zeroed and reported |
| `PARITY_ETOL` | `1e-6` | near-tie report threshold (Ha) |
| `PARITY_ONLY` | none | solve one sector only, e.g. `[1,1]` (validation, reproducing a single run) |

- **Naming.** The keys use `PARITY_` so they can't be confused with the charge-sector `DOP.SECTOR_*`
  keys.
- **Refused with `PARITY_SOLVE`:**
  - `NROTS > 0` (until phase 2, §9);
  - `GROW_WITH_ROT`;
  - `n_inactive > 0` or `n_active < norb`;
  - `n_imp % nbands != 0`;
  - more than 32 groups.
- **Setup.** `impurity_params` gets `std::shared_ptr<const ParityLabels> parity_labels`. The driver
  builds it after reading the FCIDUMP and before `active_hamiltonian` (`run_asci_impsolv_dop.cxx:308`),
  because the cleaning acts on `p.T`, `p.Td` and `p.V`.
- **DMFT side.** The DMFT Python is outside this repo. It only has to pass `PARITY_SOLVE` through to
  `input.in` (and add the key to `Read_Vars`, plan §3.7).

---

## 7. Tests and validation

**Unit tests** (new file `tests/parity_sectors.cxx`; the model is `make_model` of
`tests/charge_sectors.cxx` with the cross-band coupling 0.15 → 0, ε_d = −3):

| # | Test | Pass |
|---|---|---|
| T0 | Labels | groups {0,2,4}/{1,3,5}. With cross coupling 0.15: throws. With 1e-12: cleaned, `max_discarded` = 1e-12. With J_P zeroed: `counts_conserved`. |
| T1 | Each sector, ASCI vs exact, NTDETS ≥ 200 | (3,3): (e,e) −8.512442367, (o,o) −8.551448456. (3,2): (e,o) −8.700141181, (o,e) −8.865079189. To 1e-8, cold and with a guess file. |
| T2 | Regression anchor | `PARITY_SOLVE` off vs on, (3,3): the (e,e) result equals the legacy energy bit for bit. The wrapper returns (o,o), 0.039 Ha lower. |
| T3 | Truncated budget (NTDETS ≈ 60) | every sector's `PARITY FILTER` count is 0; the winner is expected to stay (o,o) (the gap is 0.039 Ha), to be confirmed |
| T4 | Seed repair | from the (3,3) legacy seed, `parity_seed` reaches (o,o) in one move, and descent never raises `⟨D\|H\|D⟩` or leaves the sector; the seed is logged (whether it is one electron per impurity band is a check, not an assumption) |
| T5 | Charge-sector search + parity, warm | the N±1 seeds split into two slices each, and the final sector matches an exact scan |
| T6 | `SYMMETRIZE_DETS`, degenerate bands, odd N | the band swap is dropped from the stabilizer, both partner sectors agree to `PARITY_ETOL`, and `PARITY_TIE` is printed |

**Frozen production Hamiltonians** (plan V3/V9, NROTS = 0):

- **1×2 U = 70, `It_10`:** a cold `PARITY_SOLVE` returns −95.427914 in odd parity. The even sector
  returns −95.279685, the production value.
  - Those references come from `Parity_Test/`. If those runs used NROTS > 0, first recompute both
    references at NROTS = 0, with `PARITY_ONLY` and hand-built seeds.
- **Single-site two-band case** (plan V9c): both sectors are solved and the winner is the Hund
  triplet.
- **Overhead:** about 2× the wall time for two bands.

---

## 8. Implementation order

1. Pieces 1 and 2, with T0 and T3. Nothing changes for runs with `PARITY_SOLVE` off.
2. Piece 3 and the wrapper for `SolveImpurityASCI_rot` without guesses, with T1, T2 and T4.
3. Guess splitting (§5.4), the ED path (§5.6), `SolveImpurityASCI`, and the stabilizer (§5.5), with
   T5 and T6.
4. The frozen 1×2 U = 70 check, then a DMFT restart.
5. Phase 2: NROTS > 0 (§9), with T1/T2 repeated at NROTS = 2 and T7–T8.

---

## 9. Phase 2: NROTS > 0

NROTS > 0 breaks phase 1 in two places and touches two more. All four are local changes; nothing in
§2–§5 has to be redesigned, because the rotation can be made to **keep every orbital index in its
band group**, so the labels and masks never change.

### 9.1 Natural orbitals per band (the essential change)

**Problem.** `rotate_hamiltonian_ordm_imp_bath` (`rdms.hpp:120`) diagonalizes the whole impurity
block and the whole bath block (plan F4). For a parity-pure state the cross-band blocks of the 1-RDM
are exactly zero (⟨c†_A c_B⟩ flips two parities), so the eigenvectors *could* stay band-pure. But
when a band-A occupation equals a band-B occupation, `gesvd` may return any mixture of the
degenerate pair. With **degenerate bands this is the normal case**, not an accident. After a mixed
rotation, determinants no longer have a band parity, and the labels are gone.

**Fix.** Give the function an optional block map and diagonalize per (impurity|bath) × band block:

```cpp
void rotate_hamiltonian_ordm_imp_bath(const double* ordm, size_t nimps, double* rot_mat,
                                      bool spin_dep, double* occs_out,
                                      const std::vector<int>* group_of = nullptr);
```

- Blocks are index sets: {i < nimps, group g} and {i ≥ nimps, group g}. Each block's eigenvectors
  are written back **into the same index set**, with occupations descending inside it. So
  `group_of` is the same before and after the rotation, and so are the masks, `ParityTarget`, and
  the filter.
- With `group_of == nullptr` the blocks are exactly today's two (imp, bath). The legacy path stays
  bit-for-bit identical.
- Dark orbitals (`-1`) form their own block. They are 1×1 and never rotate.
- **Exactness.** With U block-diagonal by group, every element of the rotated T and V that breaks a
  parity is a sum of products that each contain a cleaned (exactly 0) integral, so it stays exactly
  0. The filter (§3) would catch any leak, and its drop count is the test.
- **Consistency with `orb_occs`.** The function already returns the occupations of the exact basis
  it rotated into (comment at `impurity_solver.cpp:665`). That property is kept per block.
- **Cost.** Negligible: several small `gesvd` calls instead of two.

**Plan §3.4's "covariant" choice** is not needed for parity (making band-B natural orbitals the
band-swap images of band-A ones, so band swap stays a signed permutation). It is needed only for
`SYMMETRIZE_DETS` with NROTS > 0 and for measuring point-group expectation values, so it stays
deferred (§10).

### 9.2 Sector-preserving restart (the second essential change)

**Problem.** Every macro iteration after the first restarts from
`hf_det = hf_determinant_byocc(nalpha, nbeta, orb_occs)` (`impurity_solver.cpp:688`). At Nα = Nβ
that is closed-shell, so it is always all-even (plan F2). In an odd sector the restart would lie
**outside** the target sector. With the filter on, the search from it would generate only
out-of-sector candidates, so the space would collapse to the seed alone. That is worse than the
legacy trap.

**Fix.** Pass the byocc determinant through the phase-1 seed function, using the *rotated*
generator, whose diagonal is the energy in the new basis:

```cpp
hf_det = macis::hf_determinant_byocc<N>(nalpha, nbeta, orb_occs);
if(asci_settings.parity_target)
  hf_det = parity_seed(hf_det, target, masks, labels, ham_gen, n_active);  // §4
```

- In the home sector byocc is already in sector and is returned unchanged.
- Add an assertion that every seed (cold, restart or guess slice) lies in its target sector, so the
  "collapsed space" failure can never be silent.

### 9.3 Warm starts with an inherited basis (charge-sector search)

Seeded charge sectors are solved at NROTS = 0 in the parent's basis `src.U`
(`charge_sectors.cpp:244-254`). If the parent ran with per-band rotations, `src.U` is
block-diagonal by group and the labels hold in that basis.

- Check it: in `rotate_active`, verify that the largest off-group element of U is ≤ `PARITY_TOL`,
  and throw otherwise. A basis from a legacy (non-parity) run is not block-diagonal.
- The guess split of §5.4 then applies unchanged.
- `ASCI.WFN_FILE` with NROTS > 0 stays refused, as today (plan F7).

### 9.4 What stays refused

- **`GROW_WITH_ROT`:** `rotate_hamiltonian_ordm` rotates the full active space. Allowing it would
  need the same per-group blocking inside `asci_grow`; that's possible but not needed now.
- **`SYMMETRIZE_DETS` with NROTS > 0:** already refused by `prepare_det_symmetry`. Allowing it is
  plan §3.4 (covariant natural orbitals).

### 9.5 Regression anchor and tests

- **The home sector is no longer bit-for-bit.** Per-band blocks order the natural orbitals inside
  each band's index range instead of across the whole impurity or bath range. That is the same
  space in a different order, so the ASCI determinant order changes. T2 at NROTS = 2 compares to a
  tolerance (1e-8 at a full budget), not bitwise.
- **T7:** degenerate bands, NROTS = 2, even N. The legacy `rotate_hamiltonian_ordm_imp_bath` mixes
  bands (the largest off-band element of `rot_matrix` is O(1)), while the per-band one does not
  (exactly 0). The filter drops 0 in every macro iteration.
- **T8:** odd sector, NROTS = 2. Every macro iteration's restart lies in the target sector
  (assertion), and the energy matches exact diagonalization at a full budget.
- **Side observation, worth checking on archived runs.** Today's NROTS > 0 runs with degenerate
  bands may already rotate across bands. The closed-shell restart then lies in a mixed basis, so the
  run can leave the parity sector by accident. That could explain parity-dependent behaviour that
  differs between NROTS = 0 and NROTS > 0. It is a hypothesis, cheap to check: the largest off-band
  element of an archived `rot_matrix.dat`.

**Size:** about 60 lines in `rdms.hpp`, 10 in `impurity_solver.cpp`, 15 in `charge_sectors.cpp`,
plus the tests.

---

## 10. Still pending from `symmetry-sector-solve.md` after phases 1 and 2

| Plan § | Item | Status here | Needed for |
|---|---|---|---|
| 3.1 | **Per-sector failure handling.** Refinement throws on non-convergence (F8). In the wrapper one failing sector would abort the whole call. | missing | **any production use. Recommended to pull into phase 1:** catch per sector, mark it `FAILED`, report `COVERAGE INCOMPLETE`, and never let a failed sector certify the winner (~30 lines). |
| 3.1 | Refine 2-cycle detection, `NONCONVERGED_CYCLE` status, union-of-spaces recovery | missing | robustness (V0) |
| 3.1 | Cold-seed closure bug (F3: the `SYMMETRIZE_DETS` closure discarded at `:617`) | missing | `SYMMETRIZE_DETS` runs. A one-line fix that can ride with phase 1. |
| 3.2 | Bath modes A and B: exact adaptation, and projection of SDP/off-diagonal fits, with discard metrics and guards | missing (non-band-diagonal baths are refused) | SDP baths, `TEST_D` |
| 3.3 | K labels, impurity momentum basis, 2×2 sector table, reduction to orbit representatives (band swap, C4) | missing | 2×2 (V4, V5, V8) |
| 3.4 | Covariant natural orbitals (band-B as images of band-A; (0,π) as images of (π,0)), `LABEL_LEAK`, `SYMMETRIZE_DETS` with NROTS > 0 | partly: per-band blocks only (§9.1) | point-group measurement, symmetrization with rotations |
| 3.5 | Several starts per sector; candidates by dynamic programming over levels (k lowest channel allocations); impurity enumeration at finite U | one seed per sector, by repair and descent (§4) | small-U channel traps (2×2 U = 0, V2) |
| 3.6 | ⟨S²⟩ of every result, ⟨R⟩/⟨σ_d⟩, `SPIN_CHECK` (Sz = 1 vs 0 in the same sector) | missing | spin-trap diagnosis, the 1×2 `--check-spin` violations (V6) |
| 3.7 | Parallel (sector × start) communicator split; per-sector subdirectories; per-sector wavefunctions for later warm starts; wrapping or refusing cheap mode | serial; cheap mode stays in the winner's sector | wall time at 3 bands or many starts |
| 3.8 | Screening at small budgets; µ search that re-checks sectors at the converged µ and reports a density jump when the winner switches between µ points | partial: sectors are re-solved at every µ evaluation, so each point is on the lowest branch, but a jump in n(µ) is not detected | doping runs near a crossing (V7) |
| 3.9 | Odd-N Green's function: compute G↑ and G↓ and average (F10). Orbit-aware G projection on the DMFT side (F11) | missing; ties are only reported (`PARITY_TIE`) | odd-N runs (`RUN_U2_Irrep`), degenerate bands at odd N |
| 3.10.1 | Count labels (J_P = 0, or density-density Hund with per-spin counts); sector windows; detecting the group from verified permutations | detected and warned only | J_P = 0 and density-density runs; the 3-band J = 0 collapse hypothesis |
| 4, step 7 | DMFT side: keys in `Read_Vars`; health check reads `parity_sectors.dat`; orbit-aware projection threshold | missing (outside this repo) | DMFT integration (V8) |
| 5 | Validation V0, V1, V2, V4–V8, V9b, V9d | only V3/V9c-type checks and the unit tests here | sign-off of the full plan |
| 6 | Decisions D1–D7 (SDP mode, Irrep σ_d, odd-N G, which N, queued runs, intentional symmetry breaking, odd-N survey) | open; D3 and D7 also matter for the single-site and 1×2 scope | — |

**Within the current scope** (single site or 1×2, band-diagonal bath), the items most likely to
matter next are:

1. per-sector failure handling;
2. odd-N G↑/G↓;
3. ⟨S²⟩, to confirm the Hund-triplet winners;
4. several starts per sector, if the U → 0 runs show traps inside a sector.

The rest belongs to the 2×2, SDP baths, or J_P = 0.

---

## 11. Implementation notes (phase 1)

### 11.1 What was built

| Piece | Where |
|---|---|
| `ParityLabels`, `ParityMasks<N>`, `ParityTarget`, `parity_key_string` | `include/macis/asci/parity_labels.hpp` |
| `build_parity_labels` (detection, verification, cleaning, count warning) | `src/macis/parity_sectors.cpp` |
| `enumerate_parity_keys`, `parity_stabilizer`, `parity_seed` | `include/macis/parity_sectors.hpp` |
| Sector filter, `ASCISettings::parity_target` | `include/macis/asci/determinant_search.hpp` |
| Sector wrapper, guess split, ED sector path, per-sector failure handling, `parity_sectors.dat`, `setup_parity_sectors` | `src/macis/impurity_solver.cpp` |
| `impurity_params::{parity_labels, parity_etol, parity_only}` | `include/macis/impurity_solver.hpp` |
| Keys `ASCI.PARITY_SOLVE`, `PARITY_TOL`, `PARITY_ETOL`, `PARITY_ONLY` | `run_asci_impsolv_dop`, `run_asci_impsolv_mu_vs_n`, `explore_charge_sectors` |
| Tests T0–T6 | `tests/parity_sectors.cxx` |

### 11.2 Differences from the proposal

- **Seed (§4).** On the toy, the (o,o) seed is `0d22u0`: one impurity orbital empty and one singly
  occupied, with ⟨D|H|D⟩ = −10.4 against +12.0 for the (e,e) home seed. It is a local minimum of the
  diagonal energy, but it is **not** one electron per impurity band. T4 therefore checks the
  local-minimum property only.
- **Descent doubles** are limited to pairs of moves that each touch an impurity orbital. That keeps
  the cost bounded at large `n_active`. Bath-only rearrangements are one-body and are reached by
  singles.
- **Failure handling (pulled forward from §10).**
  - A sector that throws (for example, refinement not converging) is reported as `FAILED`, and the
    others still run.
  - The output says `PARITY_COVERAGE INCOMPLETE` and names the failed sectors. The call throws only
    if every sector fails.
  - The seed assertion (`require_in_sector`) uses the same path, so a seed outside its sector is
    reported as a failed sector, never solved silently.
- **Guess slices** are written as `<fname>.par_<letters>`, e.g. `wfn.dat.par_oe`.
- **`active_ordm.dat` and `rot_matrix.dat`** are now written by `SolveImpurityASCI_rot` after the
  solve, once, for the returned state, on every call (as before).
- **Cheap mode** (`ASCI_cheap`) is not refused. It re-diagonalizes the winner's space, so it stays
  in the winner's sector, as §5.1 says.

### 11.3 Verification

- **Serial unit tests:** all 7 parity test cases pass, plus `ASCI impurity seed ordering`.
  - Each sector, ASCI and ED, matches `parity-sector-toy-ed.py` to 1e-8.
  - With `PARITY_SOLVE`, the (e,e) sector is bit-for-bit the legacy solve.
  - The filter drops 0 candidates, also at a 60-determinant budget, where (o,o) still wins.
  - Guess files work both in one sector and spanning both sectors.
  - `SYMMETRIZE_DETS` with degenerate bands: `PARITY_TIE` is reported at odd N.
  - The charge-sector search with parity on finds the exact ground sector, for both CAS and ASCI.
- **Full suite:** 41 of 42 test cases pass. The failing one, `ASCI Symmetric Search`
  (`determinant_symmetry.cxx:318`), fails identically without these changes.
- **Not verified:**
  - **MPI.** On 2 ranks, ASCI on these tiny models already fails without the parity code (the
    upstream `ASCI` test: "Davidson Did Not Converge!"), and MPI issues are being debugged on another
    branch. The MPI-specific code here is untested: the drop-count `allreduce`, `gather_ci_vector`,
    and the root-only guess-slice writes followed by a barrier.
  - **Frozen production Hamiltonians** (§7: 1×2 U = 70, the single-site two-band case).
  - **Phase 2** (§9).

