# Plan: Orbital-permutation symmetry enforcement in the ASCI determinant search

> **STATUS: NOT IMPLEMENTED** — written 2026-08-26. Implementation plan for closing the ASCI
> determinant space under the band-permutation group so the solver cannot artificially break the
> orbital symmetry. All line anchors verified against the current working trees on 2026-08-26
> (MACIS fork branch `feature/spin_dep` HEAD `f07c825`; PolClassy_DMFT branch `extra-local-features`).
>
> Companion documents: `~/Code/DMFT/PolClassy_DMFT/docs/debugging-single-site-dmft.md`,
> `~/Code/DMFT/PolClassy_DMFT/docs/plans/asci-stall-guards.md`,
> `~/.claude/plans/Possible_MACIS_bug_orb_rots.md`.

---

## 1. Context and motivation

ASCI keeps only the `ntdets_max` highest-scoring determinants. When the impurity Hamiltonian has an
exact flavor-permutation symmetry (N degenerate bands, Kanamori at J=0, each band coupled to an
identical bath), the truncated determinant space is not group-invariant: the top-K cut splits
score-degenerate orbits arbitrarily, the variational ground state in the lopsided space favors one
flavor, and the DMFT loop amplifies the bias into artificial orbital polarization (observed on
`.../3band/singlesite/Doping/Irrep/Nb15/J_0/U_31.00` and `U_30.00`; mechanism documented in
`Possible_MACIS_bug_orb_rots.md` §5 and §12.3 — at J = 0 the break is generated inside
macro-iteration 1, before any rotation, so neither the natural-orbital degeneracy guard nor
`nrots = 0` prevents it). The existing `symmetrize_solver_output` projection recovers the symmetry
label but not the physics: group-averaging a polarized ρ is an incoherent mixture of polarized
states, not the symmetric ground state (`debugging-single-site-dmft.md` §3.3).

**Fix:** close the selected determinant set under the group G at every ASCI selection step. A
G-invariant basis makes the projected Hamiltonian commute with the group representation, so
truncation cannot split the multiplet — the counterfactual stated in `Possible_MACIS_bug_orb_rots.md`
§5: "if 𝒮 were exactly S₃-invariant all δ_m would coincide and the degeneracy would survive."

Only the determinant *set* must be closed — no fermionic sign bookkeeping is needed. A permutation of
orbital labels maps a determinant bitstring to another bitstring; signs would matter only if
coefficients were transformed, and Davidson re-solves the coefficients in the closed space anyway.

**Caveat to keep in mind (from the U_30.00 post-mortem):** this feature fixes the symmetry-breaking
disease, not the refine-stall disease (`asci-stall-guards.md`). A symmetric space can stall
symmetrically — and with symmetry imposed, a catastrophic stall no longer announces itself through
`std(Occs)` or the discarded projection weight. `E0` continuity and the variational bound
(`debugging-single-site-dmft.md` §2.1) become the primary watchdogs and should be monitored on every
production run using this feature.

---

## 2. Scope

**Group implemented: G = S_nbands band permutations**, acting jointly on the impurity band index and
on that band's bath orbitals.

| Case | Status |
|---|---|
| Single-site multiband (`nsites == 1`) | **Fully supported and tested** (the production case, e.g. 3-band Nb15 J=0). |
| Multisite multiband (cluster DMFT) | **Band permutations supported conditionally.** The impurity layout is band-major — `orbital = band*nsites + site` (`src/macis/comp_observables.cpp:233`, `src/macis/doping/fix_mu.cpp:35-63`) — so a band permutation is still an orbital permutation: `band*nsites + site → σ(band)*nsites + site`, bath slots likewise. Gated by the assertions below. |
| Site permutations (cluster point group) | **Out of scope.** In the Irrep bath basis the point group acts by irrep rotations/signs (2D E irreps mix bath orbitals by 2×2 rotations, 1D irreps by signs), not by orbital permutation — not expressible as a bitstring relabeling. See §10. |

### Assert, never presuppose

The band-permutation construction relies on the bath coupling being **band-diagonal** (each bath
orbital couples to exactly one band's impurity block). This must be **asserted on the actual data,
never inferred from `bath_struct == 'Irrep'`** — a future or alternative bath structure need not be
band-diagonal, and the code must detect that rather than assume it away:

- **MACIS side (general gate):** the solver validates that each supplied permutation is an exact
  symmetry of `T_active` (and `Td_active` if `spin_dep`) and `V_active` within `SYM_TOL`, and refuses
  to run otherwise. This is the fully general sufficient condition — it presupposes nothing about the
  bath structure; any bath that is not band-diagonal (or not band-symmetric) fails this check.
- **Python side (construction gate):** before emitting generators, assert from `cAp.imp_vs` itself
  that `V[a, k] == 0` for every impurity orbital `a` outside the band block bath `k` belongs to, and
  that the bath parameters are exactly band-symmetric. On failure raise with a clear message instead
  of emitting a wrong permutation.

*Precision on what is fundamentally required (single-site included):* band-diagonality is **not**
necessary for the symmetry itself — what is necessary is that the band permutation σ **lifts to a
pure (unsigned) permutation π of the bath orbitals** with `ε_{π(k)} = ε_k` and
`V[σ(a), π(k)] = V[a, k]`. Band-diagonal + band-symmetric parameters is the simple sufficient case
where the lift is algebraic (P4's `lift()`), and it is what the Irrep bath produces. A
non-band-diagonal bath can still be permutation-symmetric (e.g. a bath orbital coupling equally to
all bands is a fixed point, π(k) = k), but then the lift must be found by matching coupling vectors
— out of v1, MACIS side unchanged. And some symmetric baths admit **no** pure-permutation lift at
all: symmetry-adapted combinations such as a `(v, −v)/√2` coupling map to minus themselves under
the band swap — a *signed* permutation, the same obstruction as the multisite E irreps, not
expressible as a bitstring relabeling. The P3 band-diagonality assertion therefore gates the
**generator construction**, while the MACIS invariance check gates the **physics** — both fire
independently, and neither presupposes the bath structure.

### Hard incompatibilities (validated, with clear errors)

`ASCI.NROTS > 0` and `ASCI.GROW_WITH_ROT = TRUE`: natural-orbital rotations (`grow.hpp:69-167` and
the macro loop in `SolveImpurityASCI_rot`, `src/macis/impurity_solver.cpp:299-394`) destroy the
flavor labeling of the orbital indices that the permutation acts on. `spin_dep == true` is allowed —
a permutation acts identically on both spin sectors; it merely requires both `T_active` and
`Td_active` to pass the symmetry check. (`nrots = 0` is already the production recommendation at
J = 0, `Possible_MACIS_bug_orb_rots.md` §12.7.)

---

## 3. Group representation and input format

### In memory (MACIS)

- A permutation is `std::vector<uint32_t>` of length `n_active`; `perm[p]` is the image of active
  orbital `p` (0-based active-space indices; production runs with `n_inactive = 0`).
- The input supplies **generators** (an n-cycle plus a transposition generate S_n; the transposition
  alone suffices for n=2). The solver expands them to the **full group** once at entry and stores it
  in `ASCISettings` behind a `shared_ptr` — the settings struct is passed **by value** through
  `asci_grow → asci_iter → asci_search`, so members must be cheap to copy.

### Input file format (`[ASCI]` section)

Parsed with the existing `getData<std::vector<int>>` specialization (`tests/ini_input.cxx:342-350`,
whitespace-separated ints on one line; precedent: `GF.ORBS_BASIS` at
`main/run_asci_impsolv_dop.cxx:545`):

```ini
[ASCI]
SYMMETRIZE_DETS = TRUE
SYM_NPERM = 2
# image of active orbital p at position p, 0-based, length = n_active
SYM_PERM_1 = 1 2 0  4 5 3  7 8 6  10 11 9  13 14 12  16 17 15
SYM_PERM_2 = 1 0 2  4 3 5  7 6 8  10 9 11  13 12 14  16 15 17
SYM_TOL = 1e-8
```

The example is the 3-band cycle and the (0 1) transposition for `nsites=1`, `n_imp=3`, `nbaths=15`:
impurity orbitals 0–2, bath interleaved cyclically over bands — matching the verified live-run
FCIDUMP layout where the bands own baths {4,7,10,13,16}, {5,8,11,14,17}, {6,9,12,15,18} (1-based).

---

## 4. Design overview

Selection pipeline in `asci_search` (`include/macis/asci/determinant_search.hpp:427-696`):

```
candidate generation (serial :479-482 / MPI-constraint :483-488)
→ serial dedup :530
→ score finalize (candidates rv = -|rv| :562; seeds re-injected with +|C| :565-568;
                  keep_only_largest_copy :572)
→ top-K (MPI dist_quickselect + Allgatherv :600-653  [MODIFY: also gather scores]
         / serial nth_element :654-660)
→ [NEW] G-closure + whole-orbit budget selection      ← primary insertion
→ new_dets extraction :681-687, return :695
```

The closure is inserted **after** the top-K, where the surviving list is bit-identical on every MPI
rank in both paths (serial: trivially; MPI: replicated by `MPI_Allgatherv` at :636-638), so a
**deterministic local closure needs no MPI communication**. Closure before the top-K is rejected:
orbit members of a determinant live under different mask constraints on different ranks
(`mask_constraints.hpp:585-595`), so an early closure would break the constraint-partition
invariants and require a distributed orbit-dedup.

**Verified correction found during design:** the MPI top-K path *strips scores* (:602-605) and
rebuilds `topk` with fabricated `rv = -1.0` (:646-651). Orbit-aware truncation needs real scores in
both paths, so Step 3 gathers the scores alongside the bitstrings (one extra `MPI_Allgatherv` of
doubles with identical counts/displacements). Behavior-neutral when the feature is off — the
fabricated scores are never used downstream today.

**Size-budget policy — strict budget, whole orbits:** group the surviving determinants into orbits,
score each orbit by the max |rv| over its members present in the top-K, sort orbits by
(score desc, canonical representative asc), and greedily emit whole orbits while the total stays
≤ `ndets_max`. The result is G-closed, deterministic, rank-identical, and never exceeds the budget.
Two knock-on effects, handled explicitly:

- The final size may sit slightly below `ndets_max` (deficit < one orbit ≤ |G|) forever.
  `asci_grow`'s loop `while(wfn.size() < ntdets_max)` (`grow.hpp:51`) would then **spin
  indefinitely** — the "didn't grow enough" branch at `grow.hpp:64-66` only logs, and the
  `prev_size` variable at `grow.hpp:47` is dead code. Step 5 adds a stall guard, which also fixes
  this latent infinite-loop bug in general.
- `asci_refine` throws if the size changes (`refine.hpp:52-53`). Step 6 relaxes the check, when
  symmetrization is on, to "must not *grow* beyond `ndets`" with shrinkage logged.

When the Hamiltonian and the seed set are exactly symmetric, orbit members have identical scores up
to Davidson tolerance (`ci_res_tol`, default 1e-8, `include/macis/util/mcscf.hpp:27`), so the top-K
boundary cuts at most a handful of score-degenerate orbits — closure churn is O(|G|), not O(ndets).
Step 7 closes the *seed* (cdets) set too, which is what makes the candidate scores symmetric in the
first place.

**Rejected alternative — reserve headroom** (shrink `top_k_elements` by |G|, close upward): the
closure-added partners lack principled individual scores, the final size is nondeterministic relative
to the budget, and it complicates the refine invariant in the opposite direction (overshoot).

---

## 5. MACIS implementation steps

### Step 1 — new header `include/macis/asci/determinant_symmetry.hpp`

Greenfield: a full-tree grep confirms **no orbital-permutation utility and no symmetry
infrastructure exist anywhere in MACIS**. Contents:

```cpp
#pragma once
#include <macis/asci/determinant_contributions.hpp>  // asci_contrib
#include <macis/bitset_operations.hpp>               // bitset_less
#include <macis/types.hpp>

namespace macis {

// Image of a determinant under an orbital permutation, acting on both
// spin sectors (alpha = low N/2 bits, beta = high N/2 bits).
// perm[p] = image of orbital p; orbitals p >= perm.size() are fixed.
template <size_t N>
wfn_t<N> permute_orbitals(wfn_t<N> w, const std::vector<uint32_t>& perm) {
  wfn_t<N> out(0);
  const size_t norb = perm.size();
  for(size_t p = 0; p < norb; ++p) {
    if(w[p])         out.set(perm[p]);
    if(w[p + N / 2]) out.set(perm[p] + N / 2);
  }
  for(size_t p = norb; p < N / 2; ++p) {   // pass through untouched orbitals
    if(w[p])         out.set(p);
    if(w[p + N / 2]) out.set(p + N / 2);
  }
  return out;
}

// True iff perm is a bijection on [0, norb).
bool is_valid_permutation(const std::vector<uint32_t>& perm, size_t norb);

// Expand a generator list to the full group by BFS over composition.
// Deterministic (elements kept lexicographically sorted).
// Throws if the group exceeds max_size (misconfigured generators).
std::vector<std::vector<uint32_t>> expand_permutation_group(
    const std::vector<std::vector<uint32_t>>& gens, size_t norb,
    size_t max_size = 40320 /* 8! */);

// Canonical orbit representative: min over group images by bitset_less.
template <size_t N>
wfn_t<N> orbit_representative(
    wfn_t<N> w, const std::vector<std::vector<uint32_t>>& group);

// Whole-orbit budget selection (Section 4). Input: scored, deduplicated,
// rank-identical top-K list. Output: G-closed det list, size <= budget.
template <size_t N>
std::vector<wfn_t<N>> symmetric_orbit_select(
    const asci_contrib_container<wfn_t<N>>& pairs,
    const std::vector<std::vector<uint32_t>>& group, size_t budget);

}  // namespace macis
```

`symmetric_orbit_select` algorithm (the non-obvious part; fully deterministic):

1. For each pair, compute `rep = orbit_representative(state)`.
2. Aggregate into `std::map<wfn_t, orbit_info>` keyed by rep via `bitset_less_comparator<N>`
   (`bitset_operations.hpp:313`); `orbit_info.score = max(|rv|)` over members seen.
3. Materialize each orbit **once** from its rep: apply every group element, sort members with
   `bitset_less`, unique. (Orbit size divides |G|.)
4. Sort orbits by (score desc, rep asc via `bitset_less`) — total tie-break.
5. Greedy: emit whole orbits while `emitted + orbit.size() <= budget`; on the first orbit that does
   not fit, **stop** (strict budget; later smaller orbits are not back-filled, so the result is a
   clean prefix of the orbit ranking).
6. Return the concatenated members sorted by `bitset_less` (the canonical sort+unique idiom already
   used at `determinant_search.hpp:196-201`).

Cost: O(npairs·|G|·norb) bit operations. Production case |G| = 3! = 6, npairs ≤ ntdets_max —
negligible. If ever needed, `permute_orbitals` can be accelerated with precomputed per-orbital
scatter masks (noted, not required).

### Step 2 — `ASCISettings` fields (`include/macis/asci/determinant_search.hpp:34-63`)

```cpp
// Orbital-permutation symmetry enforcement (band permutations).
bool symmetrize_dets = false;
double sym_tol = 1e-8;  // integral-invariance validation tolerance
// Full group (identity included), expanded from input generators at
// solver entry. shared_ptr because ASCISettings is passed by value.
std::shared_ptr<const std::vector<std::vector<uint32_t>>> sym_group;
```

(The existing `n_imp_orbs = -1` field at :46 is precedent for solver-specific fields — but note it is
never set by any driver; ours gets plumbed in Step 9.)

### Step 3 — gather scores in the MPI top-K path (`determinant_search.hpp:600-653`)

Next to the existing bitstring gather (:624-638), gather the scores with the *same*
counts/displacements:

```cpp
std::vector<double> keep_scores_local(n_geq_local);
std::transform(g_begin, l_begin, keep_scores_local.begin(),
               [](const auto& p) { return p.rv; });
std::vector<double> keep_scores_global(n_geq_global);
MPI_Allgatherv(keep_scores_local.data(), n_geq_local, MPI_DOUBLE,
               keep_scores_global.data(), local_sizes.data(), displ.data(),
               MPI_DOUBLE, comm);
```

and build `topk` from `(string, score)` instead of the fabricated `{s, -1.0}` at :646-651. Apply the
same `resize` in the `n_geq_global > top_k_elements` edge case (:641-644). `keep_strings_global`
ordering is rank-deterministic (Allgatherv concatenates in rank order), and identical partitioning
keeps the string↔score pairing exact.

### Step 4 — closure call in `asci_search` (after :661, before :681)

```cpp
if(asci_settings.symmetrize_dets && asci_settings.sym_group) {
  auto closed = symmetric_orbit_select(asci_pairs, *asci_settings.sym_group,
                                       ndets_max);
  logger->info("  * SYM CLOSURE: {} dets -> {} dets", asci_pairs.size(),
               closed.size());
  // new_dets comes from `closed`; skip the transform at :681-683.
}
```

Restructure :681-687 so `new_dets` comes either from `asci_pairs` (feature off — unchanged) or from
`symmetric_orbit_select` (feature on). The small-list branch skips the top-K entirely (:598); the
closure sits after that `if` block, so it still runs. Add the
`#include <macis/asci/determinant_symmetry.hpp>`.

### Step 5 — stall guard in `asci_grow` (`include/macis/asci/grow.hpp:47-70`)

Use the currently-dead `prev_size` (:47) at the bottom of the while body:

```cpp
if(wfn.size() == prev_size) {
  logger->warn("ASCI grow stalled at {} determinants (target {}); "
               "terminating grow loop.", wfn.size(), asci_settings.ntdets_max);
  break;
}
prev_size = wfn.size();
```

Required for the strict-budget closure (final size can sit a few dets below `ntdets_max` forever);
independently fixes a latent infinite loop.

### Step 6 — relax the refine size invariant (`include/macis/asci/refine.hpp:52-53`)

```cpp
if(wfn.size() != ndets) {
  if(asci_settings.symmetrize_dets && wfn.size() <= ndets)
    logger->info("Refine wavefunction size {} <= {} (whole-orbit budget)",
                 wfn.size(), ndets);
  else
    throw std::runtime_error("Wavefunction size can't change in refinement");
}
```

`ndets` stays frozen at the entry size (:43); sizes fluctuate only within one orbit of it.

### Step 7 — close the seed (cdets) set in `asci_iter` (`include/macis/asci/iteration.hpp:22-28`) — phase 1b

After `reorder_ci_on_coeff` and `nkeep` (:25), when the feature is on: compute the orbit closure of
`{wfn[0..nkeep)}`. Since the incoming `wfn` is G-closed (it came from a closed `asci_search` or a
closed guess), the closure is a subset of `wfn`; stable-partition `wfn`/`X` jointly so closure
members precede the rest, and extend `nkeep` to the closure size. Orbit partners have |C| equal to
Davidson tolerance, so this only reorders within near-degenerate blocks. This makes the candidate
score function symmetric to ~`ci_res_tol`, minimizing orbit churn at the top-K boundary.

*Correctness note:* Step 4 alone already guarantees a G-invariant space; Step 7 improves selection
quality/stability and can be deferred without weakening the symmetry guarantee.

### Step 8 — validation and setup at solver entry (`src/macis/impurity_solver.cpp`)

New `prepare_det_symmetry(impurity_params<N>& p)` (declaration alongside the Step-1 header):

```cpp
template <size_t N>
void prepare_det_symmetry(impurity_params<N>& p) {
  auto& s = p.asci_settings;
  if(!s.symmetrize_dets) return;
  if(s.nrots > 0 || s.grow_with_rot)
    throw std::runtime_error("SYMMETRIZE_DETS requires NROTS = 0 and "
        "GROW_WITH_ROT = FALSE: natural-orbital rotations destroy the "
        "flavor labels the permutation group acts on.");
  // generators arrive in s.sym_group (from the driver); per generator g:
  //  - is_valid_permutation(g, p.n_active)                     else throw
  //  - max |T_active[g(p) + g(q)*n] - T_active[p + q*n]| <= s.sym_tol
  //    (and Td_active if p.spin_dep)                           else throw,
  //    printing the max deviation and the offending generator
  //  - max |V_active[g(p),g(q),g(r),g(s)] - V_active[p,q,r,s]| <= s.sym_tol
  //    (full n_active^4 sweep; n_active <= 32 -> <= ~1.05e6 doubles/gen)
  //  - permute_orbitals(canonical_hf_determinant<N>(nalpha, nbeta), g)
  //      == the HF det                                         else throw,
  //    message: the filling must close whole flavor multiplets
  //    (nalpha - n_imp must fill complete band groups in the bath prefix)
  // then expand and store the full group:
  //   s.sym_group = make_shared(expand_permutation_group(gens, p.n_active));
  // log group order and generator count.
}
```

The T/V invariance sweep **is the general "assert, never presuppose" gate of §2**: it verifies the
band-diagonality and band-symmetry of the actual integrals, whatever bath structure produced them.

Call sites:
- `SolveImpurityASCI` (`impurity_solver.cpp:81`): after the `ham_gen` setup / `SetNimp` (:120-133),
  before the guess block (:135). In the `asci_wfn_fname` guess branch (:135-163): **close the guess
  set under the group**, padding `C_local` with `0.0` for added determinants (zero-coefficient dets
  do not change the recomputed `E0` at :142-156; Davidson re-solves anyway); log the count added.
- `SolveImpurityASCI_rot` (:215): after the `ham_gen` setup (:266-279), before the macro loop
  (:299). The `nrots > 0` reseed path (`hf_determinant_byocc`, :374) is unreachable because
  validation already rejected `nrots > 0`.

Living in the library (not the driver) covers every entry path, including the GSL μ-root-find
wrappers `Fix_Mu_der` / `Fix_Mu_noder` (driver :329/:333) that call the solver repeatedly. Explicit
instantiations: `<64>` only (existing, `impurity_solver.cpp:524-527`; production enforces
`n_active <= 32` at `main/run_asci_impsolv_dop.cxx:155-156`).

### Step 9 — driver keyword plumbing (`main/run_asci_impsolv_dop.cxx:176-203`)

After `CONSTRAINT_LVL` (:195-196):

```cpp
OPT_KEYWORD("ASCI.SYMMETRIZE_DETS", params.asci_settings.symmetrize_dets, bool);
OPT_KEYWORD("ASCI.SYM_TOL", params.asci_settings.sym_tol, double);
if(params.asci_settings.symmetrize_dets) {
  size_t nperm = 0;
  OPT_KEYWORD("ASCI.SYM_NPERM", nperm, size_t);
  if(nperm == 0)
    throw std::runtime_error("SYMMETRIZE_DETS=TRUE requires SYM_NPERM >= 1");
  auto gens = std::make_shared<std::vector<std::vector<uint32_t>>>();
  for(size_t i = 1; i <= nperm; ++i) {
    std::string key = "ASCI.SYM_PERM_" + std::to_string(i);
    if(!input.containsData(key))
      throw std::runtime_error("Missing " + key);
    auto v = input.getData<std::vector<int>>(key);
    gens->emplace_back(v.begin(), v.end());
  }
  params.asci_settings.sym_group = gens;  // generators; expanded in Step 8
}
```

(`OPT_KEYWORD` macro at :86-89.) **Policy for the duplicated keyword blocks:** mirror the same lines
into `main/run_asci_impsolv_mu_vs_n.cxx` (:155-171 — production doping study, keep in lock-step).
Do **not** touch `tests/test_driver.cxx`, `tests/test_driver_dop.cxx`, `tests/standalone_driver.cxx`
in v1 — the divergence is pre-existing; state this in the commit message.

### Step 10 — optional 1-RDM symmetry post-check (low priority)

After `form_rdms` in both solvers (`impurity_solver.cpp:200-202` and :398-401): when the feature is
on, compute `max_{g,p,q} |D[g(p),g(q)] - D[p,q]|` and log it (warn if > 1e-6, never throw —
Davidson-tolerance asymmetry is expected). Cheap (n² per group element); gives an immediate health
metric in production logs.

---

## 6. Size-budget and refine-invariant summary

| Phase | Budget behavior with closure | Guard |
|---|---|---|
| grow (`asci_grow`) | size ≤ `ndets_new`, possibly a few short of `ntdets_max` at the end | stall guard (Step 5) breaks the loop; deficit < \|G\|, logged |
| refine (`asci_refine`) | size ≤ frozen `ndets`, never above | relaxed invariant (Step 6): shrink allowed + logged, growth throws |
| MPI vs serial | identical output: closure input is the replicated top-K list; closure is deterministic (total tie-break by `bitset_less`) | no new communication |

---

## 7. PolClassy_DMFT implementation steps

The written FCIDUMP must be an *exact* invariant of the algebraic generator, or the solver-side
validation (Step 8) correctly refuses to run. Verified obstruction on the live U_31.00 run: the
per-band bath pole ordering is **scrambled** (fitted independently per irrep slot; measured band 0
order [4,13,10,7,16] by ascending eps vs band 1 [5,17,11,8,14]), parameters match across bands only
to ~1e-5, and one weak-coupling V element flips sign between bands.

### P1 — new flag and guards (`constANDparams.py`)

- Default next to `symmetrize_solver_output` (:217): `self.asci_symmetrize_dets = False`.
- Read next to the other symmetry keys (~:544): `read_key('asci_symmetrize_dets')` (`read_key`
  silently ignores absent keys → old inputs keep working).
- Guards in `Read_Vars` when true: `solv_version == 'MACIS'`; `nrots == 0` (fail fast in python);
  `nbands >= 2`; `band_sym == True` (see P2); numerical-gradient fit path — `Fits.py:127-129` raises
  `NotImplementedError` for `Irrep_gradient` under `band_sym` (verified). v1 targets
  `bath_struct == 'Irrep'`, but the operative gate is the **data assertion in P3**, not the struct
  label.

### P2 — enable `band_sym` for the Irrep bath (`constANDparams.py:486`)

`read_key('band_sym')` is currently reached only in the `Replica` branch (:475-486), yet the Irrep
fold/unfold already honor it (fold: `Fits.py:345-364` keeps only band 0's parameters; unfold:
`Fits.py:481-509` tiles them over all bands). Move/duplicate the read into the Irrep branch
(:472-474). With `band_sym = .true.` the fitted `(eps, vs)` are **exactly band-symmetric and
pole-aligned across bands by construction** (the tiling preserves slot order through
`expand_ind_vs`, `Fits.py:102-114`), sign gauge included — the written FCIDUMP becomes exactly
symmetric under the algebraic generator with **no numerical canonicalization on the steady path**.

### P3 — exactness + band-diagonality assertion before writing (`Solver.py`)

In `Solve` (:865-883), before `WriteFCIDUMP` (:876), when the flag is on, assert **on the data**:

1. **band-diagonality**: `cAp.imp_vs[a, k] == 0` for every impurity orbital `a` outside the band
   block bath `k` belongs to — checked on the array itself, never inferred from `bath_struct`;
2. **exact band symmetry** of `cAp.imp_eps` / `cAp.imp_vs` under the algebraic band generator
   (tolerance ~1e-13, float noise only).

Raise with a diagnostic dump on failure. This catches the transition iteration when restarting an
old non-`band_sym` run: the first fit after enabling `band_sym` folds band 0's parameters and
discards the rest — before that fit has happened, refuse to emit `SYMMETRIZE_DETS=TRUE` rather than
silently symmetrize numbers behind the fit's back. Optionally provide a one-shot standalone helper
script for legacy restarts (per band, stable-sort pole sets by `(eps, |v|)`; fix the V sign gauge —
legal because bath energies are diagonal; average across bands). It lives outside the DMFT loop.

### P4 — emit the generators (`Solver.py WriteASCIinput`, `[ASCI]` section :392-429)

Following the optional-key pattern at :422-426, after `NROTS` (:413):

```python
if getattr(cAp, 'asci_symmetrize_dets', False):
    gens = band_permutation_generators(cAp)
    ofile.write("#ENFORCE BAND-PERMUTATION SYMMETRY ON THE DET SPACE\n")
    ofile.write("SYMMETRIZE_DETS = TRUE\n")
    ofile.write("SYM_NPERM = {0:d}\n".format(len(gens)))
    for i, g in enumerate(gens, 1):
        ofile.write("SYM_PERM_{0:d} = {1}\n".format(
            i, " ".join(str(x) for x in g)))
    ofile.write("SYM_TOL = 1e-8\n")
```

New helper `band_permutation_generators(cAp)` (near `WriteASCIinput`): for each band-level generator
σ (the nbands-cycle plus the (0 1) transposition; just the transposition at nbands = 2), lift to the
full 0-based orbital image of length `n_imp_orbs + nbaths`:

```python
def lift(sigma, cAp):
    ns, nb, nimp = cAp.nsites, cAp.nbands, cAp.n_imp_orbs
    perm = list(range(nimp + cAp.nbaths))
    for band in range(nb):
        for site in range(ns):
            perm[band*ns + site] = sigma[band]*ns + site          # impurity
    for n in range(cAp.nbaths):                                    # bath
        k, r = divmod(n, ns*nb)
        b, s = divmod(r, ns)
        perm[nimp + n] = nimp + k*ns*nb + sigma[b]*ns + s
    return perm
```

The bath indexing `band(n) = (n // nsites) % nbands` matches `Fits.py:102 expand_ind_vs` exactly and
was verified against the live U_31.00 FCIDUMP.

**Production recommendation:** keep `symmetrize_solver_output = .true.` as an independent safety
net. With the solver-side fix its discarded weight should collapse to ~Davidson noise — itself a
useful A/B observable.

---

## 8. Failure-mode guards (consolidated)

| Condition | Where | Action |
|---|---|---|
| flag + `NROTS>0` or `GROW_WITH_ROT` | `prepare_det_symmetry` (+ python guard P1) | throw |
| generator not a bijection / wrong length | `prepare_det_symmetry` | throw |
| group expansion exceeds cap (40320) | `expand_permutation_group` | throw |
| `T/Td/V` not invariant within `SYM_TOL` | `prepare_det_symmetry` | throw, print max deviation + offending generator |
| HF reference not invariant | `prepare_det_symmetry` | throw; message explains the filling constraint |
| bath coupling not band-diagonal **in the data** | Solver.py P3 | RuntimeError before writing input |
| bath params not exactly band-symmetric | Solver.py P3 | RuntimeError before writing input |
| `WFN_FILE` guess not closed | guess branch, Step 8 | auto-close, pad C = 0.0, log |
| grow stalls below `ntdets_max` | grow.hpp stall guard | break + warn |
| refine size shrinks / grows | refine.hpp relaxed check | log / throw |
| `SYM_NPERM` missing or 0 with flag on | driver parse (Step 9) | throw |

---

## 9. Verification and test plan

### 9.1 Unit tests — new `tests/determinant_symmetry.cxx` (register in `tests/CMakeLists.txt:9-27`)

- `permute_orbitals`: identity; 3-cycle on hand-built alpha/beta bitstrings (both halves move
  together); orbitals ≥ `perm.size()` fixed; involution ∘ involution = identity.
- `is_valid_permutation`: rejects repeats, out-of-range, wrong length.
- `expand_permutation_group`: {3-cycle, transposition} → 6 elements; determinism; cap throws.
- `symmetric_orbit_select`: hand-scored list where the budget cuts an orbit → whole-orbit prefix
  returned, closed, ≤ budget; deterministic tie-break; seeds (positive rv) and candidates (negative
  rv) mix correctly via |rv|.

### 9.2 Integration test

Template: `tests/asci.cxx` `TEST_CASE("ASCI")` (:209-276). The water FCIDUMP has **no** permutation
symmetry — build a synthetic band-symmetric impurity model in-test (e.g. nbands = 2, 2 bath
orbitals/band, n_active = 6, Kanamori J = 0, `SDBuildHamiltonianGenerator<64>`), or reuse
`macis::hubbard_1d` with `pbc = true` and a cyclic-translation generator
(`include/macis/model/hubbard.hpp:18`). Run `asci_grow` + `asci_refine` with a deliberately awkward
odd `ntdets_max` (so unsymmetrized truncation must split orbits). Assert:

- every det's image under every group element is in the final set (closure);
- 1-RDM band blocks degenerate to < 1e-8;
- energy within a bounded difference of the unsymmetrized run at the same budget (symmetrized may be
  equal or very slightly higher; the exact occupation degeneracy is the point);
- `dets.size() <= ntdets_max`.

### 9.3 MPI

The existing harness runs `macis_test` both serial and `mpiexec -n 2` (`tests/CMakeLists.txt:52-57`)
— the 2-rank run exercises the Step 3 score gather automatically. Assert closure + identical energy
(1e-10) + identical sizes in both.

### 9.4 Production A/B (the decisive check)

On `/leonardo_work/Sis26_collura/dfloreza/g100_move/Leonardo_move/3band/singlesite/Doping/Irrep/Nb15/J_0/U_31.00`
(nbands = 3, nsites = 1, Nb = 15, n_active = 18, nα = nβ = 9 — the HF prefix closes whole band
multiplets: bath prefix 6 = 2×3):

1. Phase 1 can be driven by a hand-edited `input.in` + hand-canonicalized FCIDUMP before the python
   plumbing lands.
2. Expected: `std(Occs)` < 1e-3 at every iteration (vs the documented polarization blow-up);
   impurity entropy S ≈ ln 15 + charge-fluctuation correction ≈ 2.75–2.85, matching the healthy
   `Doping/nbaths_15/J_0/U_50.00` control; `symmetrize_solver_output` discarded weight collapses to
   ~1e-6 or below; E_ASCI shows no variational excess against the cross-Hamiltonian bound
   (`debugging-single-site-dmft.md` §2.1). Run `check_dmft_health.py` on the tree.
3. Negative control: the same run with `SYMMETRIZE_DETS = FALSE` reproduces the polarization.

---

## 10. Sequencing / landability

| Phase | Contents | Independent? |
|---|---|---|
| 1. MACIS core | Steps 1–6, 8, 9 + tests 9.1–9.3 | Yes — fully testable with hand-written `input.in`/FCIDUMP; no python changes needed |
| 1b. MACIS quality | Step 7 (seed closure), Step 10 (RDM check) | Yes — pure additions on top of Phase 1 |
| 2. PolClassy | P1–P4 + production A/B (9.4) | Depends on the Phase-1 binary being deployed |

## 11. Out of scope

- **Site-permutation (cluster point-group) symmetry** — not an orbital permutation in the Irrep bath
  basis (2D irreps mix bath orbitals by rotation); would require CI-space projectors or
  signed/generalized permutations. The python-side `symmetrize_fct` Manhattan-class averaging
  (capped at 2×2 clusters) remains the only site-channel mitigation.
- Symmetrizing coefficients (Davidson already delivers them symmetric to `ci_res_tol` once the space
  is closed).
- `wfn_t<128>` instantiations, molecular point-group symmetry, `SolveImpurityCheapASCI` /
  `SolveImpurityED` (ED is exact and needs no fix; cheap-ASCI can call the same closure later).
- Automatic HF-orbit seeding when the reference is not G-invariant (v1 throws with a clear message).
- The refine-stall guards of `asci-stall-guards.md` (R2/R3/R4) — independent failure mode, still
  needed; this feature does not replace them and partially blinds the polarization-based alarms they
  rely on (§1 caveat).

---

## Critical files

- `include/macis/asci/determinant_search.hpp` — `ASCISettings` :34-63; MPI top-K score gather
  :600-653; closure insertion :661-687
- `include/macis/asci/determinant_symmetry.hpp` — **new**: permutation utilities, group expansion,
  whole-orbit selection
- `include/macis/asci/grow.hpp:47-70` — stall guard; `refine.hpp:52-53` — relaxed invariant;
  `iteration.hpp:22-28` — seed closure (phase 1b)
- `src/macis/impurity_solver.cpp` — `prepare_det_symmetry` calls at :81-135 and :215-299; guess
  closure :135-163
- `main/run_asci_impsolv_dop.cxx:176-203` — keyword plumbing (+ mirror
  `main/run_asci_impsolv_mu_vs_n.cxx:155-171`)
- `tests/determinant_symmetry.cxx` — **new**; `tests/CMakeLists.txt`
- PolClassy_DMFT: `constANDparams.py` (:217, :486, :544), `Solver.py` (:392-429, :865-883, new
  `band_permutation_generators` helper)
