# Symmetry-sector ground-state search: consolidated plan

> **STATUS (2026-10-03): PARTLY IMPLEMENTED.** A reduced first step has landed: band-parity sectors
> for a band-diagonal bath at NROTS = 0 (single site or 1×2), described in
> `parity-sector-solve-simple.md`. Everything else here is still open. The status table after this
> header says, item by item, what is done. No production input has changed, and nothing has been
> run on production Hamiltonians yet.
>
> **Consolidates and supersedes:**
> - `parity-sector-solve.md`: the 1×2 band-parity plan, 2026-10-02/03.
> - `parity-sector-solve-2x2-context.md`: the 2×2 amendments and the review in its §9.
> - `parity-sector-solve-independent-assessment.md`, with `parity-sector-assessment-checks.py`
>   and `-results.json`.
>
> The last three now live in `symmetry_sector_files/`. Those files stay as the record. Where they disagree with this one, this one applies; §7 lists the
> superseded claims. Solver tree: `MACIS_fork/MACIS_claude` at `aa26eaf`. Line numbers refer to
> that commit.
>
> **Revised 2026-10-03 (Irrep bath structure).** Reading `Fits.py` showed that the Irrep bath is
> momentum-pure by construction (F12). §1.4, §3.2, §3.3, §3.4 and D2 are updated: the "split E by
> diagonalizing T_x" step is dropped. What remains for K at NROTS ≥ 1 is the `fullM` impurity
> rotation and a finer `group_of` (one group per (band, K) channel) in the per-band natural
> orbitals that already exist. A read-only check of archived rotation matrices decides how urgent
> that is (§3.4).

---

## Implementation status (2026-10-03)

Code on branch `claude/happy-franklin-1lfi3g`. Details, deviations and tests:
`parity-sector-solve-simple.md` §11.

| Plan item | Status | Where / what remains |
|---|---|---|
| §3.1 refine 2-cycle detection, `NONCONVERGED_CYCLE` | open | refinement still throws on non-convergence (F8) |
| §3.1 per-call coverage status | **done for parity sectors** | a failed sector is reported `FAILED`, the call prints `PARITY_COVERAGE COMPLETE/INCOMPLETE`, and the winner is the lowest converged sector |
| §3.1 isolation; winner outputs written once (F4, F5) | **done for parity sectors** | each sector solves an isolated copy of `impurity_params`; `active_ordm.dat` and `rot_matrix.dat` are written once, for the returned state |
| §3.1 cold-seed closure bug (F3) | open | |
| §3.2 bath modes A/B (SDP projection) | open | a bath that is not band-diagonal is refused, with the offending coupling named |
| §3.3 labels: band parity | **done** (band-diagonal bath, band-major layout) | detected from T, verified and cleaned against every integral, `ASCI.PARITY_TOL` |
| §3.3 labels: K, point group, orbit representatives | open | 2×2 out of scope. On the Irrep bath, the bath orbitals already carry K (F12); what is missing is the `fullM` impurity rotation, (band, K) groups in `group_of`, and the sector table |
| §3.4 per-band natural orbitals, sector-preserving restarts | **done** | `PARITY_SOLVE` now runs with NROTS > 0: natural orbitals per (impurity\|bath) × band block, restarts moved into the sector, inherited charge-sector bases checked; `parity-sector-solve-simple.md` §12 |
| §3.4 covariant natural orbitals, `LABEL_LEAK` | **partly done** | `LABEL_LEAK` is printed before every rotation; the covariant choice (and with it `SYMMETRIZE_DETS` with NROTS > 0) is open |
| §3.5 seeds | **partly done** | one seed per sector: the energy-ordered reference if it lies in the sector, otherwise repaired and descended on ⟨D\|H\|D⟩. The energy-ordered seed is now the default for every run (`ASCI.HF_BY_ENERGY`, F3's "energy-sorted only with `SYMMETRIZE_DETS`" no longer holds) |
| §3.5 several starts per sector, level-allocation candidates | open | |
| §3.6 ⟨S²⟩, ⟨R⟩/⟨σ_d⟩, `SPIN_CHECK` | open | |
| §3.7 sector wrapper, all solver paths (F6) | **done for ED, `SolveImpurityASCI`, `SolveImpurityASCI_rot`** | entered inside the solvers, so the driver, the µ search and the charge-sector search use it. Cheap mode is not wrapped: it re-diagonalizes the winner's space and so stays in the winner's sector |
| §3.7 sector outputs and keys | **done** | `PARITY_SECTOR` lines, `parity_sectors.dat`, `PARITY_TIE`; keys `ASCI.PARITY_SOLVE/TOL/ETOL/ONLY` in all three impurity drivers |
| §3.7 warm starts split by label | **done** | a guess spanning several sectors is split into `<fname>.par_<key>` slices, each re-diagonalized |
| §3.7 `SYMMETRIZE_DETS` interplay | **done** | each sector uses the stabilizer of its parity key |
| §3.7 parallel sector loop | open | sectors run one after another |
| §3.8 screening; branch-consistent µ search | open | sectors are re-solved at every µ evaluation, but a jump in n(µ) is not detected |
| §3.9 odd-N G↑/G↓; orbit-aware projection | **spin average done**; orbit-aware projection open | `evaluate_GF`: when NALPHA ≠ NBETA (every odd N) the returned G is [G_m(a,b) + G_m(flip a, flip b)]/2, the average over the solved state and its degenerate spin flip (with SU(2), the whole multiplet). The opposite channel comes from a permutation when `GF.IS_UP_COMP` lists both spins of every orbital, else from a second GF run. Skipped (and reported) when T ≠ Td; `GF.SPIN_AVERAGE = FALSE` turns it off. Parity ties are only reported |
| §3.10.1 count labels (J_P = 0) | detected, not handled | warned as `counts_conserved` |
| §4 step 7 (DMFT side) | open | outside this repo |
| §5 V2/V9c-type checks on a toy | **done** (single rank) | two-band toy: every sector matches exact diagonalization; the wrapper returns the true ground state where the legacy solve is 0.039 Ha high |
| §5 V3 (1×2 U = 70), V9c on production Hamiltonians | open | first thing to run |
| MPI | not validated | ASCI on the toy models already fails on 2 ranks without these changes; MPI issues are being debugged on another branch |

## 0. Summary

The impurity solver returns the lowest state it can reach from its start, which is not
necessarily the ground state. There are three independent pieces of evidence:

1. **1×2, U = 70.** The ground state is 0.148 Ha below the production result. It lies in a band-parity
   sector the production seed can never reach.
2. **2×2, U = 4.** `SECTOR_CHECK` is negative in every archived iteration and does not shrink with
   the determinant count. The returned states are not the grand-canonical ground state at their µ.
3. **2×2, U = 0.** The error is 1.42 Ha, and it lies **inside** the correct (parity, K) sector: one
   electron pair sits in the wrong single-particle channel.

So a better single seed will not fix this. **The search has to cover the competing states:**
- solve each sector of the exactly conserved, determinant-diagonal labels separately;
- run several competitive starts inside each sector;
- measure the symmetries that cannot be labels (total S, point group) on the result;
- report honestly when convergence or coverage is incomplete.

**Order:**
1. bookkeeping and status;
2. bath symmetry;
3. labels and sector-preserving natural orbitals;
4. seeds and multiple starts;
5. a parallel sector wrapper;
6. validation on frozen Hamiltonians (1×2 U = 70, 2×2 U = 0, 2×2 U = 4);
7. screening and the µ search;
8. DMFT.

---

## 1. Evidence

### 1.1 1×2, U = 70, frozen `It_10` Hamiltonian (`…/J_0.25/Parity_Test/U_70.00_It_10/`)

| Run | Band parity | E(CI) | total ⟨S²⟩ | ⟨R⟩ (site swap) |
|---|---|---|---|---|
| (5,5), production default seed, 600k and 1.2M dets | even | −95.279685 | 0.000000 | +1 |
| (5,5), hand-built seed `u00d2222`, 600k and 1.2M | odd | **−95.427914** | 2.000000 | +1 |
| (6,4), hand-built and default seeds | odd | −95.427914 | 2.000000 | +1 |

- **The ground state:** total S = 1 (an impurity spin triplet, not screened by the closed-shell bath),
  band singlet, odd parity.
- **Reproduce with** `analyze_parity.py`, `total_spin.py` and `site_parity.py` in `Parity_Test/`.
- **Earlier sector scans on the 1×2 point the same way.** Their Sz = 1 solves came out below Sz = 0
  at U = 32 (−0.036), 48.01 (−0.010), 57 (−0.172) and 60 (−0.0065). SU(2) requires
  E(Sz = 0) ≤ E(Sz = 1), so those Sz = 0 solves missed states. Most likely this is the same parity
  trap (an Sz = 1 default seed can land in odd parity), but it has not been checked case by case.

### 1.2 2×2 `SECTOR_CHECK` (`g100_move/2band/Nimp2x2/Doping/SDR/Nb16/J_0.2/`)

| Run | (Nα, Nβ) | dE_add ≤ | dE_rem ≤ | Source |
|---|---|---|---|---|
| `RUN_U4_Irrep_N18_400k` It_1, It_2 | (9,9) | −0.0662, −0.0872 | +0.0363, +0.0429 | read |
| `RUN_U4_Irrep_N18_750k`, `RUN_U4_Irrep_N18` (1.5M) | (9,9) | −0.065 … −0.084 | +0.034 … +0.041 | reported |
| `RUN_U2_Irrep` It_1 | (10,9) | −0.0704 | −0.3155 | read |
| `RUN_U2_Irrep` It_2 | (10,9) | **+0.2946** | **+0.0382** | read (OK) |

**What it proves** (`include/macis/impurity_solver.hpp:92-129`): a negative value means a state
with N±1 electrons lies below the **returned** E(N). It doesn't tell a wrong N apart from a trapped
or truncated state at fixed N. A local maximum in N is not proof of trapping either, because E(N)
need not be convex (pair binding on plaquettes). A positive value certifies nothing.

### 1.3 2×2, U = 0 (`SOLVER_TESTS/U_ladder/U_0.0`)

Reproduced by re-running `parity-sector-assessment-checks.py` (bit-identical output, 2026-10-03).

- **Sector:** Nα = Nβ = 14, so total N = 28. Earlier notes call this "N = 14".
- **Energies:** exact −16.791310; archived ASCI −15.368085; error +1.423225 Ha.
- **Symmetry:** both states are in **even parity, K = (0,0)**.
- **What went wrong:** the archived state's spin-summed channel counts are
  [4,4,4,4,4,4,2,2]. One pair sits in the level at +0.7565 instead of +0.0448, which costs
  2 × 0.7116.
- **Exact minimum of each sector** (dynamic programming over all one-body levels):

| Parity \ K | (0,0) | K1 | K2 | (π,π) |
|---|---|---|---|---|
| even | **−16.791310** | −16.114310 | −16.114310 | −15.973709 |
| odd | −15.946267 | −16.114310 | −16.114310 | −15.973709 |

At U = 0 the occupation of each (band, K, spin) channel is conserved. These are finer walls than
the sector labels.

### 1.4 Bath symmetry of the archived inputs

Both reviews measured this, by different metrics, and they agree.

| Input | cross-band residue | cross-K residue | C4 / σ_d defect | Δ change if projected |
|---|---|---|---|---|
| `RUN_U4_Irrep_N18/It_2` | 0 | 3e-17 | 2.2e-7 (commutator), 4.3e-7 (weight difference) | 9e-17 (rel.) |
| `RUN_U0.5_SDP_GFall/It_1` | 3.9e-4 | 0 | 0 | 0.068 % |
| `TEST_D_SDP_b157/It_3` | 1.2e-2 | 5e-15 | 2e-15 | **11.7 %** |

Rotations inside a degenerate pole leave Γ = VᵀV unchanged. The SDP cross-band residues are
therefore a real defect of the Hamiltonian, not a choice of basis.

**Why the Irrep row looks the way it does** (F12): every Irrep bath orbital couples to exactly one
(band, irrep) channel, and the four c4v irrep vectors are the four cluster momenta. The cross-K
residue is therefore zero by construction (3e-17 is round-off). The C4/σ_d defect has a single
source: E_1 and E_2 share ε exactly (`degens`), but each bath orbital has its own V, so
V_{E_1} − V_{E_2} is a free fit residual of about 2e-7. It breaks C4 but not translations.

---

## 2. Solver facts that shape the design (verified in source)

| # | Fact | Where |
|---|---|---|
| F1 | Each ASCI iteration discards the previous vector and starts Davidson from the **single lowest-diagonal determinant** of the space. The preconditioner is diag(H) − E. | `asci/iteration.hpp:64` (`X_local; // Precludes guess reuse`), `asci/grow.hpp:137`, `solvers/davidson.hpp:65` |
| F2 | Every macro iteration after the first restarts from `dets = {hf_det}`. `hf_determinant_byocc` fills the **same** orbitals for α and β, so at Nα = Nβ the restart is closed-shell. | `impurity_solver.cpp:617`, `sd_operations.hpp:74-88` |
| F3 | The cold seed of macro iteration 1 is `asci_reference_determinant` (one-body order, energy-sorted only with `SYMMETRIZE_DETS`). Its group closure (`:597`) is discarded at `:617`. This bug already exists in the current code. | `impurity_solver.cpp:119, 597, 617` |
| F4 | Natural orbitals come from `gesvd` on the whole impurity block and the whole bath block, with no split by band or K. The integrals in `p.T_active` and `p.V_active` are rotated **in place**. | `hamiltonian_generator/rdms.hpp:120ff` |
| F5 | `active_ordm.dat` and `rot_matrix.dat` are written inside the solve. | end of `SolveImpurityASCI_rot` |
| F6 | Callers: the driver (`run_asci_impsolv_dop.cxx:472`), `Mu_Cost_f` (`fix_mu.cpp:261`) and `charge_sectors.cpp:285` use `SolveImpurityASCI_rot`. **But** `fix_mu.cpp:155` uses `SolveImpurityASCI`, cheap mode uses `SolveImpurityCheapASCI` (`fix_mu.cpp:250-265`), and CAS uses `SolveImpurityED`, which has the same diagonal-guess trap. | as listed |
| F7 | `load_asci_guess` requires NROTS = 0, so there are no warm starts with rotations. | `impurity_solver.cpp:35` |
| F8 | Refinement throws if it doesn't converge. | end of `asci/refine.hpp` |
| F9 | `MCSCF.CI_NSTATES > 1` switches to LOBPCG, but only root 0 goes back to the ASCI search. Searching around several roots would be new code. | `solvers/selected_ci_diag.hpp:65-85` |
| F10 | The GF spin channel is chosen per orbital by `GF.IS_UP_COMP` (the DMFT key `UPoComp`). `RUN_U2_Irrep` at (10,9) computes **↑ only**. | `run_asci_impsolv_dop.cxx:626`, `Solver.py:1011` |
| F11 | The DMFT symmetry projection of G is **off** by default (`symmetrize_solver_output = False`). When on, it **raises** above `symmetrize_warn_thresh = 0.5`. | `constANDparams.py:218-219`, `Solver.py:1357-1362` |
| F12 | **Irrep bath (c4v, 2×2).** Each bath orbital n couples to one band, (n // 4) % nbands, through one vector, `vs_ind[n] * v_dict[irrep]`, so the bath is band-diagonal and single-channel by construction. The vectors are v_A1 = (1,1,1,1)/2, v_B2 = (1,−1,−1,1)/2, v_E_1 = (−1,1,−1,1)/2, v_E_2 = (1,1,−1,−1)/2. With the site order i → (i // nsitesY, i % nsitesY), these are K = (0,0), (π,π), (0,π), (π,0). The table below holds for every one of the 24 site orderings: T_x, T_y and T_xT_y form the Klein group, which acts as XOR on the indices, and the Hadamard rows are its characters. Only which irrep carries which K depends on the ordering. With this ordering B2 is (π,π), so `degens = [1,1,2]` ties ε for the right pair (E_1, E_2). V is not tied: one `vs_ind` per bath orbital. With `band_sym`, the parameters are copied to every band, so the bands are exactly degenerate. The impurity rows of the FCIDUMP are in the site basis; `fullM` = blockdiag(M_to_irrep, …) is the per-band momentum transform. 16 baths = 2 bands × 4 channels × 2 orbitals: each (band, K) channel holds 1 impurity and 2 bath orbitals. | `Fits.py` (PolClassy_DMFT): `IrrepStruct` `:23-91`, `expand_uniq_eps` `:93`, `expand_ind_vs` `:102`, `band_sym` tiling `:481-506`, site order `:931` |

Irrep → K under F12's site order (T_x and T_y eigenvalues):

| Irrep | (T_x, T_y) | K |
|---|---|---|
| A1 | (+1, +1) | (0,0) |
| B2 | (−1, −1) | (π,π) |
| E_1 | (+1, −1) | (0,π) |
| E_2 | (−1, +1) | (π,0) |

### 2.1 Which symmetries trap, and how

| Kind | Examples | Behaviour in this solver | Treatment |
|---|---|---|---|
| **Label diagonal on determinants** | band parity (band-pure basis); per-band counts without pair hopping, per band-and-spin counts without spin flip (§3.10.1); total K (momentum-pure basis); Sz | Hard wall. H, diag(H) and the start determinant all respect it, and a space that mixes sectors converges in the sector of its lowest-diagonal determinant (F1). | Explicit sector loop; never mix sectors in one solve. |
| **Signed orbital permutation** | site swap (1×2), σ_d/C4 (2×2), K in the site basis, band swap | H and diag(H) commute with it. It traps **when the lowest-diagonal determinant of the space is invariant** (typically closed shells), and otherwise doesn't. | Seed candidates of both kinds; measure on the winner; reduce sectors by orbits. |
| **Neither** | total S | The preconditioner mixes S, so it is never a strict wall. Selection can still favour the wrong S. | Measure ⟨S²⟩; Sz consistency checks. |
| **Approximately conserved** | (band, K, spin) channel counts as U → 0 | Exact walls at U = 0; barriers at small U. | Several starts with different channel allocations. |

---

## 3. Design

### 3.1 Bookkeeping and status (lands first)

- **Refine cycles.** Detect a 2-cycle by energy recurrence and, better, by recurrence of the
  determinant space. Keep the lowest valid state and return it with status `NONCONVERGED_CYCLE`:
  don't throw (F8), and don't silently count it as converged. A cheap recovery to test: solve once
  in the union of the two cycling spaces.
- **Status.** Each solve reports `CONVERGED`, `NONCONVERGED` or `FAILED`. Each call reports its
  **coverage**: `COMPLETE` (every sector and start finished) or `INCOMPLETE` (with the list of what
  failed). A failed competitor never certifies the survivor.
- **Isolation.** Each sector or start begins from an identical copy of the base Hamiltonian (F4).
  Its diagnostics go to its own subdirectory. The winner's `wfn.out`, `active_ordm.dat` and
  `rot_matrix.dat` are written once, at the end (F5).
- **Closure bug.** Fix the cold-seed closure lifecycle (F3).

### 3.2 Bath: exact adaptation vs explicit projection (DMFT side, Python)

There are two modes, and they must be kept distinct:

- **Mode A, exact adaptation.** Rotate only inside degenerate poles, which leaves Γ and Δ exactly
  unchanged. This produces band-, K- or σ_d-pure orbitals wherever Γ already commutes with the
  group. Check: Δ is unchanged to 1e-12.
  - **On the Irrep bath, mode A is the identity** (F12): every bath orbital is already band- and
    K-pure. What's left is a verification: in the `fullM` impurity basis, each bath column of V has
    exactly one nonzero channel. Mode A is needed only for baths that don't have this structure
    (SDP, Replica, Unrestricted).
- **Mode B, symmetry projection.** Project each pole's Γ onto the commutant of the chosen group
  (band-diagonal, translations, band swap, and optionally σ_d). **This changes the Hamiltonian:**
  - record the change (`BATH_SYM_DISCARD`, as a residue metric and as a Δ metric on a frequency
    grid);
  - use the projected bath **consistently** in G0, Σ and the restart files;
  - flag or refuse above a threshold (TEST_D's 11.7 % must not pass silently).
- **Guards.**
  - Each degenerate group must be closed under the group, and no pole's rank may grow. Refuse
    otherwise rather than change the bath size.
  - Keep zero-coupling ("dark") bath orbitals during validation: they still carry electrons and
    energy.
- **Longer term:** enforce the symmetry in the fit itself.
- **Irrep σ_d** (decision D2): the only C4 breaking in the Irrep bath is V_{E_1} ≠ V_{E_2} (F12).
  The cheapest way to make D4 exact is to tie V for E_1 and E_2 in the fit, as `degens` already
  does for ε: a few lines in `expand_ind_vs` and fold/unfold. The other options are to project the
  existing bath (a 4e-7 Hamiltonian change), or to keep (π,0) and (0,π) as separate sectors.

### 3.3 Labels

- **Which labels exist** is detected per Hamiltonian (§3.10.1: count, parity, or none per orbital
  group; K only if translations verify). The bullets below describe the two-band 2×2 case.
- **`symmetry_labels`** gives each active orbital its (band, K) after §3.2.
  - **2×2 Irrep bath:** the bath orbitals already carry (band, K) (F12). No split of E is needed;
    "split E by diagonalizing T_x restricted to E" from the first version applies only to baths whose
    E orbitals are not T_x eigenvectors.
  - **2×2 with NROTS ≥ 1 and K as a label:** rotate the impurity block by the fixed per-band
    momentum transform `fullM` (F12) inside the solver, after µ is set, and start `orb_rot` from it
    (`impurity_solver.cpp:536`). Not in the FCIDUMP: the µ search replaces the impurity diagonal
    in the site basis (`fix_mu.cpp:35-46`), and in a momentum basis that diagonal also holds the
    hopping eigenvalues. Composed into `orb_rot`, G, the RDMs and the observables still come back
    in the site basis. For a state with definite K and band parity, these orbitals are exact
    impurity natural orbitals (ρ_{bk,b'k'} = 0 unless b = b' and k = k'). So relative to the current
    NROTS ≥ 1 runs this is no new approximation, only a fixed choice inside degenerate blocks (§3.4).
    Cost: macro iteration 1 leaves the sparse site basis, at about N_sites² = 16× more on-site
    Kanamori two-body terms. Later macro iterations already pay this in today's natural-orbital
    basis.
  - **2×2 at NROTS = 0:** with the Irrep bath, translations are signed orbital permutations in the
    original basis: they permute the impurity sites (XOR on the site index) and act on bath orbitals
    by signs. So the determinant space can be closed under translations without changing basis,
    as `SYMMETRIZE_DETS` does for band permutations. This corrects the "site permutations out of
    scope" note in `PLAN_symmetric_asci_search.md` §2 for this bath. C4 stays out of the group
    until D2 makes it exact.
  - **1×2 at NROTS = 0:** stays in the site basis. K and the site swap act there as permutation
    symmetries (§2.1), and on-site Kanamori in momentum orbitals has about N_sites² more two-body
    terms.
- **Verify each label separately** on the full FCIDUMP:
  - T conserves it;
  - every V term changes each band's count by an even amount;
  - every V term conserves K.

  A label that fails is reported as `undefined` and not looped over. That doesn't invalidate the
  other labels.
- **nbands > 2:** use a full orbital→band map (not a single band-0 list) and enumerate only feasible
  parity vectors.
- **Representatives:** one per orbit of {band swap, C4 if exact}.

| N | Parity sectors | K classes | Sectors solved |
|---|---|---|---|
| even | 2 | 3 if σ_d exact, else 4 | 6 (or 8) |
| odd | 1 representative | 3 (or 4) | 3 (or 4) |

### 3.4 Natural orbitals that preserve the sector

- **Block-wise diagonalization.** Diagonalize the 1-RDM per (impurity|bath) × band × K block. Log
  the largest off-block element as `LABEL_LEAK`.
  - **Why the whole-block `gesvd` isn't enough, even though the Irrep bath starts pure.** For a
    state with definite labels, ρ_bath = ⊕_{b,k} ρ_{bk}, with each ρ_{bk} a 2×2 block on the Irrep
    bath. `gesvd` on the full 16×16 block (F4) respects those blocks only when no two blocks share
    an eigenvalue. On the Irrep bath they do (F12):
    - `band_sym`: spec ρ_{A,k} = spec ρ_{B,k}, exactly;
    - E_1/E_2: equal to about 2e-7.

    Inside a degenerate group, the rotation is set by the solver's own asymmetry ε (Davidson
    tolerance, uneven truncation), with a mixing angle θ ~ ε/δn. That is of order one when
    δn ≲ ε. The impurity block behaves the same way. After macro iteration 1, the orbitals can mix
    bands and K, and the result depends on round-off.
  - **Status:** the per-band part is done under `PARITY_SOLVE` (the `group_of` argument of
    `rotate_hamiltonian_ordm_imp_bath`; `parity-sector-solve-simple.md` §12). On the Irrep bath,
    K needs only a finer `group_of`, one group per (band, K) channel. The bath orbitals can be
    grouped directly (F12). The impurity orbitals can be grouped after the `fullM` rotation of §3.3,
    which makes every impurity block 1×1, so the impurity no longer rotates. The bath blocks are 2×2.
  - **Check on archived runs** (read-only, NROTS ≥ 1, e.g. `RUN_U4_Irrep_N18/It_2`). Off-block
    weight alone doesn't decide it: `gesvd` sorts by occupation, so the legacy rotation reorders
    orbitals across groups even without degeneracy (`parity-sector-solve-simple.md` §12.3). That
    makes the off-block weight O(1) by reordering alone. Test each column instead: is it supported
    on a single channel (a signed permutation of channel-pure vectors, harmless) or spread over
    several (real mixing)?
    - bath block of `rot_matrix.dat`: the largest single-channel fraction of each column's weight
      (the bath is already in channel order);
    - impurity block: the same for the columns of `fullMᵀ R_imp`;
    - `active_ordm.dat`: cross-channel elements of the impurity block after `fullM`, and the
      splittings between occupations.

    If every column is single-channel to round-off, the degenerate mixing doesn't occur in
    practice, and (band, K) blocks can wait until K becomes a label. Columns split over channels
    are direct evidence that today's NROTS ≥ 1 runs mix labels.
- **Covariant choice.** Take the band-B natural orbitals as the band-swap images of the band-A ones,
  and the (0,π) natural orbitals as the σ_d images of the (π,0) ones. Then band swap and σ_d stay
  signed permutations, so ⟨σ_d⟩ can be measured, and `SYMMETRIZE_DETS` can later be re-enabled with
  NROTS > 0. Until this exists, keep `SYMMETRIZE_DETS` off with NROTS > 0, as the code already
  enforces.
- **Restarts.** Every macro-iteration restart stays in its sector and uses the candidate logic of
  §3.5. It replaces the closed-shell `hf_determinant_byocc` fill (F2).

### 3.5 Seeds and multiple starts within a sector (the central change)

- **Candidate generation.**
  - **Dynamic programming over levels:** one-body eigenvalues in macro iteration 1, natural-orbital
    occupations or energies afterwards. The state is (Nα, Nβ, parity vector, K), and it returns the
    **k lowest** channel allocations per sector, not only the lowest. Use the assessment script's
    U = 0 routine as the reference.
  - **Impurity enumeration for finite U:** fill the bath by energy, enumerate the impurity
    configurations, keep the lowest diagonal energy per sector (the 1×2 plan's §2.4 method). This is
    a second source of candidates.
- **Starts per sector.** `SECTOR_STARTS` single-determinant starts per sector (3–4 during
  validation; 1–2 in production, plus periodic re-checks).
  - **Include** closed- and open-shell candidates, and symmetric and symmetry-breaking ones.
  - **Do not force:** unpairing, symmetry breaking, or a spin-partner ban. All three are refuted:
    at U = 0 a forced open shell excludes the closed-shell ground state (assessment toy), and spin
    is not a wall (F1).
- **Run starts independently** and keep the lowest. Record every start's (E, status, ⟨S²⟩,
  ⟨σ_d⟩/⟨R⟩), so states that nearly compete are visible.
- **Later option:** multi-root selection (F9) as new code. Check it against the multi-start
  results.

### 3.6 Spin and point-group diagnostics

- **⟨S²⟩:** measure the total ⟨S²⟩ of every result (algebra of `Parity_Test/total_spin.py`).
- **Point group:** measure ⟨σ_d⟩ (2×2, K = (0,0) or (π,π)) or ⟨R⟩ (1×2), using the covariant
  natural orbitals of §3.4.
- **`SPIN_CHECK`:** compare Sz = 1 with Sz = 0 (for odd N, 3/2 with 1/2) **in the same parity and
  K sector**.
  - If they're violated, repair with S⁻ inside the matching sector and flag the iteration.
  - Agreement is not proof. If a higher S is plausible, compare more Sz values.

### 3.7 Sector wrapper

- **All solver paths:** wrap every path in F6, or refuse it in validated mode (cheap mode and
  `GROW_WITH_ROT` included).
- **Parallel by default.** Split the MPI communicator, one group per (sector, start), followed by a
  min-reduction. Serial fallback if MPI isn't available.
  - 6 sectors × 3 starts × 4 cores = 72 cores, which fits one 112-core `dcgp` node.
  - Memory at 1.5M determinants still has to be measured.
- **Winner:** the lowest E among `CONVERGED` results, with a near-tie warning at `SECTOR_ETOL`.
- **Outputs:**
  - `symmetry_sectors.dat`: sector label, orbit size, start, E, status, ndets, ⟨S²⟩,
    ⟨σ_d⟩/⟨R⟩, `LABEL_LEAK`, winner flag;
  - a `SECTOR = …` line in `asci.out`;
  - `wfn.out` for the winner, plus a wavefunction per sector for future warm starts.
- **Warm starts** (NROTS = 0 only, F7): split the guess by label. Never put two sectors into one
  solve (F1).

| Key (`[ASCI]`) | Default | Meaning |
|---|---|---|
| `SECTOR_SOLVE` | `FALSE` until validated | enables §3.3–3.7 |
| `SECTOR_LABELS` | inferred and verified | optional explicit orbital → (band, K) map |
| `SECTOR_STARTS` | 3, or 1 if the U = 0 decomposition has a single channel (§3.10.1) | starts per sector |
| `SECTOR_ETOL` | 1e-5 | near-tie threshold |
| `SPIN_CHECK` | `FALSE` | §3.6 |

DMFT-side keys must be added to `constANDparams.Read_Vars`, because `dmft.input` silently ignores
unknown keys.

### 3.8 Cost: screening and µ search (only after validation)

- **Parallelism comes first** (§3.7).
- **Screening** (small budgets, then a full solve of the leading sectors) is allowed only after the
  all-sector validation (V4) shows the screening order is stable: at two budgets, and at NROTS 1
  vs 3.
  - `SCREEN_MARGIN` is a measured policy with a stated uncertainty, not a guarantee. A screening
    energy is an upper bound only.
  - Keep several candidates per sector.
- **µ search:**
  - run it in the winner's branch;
  - at the converged µ, re-check every sector;
  - **repeat until branch and filling agree**, or report an unresolved crossing or density jump.
    That case must not be labelled converged.

### 3.9 Green's function and degeneracy

- **Sz = 0, integer S.** For the m = 0 member, G↑ = G↓ = the multiplet average (Wigner–Eckart; the
  assessment's toy agrees to 6e-17). **The DMFT G needs no ensemble code.** Only RDM tensor
  observables differ (⟨Sz₁Sz₂⟩ = −1/4 vs 1/12), and those are analysis only.
- **Odd N (m = ±½).** A single state has G↑ ≠ G↓, so compute both channels and average them.
  `RUN_U2_Irrep` computes ↑ only (F10), so **its G may be biased.** This is a cheap frozen check
  (V5).
- **Spatial degeneracy (orbit size > 1).** The group projection of G equals the orbit-averaged
  ensemble, **provided** the projection is enabled (F11; it's off by default) and every
  ground-state degeneracy lies inside that orbit. The `RuntimeError` above
  `symmetrize_warn_thresh` must take the orbit size into account: an expected defect from a
  degenerate winner is not a broken solver.

### 3.10 Applicability to other models

The design was derived on two-band 1×2 and 2×2 Kanamori clusters. None of it is specific to those
models, **provided labels and groups are detected from the Hamiltonian (§3.10.1), not assumed.**
§3.3 as written hard-codes the two-band, 2×2 case (band-parity vector, band swap, C4), and that has
to be generalized.

| Model | Sector labels | Orbital-swapping symmetries | Conserved channels at U → 0 | Net effect |
|---|---|---|---|---|
| single site, single band | none (one sector) | particle–hole at half filling | none beyond N, Sz | reduces to the current solver; odd-N G issue (§3.9) |
| single site, multi-band Kanamori | band parity; per-band counts without pair hopping; per band-and-spin counts without spin flip | band permutations | per-band counts | **the parity trap can occur** |
| several sites, single band | total K | C4/σ_d, or site swap on the 1×2 | per-K counts | momentum and point group carry everything |
| several sites, multi-band | band parity × K | band swap, C4/σ_d | per (band, K) | everything active (the designed case) |

#### 3.10.1 Detecting the conserved quantities (replaces the fixed band-parity vector of §3.3)

- **Orbital groups.** A group is a band: its impurity orbitals plus the bath orbitals coupled only
  to them. Optionally split each group by spin.
- **What each group conserves.** Test each group g against the full FCIDUMP:
  - if T is block-diagonal in g and every V term conserves N_g, the label is the **count** N_g
    (a U(1) charge);
  - otherwise, if every V term changes N_g by an even amount, the label is the **parity**
    (−1)^{N_g};
  - otherwise g has no label.
- **What this gives in practice:**
  - Kanamori with pair hopping: parity per band.
  - J = 0, or J_P = 0: count per band.
  - Density-density Hund (no spin flip, no pair hopping): count per band and spin. SU(2) is then
    absent, so switch off the §3.6 spin diagnostics. Sz remains a label.
- **Enumerating sectors.** Take every feasible label vector compatible with (Nα, Nβ), with one
  representative per orbit of the verified group.
  - **Counts can multiply sectors quickly** (one per way of distributing N over the bands). Rank
    them by the §3.5 candidate generator (the lowest one-body or diagonal energy per label vector),
    and solve all those within a window of the lowest few.
  - **That window is a heuristic.** Report its width in `symmetry_sectors.dat`, and widen it in
    validation until the winner stops changing.
- **The symmetry group comes from verified signed orbital permutations** of the FCIDUMP, impurity
  and bath together: translations, point-group elements, band permutations. It is not assumed.
  K is defined only if the translations verify.
- **Default number of starts.** `SECTOR_STARTS` defaults to 1 when the U = 0 decomposition has a
  single channel, and to 3 otherwise.

#### 3.10.2 Single site, single band

- **One sector.** There are no labels beyond N and Sz, and no channels at U = 0. With one start
  this is the current solver, plus label detection, which costs little.
- **What still helps:** §3.1 (refine cycles and status), ⟨S²⟩, and the odd-N Green's function.
  - **The odd-N Green's function affects every model.** At odd total N the state is an m = ±½
    doublet, so computing G↑ only (`UPoComp` all true) gives a spin-biased G (§3.9).
- **Particle–hole symmetry at half filling** swaps orbitals, so a particle–hole-symmetric start
  could trap. It is usually harmless, because the ground state is normally particle–hole even.
  Measure it if in doubt.

#### 3.10.3 Single site, multi-band Kanamori

- **The band-parity trap needs only pair hopping, not a second site.** Example: an impurity Hund
  triplet (one electron per band) with an even number of bath electrons per band is odd/odd, and a
  closed-shell Sz = 0 seed cannot reach it. Whether a run is affected depends on the per-band
  parity of (impurity + bath) electrons, so test a representative single-site case before assuming
  those runs are fine.
- **Without pair hopping** (J = 0 or J_P = 0), each band's count is conserved. The 3-band J = 0
  collapse onto two orbitals (`asci-stall-guards.md`) may be this per-band-count trap. That is a
  hypothesis, not established.
- **Without spin flip,** the counts per band and spin are conserved, and total S is not.
- **Degenerate bands have exact cross-sector degeneracies.** The Hamiltonian, bath included, is
  unchanged by any real rotation mixing the bands (verified on the 1×2 U = 70 FCIDUMP to 1e-14).
  So some even- and odd-sector states are exactly degenerate. **A near-tie between parity sectors
  can be physics, not an error:** compare the Green's functions of the tied sectors before
  flagging.
- **Inequivalent bands** (crystal field) keep band parity, but band swap no longer pairs the
  sectors, so solve every parity vector.

#### 3.10.4 Several sites, single band

- **No band labels.** Total K and the point group (C4/σ_d) carry all the structure.
- **The single-band 2×2 plaquette is the textbook case.** Its half-filled and two-hole ground
  states lie in different C4 channels (d-wave pair binding). Closed-shell restarts are always
  K = (0,0) and C4-even (F2), so they are the wrong default there.
- **At small U**, each K channel's count is conserved at U = 0, so several starts are needed (as in
  the 2×2 U = 0 failure, §1.3).
- **On the 1×2 dimer** the only extra symmetry is the site swap (bonding/antibonding). In the site
  basis it permutes orbitals, so the seed choice plus a measured ⟨R⟩ is enough.

#### 3.10.5 Out of scope

- **Spin–orbit coupling:** Sz is not conserved, so the spin labels and spin diagnostics are
  invalid.
- **Intentional symmetry breaking** (D6).
- **Physical band-mixing hybridization** (a genuinely off-diagonal SDP or Replica bath): there is
  no band label, but K may survive. §3.10.1 detects this per label.

---

## 4. Implementation order

| Step | Where | Content | Tested by |
|---|---|---|---|
| 0 | `asci/refine.hpp`, `impurity_solver.cpp` | §3.1: cycle status, coverage status, Hamiltonian isolation, winner outputs, closure bug | V0 |
| 1 | `Solver.py` + an offline tool | §3.2: bath modes A/B, guards, consistent use of the projected bath | V1 |
| 2 | `impurity_solver.cpp`, `rdms.hpp` | §3.3 labels and verification, with §3.10.1 detection of conserved counts and parities and of the symmetry group; impurity momentum basis; §3.4 block-wise covariant natural orbitals, `LABEL_LEAK` | V2 |
| 3 | `impurity_solver.cpp`, `sd_operations.hpp` | §3.5 candidate generation; sector-constrained restarts | V2, V3 |
| 4 | `impurity_solver.cpp`, driver | §3.7 parallel sector × start wrapper; §3.6 diagnostics; outputs and keys | V3, V4, V6 |
| 5 | `fix_mu.cpp`, ED/cheap paths | wrap or refuse the remaining paths (F6) | V7 |
| 6 | `fix_mu.cpp` | §3.8 screening and branch-consistent µ search | V7 |
| 7 | `constANDparams.py`, `Solver.py`, `check_dmft_health.py` | keys in `Read_Vars`; orbit-aware projection check; both spin channels at odd N; health-check reads `symmetry_sectors.dat` | V8 |

Build with `send_compile_claude.job` after backing up the binaries (`binaries_backup_20260828`
pattern). First confirm which tree each campaign runs: the 1×2 runs use `MACIS_claude_build`
(`p2isolv` in `dmft.input`), and the 2×2 document says those runs use `MACIS_build`.

## 5. Validation (frozen Hamiltonians first; DMFT only at V8)

- **V0 refine.** Re-solve `RUN_U4_Irrep_N18_200k/It_1` and `RUN_U0.5_SDP_GFall/It_1`.
  - Pass: both end with `NONCONVERGED_CYCLE`, at the lower branch (reported −12.519268 for the
    200k case at its µ).
- **V1 bath.** Run both modes on the three inputs of §1.4.
  - Pass: mode A is the identity on Irrep.
  - Pass: the mode B metrics reproduce `parity-sector-assessment-results.json`.
  - Pass: TEST_D is flagged or refused.
  - Pass: the closure and rank guards fire on a constructed counter-example (the rank-1 → rank-2
    toy).
- **V2 U = 0 reference.** All eight (parity, K) minima of §1.3 to 1e-6, and the winner equals
  −16.791310.
  - Also: start one run from the archived wrong allocation and show that the multi-start search
    leaves it.
- **V3 1×2 U = 70.** A cold `SECTOR_SOLVE` returns −95.427914, odd parity, ⟨S²⟩ = 2, ⟨R⟩ = +1.
  - The even sector gives −95.279685 with ⟨S²⟩ = 0.
  - Both are unchanged at 600k and 1.2M determinants.
- **V4 2×2 U = 4 frozen** (`RUN_U4_Irrep_N18/It_2`). Every sector × start, at 100k and 400k, with
  NROTS 1 and 3. Table of E, status, ⟨S²⟩, ⟨σ_d⟩ and `LABEL_LEAK`.
  - Pass: `SECTOR_CHECK` passes, or N moves consistently with a re-run charge-sector scan.
- **V5 2×2 U = 2.** Test the It_1 (fail) and It_2 (pass) Hamiltonians.
  - Acceptance allows N to change.
  - Compare G↑ with G↓ at (10,9) (F10).
- **V6 spin.** `SPIN_CHECK` on the V3 and V4 winners.
  - Re-run the 1×2 `--check-spin` violations (U = 32, 48.01, 57, 60); they should disappear.
- **V7 µ search and remaining paths.** The µ search converges with branch consistency, or reports
  a crossing. The ED and cheap paths are wrapped or refused.
- **V8 DMFT.** Restart U = 4 (2×2) and the 1×2 high-U branch from clean directories.
  - Pass: `SECTOR_CHECK` passes every iteration, the winner is stable or switches are flagged, and
    dHyb goes below the current floor.
- **V9 Other model classes (§3.10).** These can run alongside V2–V6.
  - **V9a, single site, single band** (any archived frozen FCIDUMP). With `SECTOR_SOLVE` on, one
    sector and one start are detected, the energy equals the current solver's to refine
    tolerance, and the overhead is a few percent at most.
  - **V9b, odd N** (same model, odd total N). G↑ and G↓ of the m = +½ state differ, and the
    averaged G is what gets returned (§3.9).
  - **V9c, single site, two-band Kanamori.** Use a frozen case with a closed-shell bath per band
    and two impurity electrons. Both parity sectors are solved, and the winner is the Hund
    triplet. With pair hopping zeroed in the FCIDUMP, detection switches to per-band counts, and
    all compositions inside the reported window are solved.
  - **V9d, single-band 2×2 plaquette** with a small bath (≤ 12 orbitals, so full diagonalization
    is feasible). The minimum of every (K, C4) channel matches exact diagonalization, and the
    winner's ⟨σ_d⟩ is correct even when the ground state is C4-odd.

## 6. Decisions for you

- **D1 SDP:** use mode B projection (a recorded Hamiltonian change), or enforce the symmetry in the
  fit?
- **D2 Irrep σ_d:** make D4 exact (6 sectors), or keep 8 sectors? To make it exact, preferably tie
  V_{E_1} = V_{E_2} in the Irrep fit (F12, §3.2); projecting the existing bath (a 4e-7 change) is
  the alternative.
- **D3 Odd-N G:** if V5 shows G↑ ≠ G↓ matters, fix the running U = 2 setup now or after the plan
  lands?
- **D4 Which N:** re-run the charge-sector scans after V4 (1×2 and 2×2).
- **D5 Queued runs:** keep them as "before" references, or cancel?
- **D6 Intentional symmetry breaking** (orbital order, AFM, nematic): switch the sector solve and
  bath projection off; that's a separate set of runs.
- **D7 Odd-N survey:** search all campaigns for runs with Nα ≠ Nβ (any model) whose G was computed
  for one spin only (§3.9, §3.10.2). This is read-only and can be done before anything here is
  implemented.

## 7. Superseded statements

From `parity-sector-solve.md`:
- **§2.1 "all callers go through `SolveImpurityASCI_rot`":** incomplete (F6).
- **§2.4a spin mechanism:** wrong (F1 and the preconditioner). This covers four claims:
  - a pure-S Davidson stays at that S;
  - never include spin partners;
  - zero unpaired impurity electrons means only S = 0 (the bath can carry spin);
  - appending determinants to a warm start can't help.
- **§3 "NROTS > 0 incompatible":** replaced by §3.4.
- **§2.2 band inference with fallback on SDP:** replaced by §3.2.
- **§6 "the ensemble changes the GF":** wrong at Sz = 0 with integer S (§3.9).
- **Validation item 4:** it tests a parity repair via Sz = 1, not a spin repair at fixed parity.

From `parity-sector-solve-2x2-context.md`:
- **§0/§3.3 SDP mixing as a "basis artefact":** partly a real Hamiltonian defect (§1.4).
- **§1 U = 0 "wrong momentum/irrep channel":** it's the wrong channel allocation inside the
  correct (parity, K) sector (§1.3).
- **§1 "µ accounts for at most 0.2 Ha":** invalid. The solver's µ shift changes only impurity
  levels, so dE/dµ = −⟨N_imp⟩.
- **§1/§8 U = 2 "trapped, not wrong N" and "fails every iteration":** overreach, and It_2 passes.
- **§7 "no new code is needed":** the projection is off by default and raises above 0.5 (F11).
- **§3.2 "`SYMMETRIZE_DETS` not needed":** keep it, via covariant natural orbitals (§3.4).
- **Its §9 review:**
  - 9.2's endorsement of §7 is withdrawn;
  - the 9.3.2 rule "seed must not be σ_d-invariant" is replaced by multi-start with both kinds of
    candidate (§3.5).

From the first version of this document (2026-10-03):
- **§3.3 "split E by diagonalizing T_x restricted to E" for the 2×2:** not needed on the Irrep
  bath, which is momentum-pure by construction (F12).
- **§1.4 C4/σ_d defect, source unstated:** it is the untied V_{E_1} − V_{E_2} fit residual (F12).

From `parity-sector-solve-independent-assessment.md`:
- **Nothing superseded.** Its "high-U table not independently reproduced" can now be checked from
  `Parity_Test/` (§1.1).

## 8. Tools and data

- **`…/2band/Nimp1x2/Doping/n_0.25/Irrep/Nb12/J_0.25/Parity_Test/`:** the U = 70 runs (inputs,
  seeds, outputs) and the analysis scripts:
  - `analyze_parity.py`: energy, band parity, spin observables, P_trip;
  - `total_spin.py`: total ⟨S²⟩ from `wfn.out`;
  - `site_parity.py`: site-swap symmetry and ⟨R⟩;
  - `seedtools.py`: determinant energies and seed files.
- **`docs/plans/parity-sector-assessment-checks.py`:** bath residues, the U = 0 sector reference
  and spin toys. Results in `parity-sector-assessment-results.json`.
- **`Fits.py` (PolClassy_DMFT):** the Irrep bath parametrization behind F12 (`IrrepStruct`,
  `expand_ind_vs`, `symmetryze_hyb`).
- **Diagnosis of the 1×2 case:** `…/J_0.25/high_U_triplet_diagnosis.md`.
