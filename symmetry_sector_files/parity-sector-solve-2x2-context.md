# Symmetry-sector solve for the 2×2 two-band cluster: amendments to `parity-sector-solve.md`

> **SUPERSEDED (2026-10-03) by `symmetry-sector-solve.md`**, which consolidates this file with the other
> parity-sector documents; its §7 lists the claims here that no longer hold.
>
> **STATUS: NOT IMPLEMENTED.** Written 2026-10-03, as a companion to `parity-sector-solve.md`
> (2026-10-02), which it amends rather than replaces. That plan was derived on the 1×2 at U = 70;
> this one adds what the 2×2, quarter-filling campaign needs.
> **Reviewed the same day: see §9** (verdict: adopt, with corrections).
> Calculation tree: `g100_move/2band/Nimp2x2/Doping/SDR/Nb16/J_0.2/` (notes `SOLVER_HEALTH.md`,
> `SOLVER_CAMPAIGN.md` there). Solver tree: `MACIS_fork/MACIS_claude` (HEAD `aa26eaf` when written).

---

## 0. Summary

The parity plan's diagnosis carries over unchanged. As written, though, it would not fix the 2×2, for
four reasons:

1. **The 2×2 has a second exact quantum number: total cluster momentum K.** It traps exactly as band
   parity does, and the parity-only loop doesn't touch it.
2. **The parity plan forbids NROTS > 0, but the 2×2 needs NROTS ≥ 3.** The incompatibility is an
   eigensolver artefact and can be removed (§3.2).
3. **The SDP fallback would fire on every SDP run, silently.** The band mixing it detects is a basis
   artefact inside degenerate poles (§3.3).
4. **At 3–14 h per solve, "solve every sector every µ evaluation" is unaffordable.** A screening
   stage is needed (§3.4).

There is also a mechanism the parity plan didn't name, and it is the main one at NROTS > 0 (§2.2):
**every macro iteration after the first restarts from a closed-shell determinant. That fixes even
band parity, K = (0,0) and S = 0, whatever the true ground state is.** It is also σ_d/C4-even
(→ §9.3.2).

---

## 1. Evidence on the 2×2 (read 2026-10-03)

**`SECTOR_CHECK` fails in every archived iteration of every running DMFT run.**
`SECTOR_CHECK` is the band-Lanczos Ritz upper bound on E(N±1) − E(N), printed after the GF stage
(`include/macis/impurity_solver.hpp:103-118`).

| run (Irrep c4v bath) | (Nα, Nβ) | dE_add = E(N+1) − E(N) ≤ | dE_rem = E(N−1) − E(N) ≤ |
|---|---|---|---|
| `RUN_U4_Irrep_N18_400k` It_1 … It_6 | (9,9) | −0.066 … −0.091 | +0.036 … +0.045 |
| `RUN_U4_Irrep_N18_750k` It_1, It_2 | (9,9) | −0.066, −0.084 | +0.034, +0.041 |
| `RUN_U4_Irrep_N18` (1.5M) It_1, It_2 | (9,9) | −0.065, −0.082 | +0.034, +0.041 |
| `RUN_U2_Irrep` It_1 | (10,9) | −0.070 | **−0.315** |

- **Not truncation.** At It_1 the gap is −0.066 / −0.066 / −0.065 Ha at 400k / 750k / 1.5M
  determinants. It would shrink with the determinant count if ASCI were merely unconverged.
- **U = 2 sits at a local maximum in N:** both neighbours are lower. A trapped state, not a wrong N.
  (→ §9.3.1: overreach, and It_2 now passes.)
- **The 10-01 sector scan was trapped too.** `explore_charge_sectors` at 100k put N = 19 at least
  +0.27 Ha above N = 18 (µ = −0.009); the Ritz bound now puts it at least 0.065 Ha below (µ ≈ 0.19).
  The µ difference accounts for at most about 0.2 Ha. **Every E(N) comparison made so far on this
  cluster is between sector-restricted energies.** The choice N = 18 rests on them.
- **U = 0, N = 14** (`SOLVER_HEALTH.md` §6.3, finding B): ASCI −15.368085 against exact −16.791310.
  One electron sits in the +0.7565 level instead of the 4-fold level at +0.0448, i.e. it is in the
  wrong momentum/irrep channel.

**What the running DMFT loops are converging is therefore not the ground state.** No determinant,
GF or fit setting can be calibrated until this is fixed (`SOLVER_CAMPAIGN.md` criterion A2).

---

## 2. Symmetries and how the solver loses them

### 2.1 Exact quantum numbers of the 2×2 impurity Hamiltonian

All of these are conserved by the FCIDUMP Hamiltonian at fixed (Nα, Nβ):

| quantity | why it is conserved | status |
|---|---|---|
| band parity `(-1)^{N_A}` (N_B's parity follows from N) | pair hopping moves 2 electrons; hopping, Hubbard U, U′ and spin flip keep each band's count; bath band-diagonal | **verified:** 1-body `T_AB = 0` in the impurity block. Bath: see §3.3 |
| total cluster momentum K ∈ Z₂×Z₂ = {(0,0), (π,0), (0,π), (π,π)} | intra-cluster hopping and on-site Kanamori are translation invariant on the periodic 2×2 | impurity hopping eigenvalues −2, 0, 0, +2: **verified**. Bath: see below |
| S_z | — | exact |
| total S | SU(2)-invariant Kanamori, spin-independent bath | exact (parity plan §2.4a) |

**What was actually checked for the bath (2026-10-03).** One-body part of the FCIDUMP of
`RUN_U4_Irrep_N18/It_2`, `RUN_U0.5_SDP_GFall/It_1` and `TEST_D_SDP_b157/It_3`. Every bath
orbital's coupling vector lies entirely in **one eigenspace of the impurity hopping**: A₁
(K = (0,0)), B₁ (K = (π,π)) or E (the (π,0)/(0,π) pair). That is purity in irrep, not yet in K inside
E. Each E level is degenerate (pairs in Irrep, eight-fold blocks in SDP), so K-purity inside E can
always be restored by rotating the bath within the level, **provided** the level's weight matrix
commutes with the translations (§3.3). This still has to be checked numerically; it is not yet
verified.

**Group relations between sectors** (they cut the number of distinct sectors to solve):
- C4 rotation maps (π,0) ↔ (0,π): these two K sectors are exactly degenerate partners.
- Band swap maps parity (e,o) ↔ (o,e). At even N it maps each parity sector to itself; at odd N it
  pairs them.
- Spin flip at Nα = Nβ maps each sector to itself.

| N | parity sectors | K classes | **sectors to solve** |
|---|---|---|---|
| even (e.g. 18) | 2: (e,e), (o,o) | 3: (0,0), (π,π), {(π,0),(0,π)} | **6** |
| odd (e.g. 19) | 1 representative | 3 | **3** |

### 2.2 Where the trap is closed, in code (`MACIS_claude`, `src/macis/impurity_solver.cpp`)

1. **Macro iteration 1** starts from `asci_reference_determinant` (`:119`, used at `:577`): raw-index
   filling, impurity first. Its K and parity are whatever the filling happens to give. Its diagonal
   energy is about 71 Ha above the final E(CI) at U = 4 (`check_dmft_health.py`, every iteration of
   the runs in §1).
2. **Every later macro iteration restarts cold from one determinant** (`:613-620`, `dets = {hf_det}`):
   `hf_det = hf_determinant_byocc(nalpha, nbeta, orb_occs)` (`:688`;
   `include/macis/sd_operations.hpp:74-88`). This fills **the same orbitals for α and β**, the
   `min(Nα, Nβ)` most occupied natural orbitals. So at Nα = Nβ the determinant is **closed shell**,
   and therefore:
   - each band's count is twice an integer, so **parity is even**, whenever the NOs are band-pure;
   - total K = 2·Σk_i = **(0,0)**, since every k ∈ Z₂×Z₂ is its own inverse, whenever the NOs are
     K-pure;
   - it is **pure S = 0**.

   At Nα ≠ Nβ (e.g. the U = 2 run, (10,9)) the extra α electrons alone set K and parity, so the
   sector is fixed by the occupation order of a single orbital.
3. **When degenerate NOs come out mixed** (`gesvd` of the impurity block in
   `rotate_hamiltonian_ordm_imp_bath`, `include/macis/hamiltonian_generator/rdms.hpp:120ff`, which
   splits only imp vs bath), the labels are undefined, the seed straddles sectors, and the sector
   reached depends on arbitrary eigenvector phases. This is a plausible source of the E0 jumps between
   iterations seen in earlier runs. **Not verified.**

**Consequence:** with NROTS ≥ 1 and Nα = Nβ, the final wavefunction is almost always in
(even, even) × K = (0,0) × S = 0. The parity plan's U = 70 example is the special case where this was
wrong in parity and spin; at quarter filling on the 2×2, the E level at E_F makes K ≠ (0,0) and odd
parity real candidates.

---

## 3. Amendments to the parity plan

### 3.1 Sector label = (band-parity vector, K, S_z), not parity alone

- Extend `band_parity_labels` (parity plan §2.2) to `symmetry_labels`: for each active orbital, return
  (band, K) after the bath canonicalization of §3.3.
  - Impurity orbitals are site orbitals and carry no K, so the solver must work in a **momentum basis
    for the impurity** (→ §9.3.7): a fixed per-band 4×4 unitary from the eigenvectors of the impurity hopping,
    applied before the first macro iteration (it is a valid `orb_rot`).
  - Inside E, split (π,0)/(0,π) by diagonalizing the translation T_x restricted to E, not the hopping
    (they are degenerate under the hopping).
- **Verify, don't trust** (same rule as the parity plan): every `|T_pq| > tol` must conserve band and
  K, and every `|V_pqrs| > tol` must change each band's count by an even amount and conserve total K.
  If either fails, print why, report the sector as `undefined`, and solve once. Don't throw.
- **Sector enumeration** as in §2.1: one representative per orbit of {C4, band swap}. Record the
  orbit size in `parity_sectors.dat` (rename it `symmetry_sectors.dat`), because a degenerate winner
  matters for the GF (§6).

### 3.2 Keep NROTS: symmetry-adapted natural orbitals

The parity plan (§3) refuses `PARITY_SECTORS` with `NROTS > 0` because "natural orbitals of
degenerate bands are arbitrary band mixtures". This is avoidable, and the 2×2 cannot afford it:
- at NROTS = 1 the ground state failed the stationarity test and had 48 % band asymmetry;
- at NROTS = 3 the asymmetry is 1.5 % and Σ is causal (`SOLVER_HEALTH.md` §6.1, §6.7).

**Why it's avoidable.** For a wavefunction of definite (parity, K), the 1-RDM is exactly
block-diagonal by label: ⟨c†_{A} c_{B}⟩ flips both band parities, and ⟨c†_k c_{k′}⟩ changes K. Any
band or K mixing in the NOs comes from diagonalizing the full imp block at once, not from the state.

**Change.** In `rotate_hamiltonian_ordm_imp_bath`, diagonalize the 1-RDM per
(imp|bath) × band × K block instead of per (imp|bath):
- labels are carried through the rotation by construction;
- `orb_occs` and the label vector stay aligned;
- `rot_matrix_*.dat` is unchanged in format.

Off-block elements of the RDM are a measure of label leakage. Log their max as `LABEL_LEAK` (should
be ≤ 1e-10 for a pure-sector wavefunction), and treat a large value as a bug signal.

**Then the macro-iteration reseed must stay in the sector.** Replace `hf_determinant_byocc` with
`sector_seed_byocc(sector, orb_occs, labels)`:
- the most-occupied determinant **within** the target (parity, K, S_z);
- **open shell where the sector allows** (parity plan §2.4a: largest number of singly occupied
  orbitals the sector permits). Then neither S = 0 nor K = (0,0) nor even parity is imposed by the
  seed;
- enumerate only the near-degenerate window: orbitals whose occupation lies within 0.2 of the
  occupation of the last filled NO. That window is small (the E level is 2–4 orbitals).

This removes the clash between `SYMMETRIZE_DETS` and NROTS that `SOLVER_CAMPAIGN.md` §1 notes.
`SYMMETRIZE_DETS` itself is not needed for this plan. (→ §9.3.5: keep it.)

### 3.3 Canonicalize the bath on the DMFT side before writing the FCIDUMP (new, Python)

- **SDP:** every bath orbital as written couples equally to both bands (|V_A| = |V_B| for all 16
  orbitals). Taken per pole, the cross-band weight max|Γ_AB| / max|Γ_AA| is:
  - **4.7e-4** at U = 0.5 (`RUN_U0.5_SDP_GFall`);
  - 4.6e-2 in `TEST_D_SDP_b157` It_3. That run fitted a Δ from the pre-fix acausal,
    band-asymmetric Σ.

  `_band_diagonal_bath_bands` (`Solver.py:101`) tests each column separately, so it raises on any
  SDP bath. The parity plan's §2.2 inference (connected components of `|T_pq| > 1e-8`) would fail in
  the same way, and the run would silently drop to a single, trapped solve (parity plan validation
  item 6).
- **Fix.** New `canonicalize_bath(eps, V, sym)` in `Solver.py`, applied just before the FCIDUMP is
  written:
  1. Group bath orbitals into degenerate poles (|Δε| < 1e-10).
  2. For each pole, form Γ = Vᵀ V (n_imp × n_imp) and project it onto the commutant of the impurity
     symmetry group: zero the band-off-diagonal blocks, and average over translations and band swap.
     Log the discarded norm as `BATH_SYM_DISCARD`. It is the fit's symmetry error and must stay at
     fit-noise level; flag above 1e-2 relative.
  3. Diagonalize the symmetrized Γ within each (band, K) block. New bath orbitals have couplings
     `√λ_k u_kᵀ`, one per nonzero eigenvalue. Drop zero-weight directions (they decouple).
  4. Assert: Δ(iω_n) rebuilt from the canonical bath equals the input Δ up to `BATH_SYM_DISCARD`.
     (→ §9.3.6: also assert no rank growth; §9.3.2: project onto D4, not only translations.)
- This handles Irrep (where it should be the identity up to E-pair rotation) and SDP uniformly. It
  also makes the per-column `_band_diagonal_bath_bands` and `SYM_PERM` work for SDP.
- **Physics check before adopting:** the lattice Δ really is band-diagonal and translation
  symmetric here: the dispersion `2x2_AllBandsEqual.latt` is band-diagonal, and a symmetric Σ keeps
  it so. If a run is meant to break band or translation symmetry (orbital order, nematicity), the
  projection must be switched off (`bath_symmetrize = .false.`) and the solver falls back to a single
  solve.
- **Not recommended:** relying on fit noise to break the symmetry. A coupling of order 1e-4 Ha between
  sectors gives the other sector's determinants ASCI scores near zero. They are never selected, the
  trap survives, and the sector label is lost while it does.

### 3.4 Screening instead of full solves of every sector

Measured costs on the 2×2, U = 4, NROTS = 3, one core:

| determinants | ground state + GF per DMFT iteration |
|---|---|
| 400k | 3 h |
| 750k | about 7 h |
| 1.5M | about 14 h |

A µ search makes 8–16 solves. Six sectors at full size per µ evaluation would multiply this by 6.

1. **Screen** (→ §9.3.3, §9.3.4)**:** cold-solve every representative sector at `SCREEN_DETS` (default 100k), NROTS 1. The
   sector scan at 100k was within 6 mHa of larger runs, against inter-sector gaps of 65–315 mHa.
2. **Full solve:** only the sectors within `SCREEN_MARGIN` (default 20 mHa) of the screening winner.
3. **µ search:** run it in the current winner's sector only. At the converged µ, re-screen. If the
   winner changed, redo the µ search once in the new winner and stop there, flagging the iteration.
   Jumps in n(µ) like the one in `EXP_U5_GFall_NROT3` (n 0.513 → 0.494 between µ = 0.81 and 0.82)
   are probably sector crossings, and this procedure exposes them instead of bisecting into them.
4. **Re-screen** every `SCREEN_EVERY` DMFT iterations (default 3), always when `SECTOR_CHECK` fails,
   and always when the winner of the previous iteration was within `SCREEN_MARGIN` of another sector.
5. **Warm starts:** `load_asci_guess` requires NROTS = 0 (`impurity_solver.cpp:35`), so with NROTS ≥ 1
   there are no warm starts across DMFT iterations. With label-preserving rotations (§3.2), a guess
   could be stored together with its `rot_matrix` and re-expressed. This is out of scope here; note it
   as the next cost lever after MPI.

### 3.5 Unchanged from the parity plan, and still required

- §2.4a spin-free seeds and the optional S_z = 1 cross-check (`SPIN_CHECK`). At NROTS ≥ 1 the
  closed-shell reseed of §2.2 makes this mandatory, not optional, during validation.
- §2.5 split warm-start guesses by sector (applies once warm starts with NROTS exist).
- §2.6 winner = lowest E, with a near-tie warning; §2.7 outputs. Add `K` and `orbit_size` columns,
  plus `LABEL_LEAK` and `BATH_SYM_DISCARD`.
- Charge-sector search (`doping/charge_sectors.cpp`) inherits the wrapper and becomes meaningful.
  **Re-run the N scans on the 2×2 after this lands; the N = 18 choice must be revisited.**

---

## 4. Prerequisite: the refine 2-cycle must not kill a run

More solves per iteration means more chances to hit `throw std::runtime_error("ACCI Refine did not
converge")` (`include/macis/asci/refine.hpp:80`). It has already killed:
- `RUN_U0.5_SDP_GFall`, after 10 h (cycle amplitude 5.7e-6);
- `RUN_U4_Irrep_N18_200k`, after 1 h (±1.82e-5, against etol 1e-5, oscillating to iteration 200).

Change:
- Detect a period-2 cycle: |E_k − E_{k−2}| < etol·1e-2 and |E_k − E_{k−1}| ≥ etol.
- Stop there, keeping the lower of the two determinant sets (store both, since the solver already
  holds the current one).
- Log `REFINE_CYCLE amplitude=…`.
- At `MAX_REFINE_ITER`, warn and return the lowest state seen instead of throwing.
- The sector wrapper additionally catches per-sector exceptions and drops that sector with a logged
  reason, unless it is the only one.

This is independent of everything else and should land first.

---

## 5. Implementation order

| step | where | content | testable on |
|---|---|---|---|
| 0 | `asci/refine.hpp` | §4 cycle-tolerant refine | V0 |
| 1 | `Solver.py` (+ small offline tool) | §3.3 `canonicalize_bath`, `BATH_SYM_DISCARD` logging, Δ round-trip assert | V1 |
| 2 | `impurity_solver.cpp`, `rdms.hpp` | §3.1 labels + verification; impurity momentum basis; §3.2 block-wise NOs, `LABEL_LEAK` | V2 |
| 3 | `impurity_solver.cpp`, `sd_operations.hpp` | `sector_seed_byocc` (open shell), sector seeds for macro 1 | V2, V3 |
| 4 | `impurity_solver.cpp` | wrapper with screening (§3.4 steps 1–2), outputs | V3, V4, V5 |
| 5 | `fix_mu.cpp` | µ search in the winner's sector, plus re-screen (§3.4 step 3) | V6 |
| 6 | `constANDparams.py`, `Solver.py`, `check_dmft_health.py` | DMFT keys (`asci_sector_solve`, `screen_dets`, `screen_margin`, `screen_every`, `bath_symmetrize`): add them to `Read_Vars`, since unknown keys are silently ignored. Health check reads `symmetry_sectors.dat`, flags winner switches, near-ties, `LABEL_LEAK`, `BATH_SYM_DISCARD` | V7 |

Build with `send_compile_claude.job`, after backing up `MACIS_claude_build` binaries as in
`binaries_backup_20260828`. Production runs currently use `MACIS_build` (the fork with a169777);
`MACIS_claude` has had a169777 merged since 09-30, but confirm that before switching production to it.

---

## 6. Validation

All on frozen Hamiltonians already on disk; no DMFT loop until V5 passes.

- **V0 refine.** Rerun the `RUN_U4_Irrep_N18_200k/It_1` µ-search solve and `RUN_U0.5_SDP_GFall/It_1`.
  - Pass: both terminate with `REFINE_CYCLE`, with E equal to the lower branch (−12.519268 for the
    200k case at its µ).
- **V1 bath (offline, seconds).** Canonicalize the baths of the three FCIDUMPs in §2.1.
  - Pass: every bath orbital is pure in (band, K) to 1e-12, and Δ is reproduced.
  - Expected `BATH_SYM_DISCARD`: about 5e-4 for U = 0.5 SDP and about 5e-2 for TEST_D. The latter
    should be flagged.
- **V2 U = 0, exact per sector (offline reference, seconds).** At U = 0 the lowest state of each
  (parity, K) sector is a Slater determinant. Compute it by constrained aufbau over low-lying
  particle-hole moves in the one-body eigenbasis.
  - Pass: ASCI per sector equals it to 1e-6.
  - Pass: the winner equals −16.791310 (`SOLVER_TESTS/U_ladder/U_0.0`).
  - Caveat: at U = 0 every (band, K, spin) channel occupation is conserved, more than the sector
    label. This test therefore also checks that `sector_seed_byocc` picks the lowest channel
    configuration within a sector. A failure there is a seed bug, not a sector bug.
- **V3 U = 4 frozen** (`RUN_U4_Irrep_N18/It_2` FCIDUMP, (9,9), and (9,10)/(10,9) for comparison):
  - table of E(CI), ⟨S²⟩, `LABEL_LEAK` per sector at 100k and 400k;
  - Pass: `SECTOR_CHECK` dE_add ≥ −(ASCI error, about 5 mHa), or the winning N moves and the
    charge-sector search agrees;
  - Pass: the screening ordering is the same at 100k and 400k for all sectors more than 20 mHa apart.
- **V4 U = 2 frozen** (`RUN_U2_Irrep/It_1`, (10,9)).
  - Pass: both dE_add and dE_rem become ≥ −tol. It is currently at a local maximum in N.
- **V5 SU(2).** `SPIN_CHECK` on the V3 and V4 winners. No violation, or a documented S ≥ 1 ground
  state.
- **V6 µ search** with `DOPING = TRUE` on the V3 input: it converges. Each evaluation logs its sector,
  and the final re-screen agrees.
- **V7 DMFT.** Restart U = 4 from `SEED_U4_1x2Sigma` with the sector solve, at 400k (the cheapest size
  that converged refine).
  - Pass: `SECTOR_CHECK` passes every iteration, the winner is stable, and dHyb falls below the
    1e-3 floor the current 400k run sits on.
  - Only then return to the determinant ladder (`SOLVER_CAMPAIGN.md` criterion C) and the fit
    comparison (criterion E).

---

## 7. Decisions this plan does not make

- **Degenerate winners and the GF.** If the winner's orbit size is > 1 (K = (π,0)/(0,π), or band
  swap at odd N), the single-state G is not symmetric.
  - DMFT's symmetry projection (`grep "Symmetry projection" output_*.out`) averages G over the
    group. For a degeneracy generated by that group, this **is** the equal-weight ensemble over the
    orbit, so it is the correct T → 0⁺ limit and no new code is needed.
  - For a spin multiplet (S ≥ 1) it is not: ↑/↓ averaging is not the average over m. That choice
    remains open, as in parity plan §6.
  - Large discarded weight in the projection then means "degenerate ground state", not "broken
    solver". `check_dmft_health.py` should report it that way once `orbit_size` is available.
- **Which N.** The sector solve makes E(N) comparisons valid; it doesn't choose N. Re-run the charge
  sector search at each U after V7.
- **Intentional symmetry breaking** (orbital order, nematic, AFM in the cluster): turn the sector
  solve and bath projection off. That is a separate set of runs.

---

## 8. What to do with the runs currently on the queue (as of 2026-10-03)

- `RUN_U4_Irrep_N18{,_400k,_750k}` and `RUN_U2_Irrep` fail `SECTOR_CHECK` every iteration (→ §9.3.1: not
  true for `RUN_U2_Irrep` It_2), so they
  are converging restricted states. Keep them only as the "before" reference for V7. All five
  current jobs, including the one below, run on `dcgp_qos_lprod`.
- `EXP_U5_GFall_NROT3` (N = 24) answers a question this plan makes moot. Candidate for cancellation.

Whether to cancel is the user's call; nothing in this plan requires it.

---

## 9. Review (2026-10-03, later the same day)

A second reading of §0–§8 against the code and the archives. **Verdict:** adopt this plan as the
2×2 amendment, with the corrections in §9.3. Markers `(→ §9.x)` in the text above point here. The
original text is left as written.

### 9.1 Verified

- **The closed-shell reseed (§2.2).** `hf_determinant_byocc` (`sd_operations.hpp:74-88`) fills the
  same `min(Nα, Nβ)` orbitals for α and β, and every macro iteration after the first restarts from
  `dets = {hf_det}` (`impurity_solver.cpp`, macro loop of `SolveImpurityASCI_rot`). It is the main
  trap mechanism at NROTS ≥ 1, and the parity plan did not name it.
- **Natural-orbital blocks (§2.2.3).** `rotate_hamiltonian_ordm_imp_bath` (`rdms.hpp:120ff`) runs
  `gesvd` on the whole impurity block and the whole bath block. It does not split by band or K.
- **The refine throw (§4)** is at `asci/refine.hpp`, after the refinement loop.
- **`SECTOR_CHECK` values (§1)**, read from the archives:
  - `RUN_U4_Irrep_N18_400k` It_1: dE_add = −6.617e-2, dE_rem = +3.632e-2;
  - `RUN_U4_Irrep_N18_400k` It_2: dE_add = −8.721e-2, dE_rem = +4.291e-2;
  - `RUN_U2_Irrep` It_1: dE_add = −7.037e-2, dE_rem = −3.155e-1.

### 9.2 Agreed, including where it corrects the parity plan

- **NROTS > 0 does not need to be refused.** A state with definite (parity, K) has an exactly
  block-diagonal 1-RDM, so per-block natural orbitals (§3.2) keep the labels. The parity plan's §3
  ban was too strict.
- **SDP.** The parity plan's band inference (§2.2 there) fails on every SDP bath. It falls back with
  a printed reason, so "silently" overstates it, but the outcome is the same: the fix would never
  apply. Canonicalizing within degenerate poles (§3.3) is the right remedy.
- **Cost.** A µ search in the winner's sector with a re-screen at the converged µ (§3.4) is
  sensible. Land the refine 2-cycle fix (§4) first.
- **Degenerate winners and the GF (§7).** Averaging G over a group, for one state of an orbit
  generated by that group, equals the equal-weight ensemble over the orbit. Averaging over ↑/↓ is
  not an average over the spin projection m.

### 9.3 Corrections and additions

**9.3.1 U = 2 (§1, §6 V4, §8).**
- *Overreach.* "Local maximum in N, therefore trapped, not a wrong N" assumes E(N) is convex in N.
  The `SECTOR_CHECK` doc comment (`impurity_solver.hpp:95-104`) itself allows either reading.
  Small Hubbard plaquettes are exactly where convexity can fail: negative pair-binding energy is
  the classic 2×2 result. The It_1 numbers show that this state is not the grand-canonical ground
  state at its µ. They don't say why.
- *Outdated.* `RUN_U2_Irrep` It_2 finished at 15:52, 8 minutes after this document was written.
  It has the same sector (10,9), NROTS = 3 and 1.5M determinants, and it reports
  `SECTOR_CHECK dE_add = +2.946e-01 dE_rem = +3.820e-02 OK`. "Fails every iteration" (§8) is no
  longer true for U = 2. A positive check proves nothing (the same doc comment says so), and
  §2.2.3 predicts a different sector can be reached by chance. But V4's premise has to be
  re-checked on It_2 before it is used.

**9.3.2 A missing symmetry: the diagonal mirror σ_d (equivalently C4).**
- *Checked on `RUN_U4_Irrep_N18/It_2`.*
  - The impurity hopping block has exactly 8 site automorphisms (D4).
  - The bath pole weight matrices Γ = VᵀV commute with the four translations to 0.
  - They commute with the four σ_d/C4-type elements only to **2.2e-7**: fit noise, not exact.
- *Which symmetries are new.* On the 2×2 the axis mirrors coincide with the identity or with
  T_x/T_y. The only point-group symmetry beyond K is σ_d, with C4 = σ_d · (an axis mirror).
- *The closed-shell reseed fixes σ_d as well.* A determinant with every orbital doubly occupied or
  empty is σ_d- and C4-even. The list in §0 and §2.2 should read: even band parity, K = (0,0),
  S = 0, **and σ_d/C4-even.**
- *This is not hypothetical.* In the single-band plaquette the half-filled and two-hole ground
  states belong to different C4 channels (the d-wave pair-binding result). At quarter filling with
  one electron per site, a C4-odd ground state is a real candidate.
- *It cannot be a fourth loop label.* σ_d swaps the (π,0) and (0,π) orbitals, so K and σ_d cannot
  both be diagonal with real orbitals. Rule:
  - **Labels diagonal in the determinant basis** (band parity, K): loop over them explicitly.
  - **Symmetries that permute determinants** (σ_d/C4, total S): leave them free through the seed,
    and measure them on the winner.
- *Seed requirement* (adds to `sector_seed_byocc`, §3.2): in the K = (0,0) and (π,π) sectors, the
  seed must not be σ_d-invariant. For example, put the electrons in the E pair asymmetrically:
  two in (π,0) and none in (0,π) keeps total K but breaks σ_d.
- *Diagnostic.* Report ⟨σ_d⟩ of the winner in K = (0,0) and (π,π). In K = (π,0)/(0,π), σ_d maps
  the state into the partner sector, so it has no expectation value there. This needs the E-block
  natural orbitals chosen as σ_d images of each other (see 9.3.5). Then σ_d acts as a signed
  orbital permutation, and ⟨σ_d⟩ can be computed like `Parity_Test/site_parity.py` does for the
  1×2.
- *Bath.* If σ_d is to be treated as a symmetry, `canonicalize_bath` (§3.3) must project onto
  the full D4 commutant, not only translations and band swap. At 2.2e-7 the breaking is
  irrelevant for trapping, consistent with §3.3's last bullet, but it would show up in
  `LABEL_LEAK`-type checks for σ_d.

**9.3.3 Parallel sector solves are the first cost lever** (§3.4, §5).
- Sector solves are independent. Production jobs use 4 cores of a 112-core node: six sectors ×
  4 cores fit on one node, at about one solve's wall time instead of six.
- MACIS already has MPI hooks (`MACIS_MPI_CODE`), so a split communicator, one group per sector,
  followed by a min-reduction is the natural form.
- Screening still saves core-hours. Do both, but parallelism is the bigger and simpler win.

**9.3.4 Screening at NROTS = 1 needs its own validation** (§3.4 step 1).
- §3.2 cites NROTS = 1 as failing stationarity with 48 % band asymmetry.
- The "within 6 mHa of larger runs" figure comes from the 10-01 scan, which §1 says was trapped.
  It therefore says nothing about screening error in the untrapped sectors.
- V3 should compare the screening ordering with NROTS = 3 at the same determinant count, as well
  as 100k against 400k.

**9.3.5 Keep `SYMMETRIZE_DETS`** (§3.2, "not needed").
- It addresses a different failure: band asymmetry from truncation, not sector trapping.
- Per-block natural orbitals make it compatible with NROTS > 0. Choose the band-B natural orbitals
  as the band-swap images of the band-A ones. Then band swap is a signed orbital permutation in
  the rotated basis, and the `SYM_PERM` closure applies.
- The same construction for σ_d inside E gives the diagnostic of 9.3.2.

**9.3.6 Guard `canonicalize_bath` against rank growth** (§3.3).
- Projecting a pole's Γ onto the commutant can raise its rank, which would emit more bath orbitals
  than were fitted and change Nb and the N bookkeeping.
- This cannot happen when each degenerate group is closed under the group. It holds here: group
  sizes 2, 4, 2, 2, 4, 2 on `RUN_U4_Irrep_N18/It_2`, with band partners grouped together.
- It can happen when partner poles are only approximately degenerate and land in different
  groups. Assert that groups are closed under the group and that no group's rank grows. Refuse
  rather than resize.

**9.3.7 A momentum-basis impurity is not free when NROTS = 0** (§3.1).
- On-site Kanamori in momentum orbitals has about N_sites² more two-body terms. At large U the
  wavefunction is less compact than in the site basis.
- With NROTS ≥ 1 the natural orbitals are delocalized anyway, so the 2×2 loses nothing.
- The 1×2 at high U (NROTS = 0, site basis) should stay as it is. There K did not separate the
  competing states (⟨R⟩ = +1 for both, `Parity_Test/site_parity.py`), and seeds that are not
  site-swap invariant suffice.

### 9.4 Resulting changes to §5 and §6

- **Step 2:** choose paired natural orbitals for band swap and for σ_d inside E (9.3.5). Report
  `LABEL_LEAK` for band and K only.
- **Step 3:** `sector_seed_byocc` must also break σ_d in the K = (0,0) and (π,π) sectors (9.3.2).
- **Step 4:** run the sector solves in parallel, with a split communicator (9.3.3). Write ⟨σ_d⟩ and
  total ⟨S²⟩ of the winner to `symmetry_sectors.dat`.
- **V1:** add the D4-commutant residual (expected about 2e-7 here) and the rank-growth assertion.
- **V3:**
  - screening ordering at NROTS = 1 vs NROTS = 3 (9.3.4);
  - ⟨σ_d⟩ of the winner in K = (0,0)/(π,π);
  - wall time with parallel sectors.
- **V4:** re-establish the premise on It_2 (9.3.1) before using It_1 as the test case.
