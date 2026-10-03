# Independent assessment of the parity/2×2 sector proposals

> Incorporated into `symmetry-sector-solve.md` (2026-10-03); nothing here is superseded.

2026-10-03. Reviewed `parity-sector-solve.md` and `parity-sector-solve-2x2-context.md`,
the live `MACIS_claude` source at `aa26eaf` (clean working tree), the current DMFT
source, the DMFT skills in `~/.claude/skills`, and relevant fork/solver-health notes.
The DMFT working tree already has user changes; these were read, not modified.

**Verdict:** explicit comparison of symmetry sectors is justified, and symmetry-adapted
natural orbitals are a good direction. Neither proposal is ready to implement unchanged.
In particular, the spin argument does not describe this ASCI implementation, the U=0
failure occurs *within* the proposed parity/momentum sector, and SDP symmetrization is
a change of Hamiltonian rather than just a choice of bath basis.

The experiments are in `parity-sector-assessment-checks.py`; numerical output is in
`parity-sector-assessment-results.json`. They use small dense matrices and archived
one-body Hamiltonians/RDMs. No MACIS calculation, build, job submission, or production
input modification was performed. The high-U energy table in the original proposal
was not independently reproduced here.

## 1. What the source actually establishes

The important distinction is between a symmetry represented by a label on each
determinant, a symmetry represented by a linear combination of determinants, and a
disconnected component of the Hamiltonian graph.

* In a band-pure basis, parity is a determinant label. Hamiltonian-connected selection
  from one parity cannot discover the other, absent explicit insertion of other-sector
  determinants. More determinants or a tighter residual tolerance cannot repair this.
* Momentum has the same property **after** transforming to momentum-pure orbitals and
  checking the full Hamiltonian. A site-basis determinant generally contains multiple
  momenta. A closed-shell state has K=0 on this cluster only if the occupied spatial
  subspace is translation-invariant; “closed shell” alone is insufficient.
* `SolveImpurityASCI_rot` really does restart from `hf_det` at every macro iteration
  (`src/macis/impurity_solver.cpp:615–623`). `hf_determinant_byocc`
  (`include/macis/sd_operations.hpp:74`) uses identical orbital orderings for both
  spins, hence a closed shell for equal spin populations. The proposed replacement
  must preserve the requested sector at **every** restart.
* Full-space CAS/ED is also vulnerable: `SolveImpurityED` clears the coefficients,
  `compute_casci_rdms` calls `selected_ci_diag`, and the default one-root path starts
  from the minimum-diagonal determinant. Including all determinants in memory does
  not give a one-vector eigensolver overlap with disconnected blocks.

A three-dimensional numerical example makes the last point exact: a scalar block
with energy 0 and a disconnected block `[[1,-2],[-2,1]]`. The minimum-diagonal guess
has energy 0 and **zero residual**, while the actual ground energy is −1.

There is a further source detail missing from both plans:

```cpp
// include/macis/asci/iteration.hpp:64
std::vector<double> X_local;  // Precludes guess reuse
```

Every growth/refinement step discards the incoming Davidson vector, and
`serial_selected_ci_diag` / its MPI counterpart then call `diagonal_guess`
(`include/macis/solvers/davidson.hpp:65`). Thus an initially open-shell seed does
not guarantee an open-shell Davidson start in subsequent iterations. Likewise,
appending determinants to a warm-start space is not necessarily ineffective for
the reason asserted in §2.4a: the old pure-spin vector is not reused there.

## 2. The U=0 benchmark proves a broader search problem

I independently read `SOLVER_TESTS/U_ladder/U_0.0/{FCIDUMP.dat,input.in,active_ordm.dat}`.
The archived input is **Nα=Nβ=14, total N=28**. Several notes call this “N=14”; that
notation must not be mixed with the total-N convention in the N=18/19 campaign.

| Quantity | Result (Ha) |
|---|---:|
| Exact one-body Aufbau energy, 2 Σᵢ₌₁¹⁴ εᵢ | −16.791309684621 |
| Energy from archived RDM, Tr(hγ) | −15.368084807868 |
| Error | +1.423224876753 |
| Exact minimum in **even parity, K=(0,0)** | −16.791309684621 |

The archived γ is idempotent to `2.80e-9` in `γ²−2γ`. Reconstructing its occupied
spatial subspace gives many-body expectations of band parity, Tx and Ty all +1
to about `2e-10`. **Both the bad state and the exact ground state are in even
parity, K=(0,0).** This failure is not evidence that the wrong total K was selected.

At U=0 the eight `(band,K)` channel occupations are separately conserved. The
archived spin-summed counts are approximately `[4,4,4,4,4,4,2,2]`; the erroneous
filled level is at 0.75645186 instead of 0.04483943. Moving both spins costs
`2 × 0.71161244`, reproducing the error. Changing a pair's channel can preserve
both band parity and total K.

For the reference calculation I used dynamic programming over **all** one-body
levels, with state `(Nα,Nβ,parity,K)`, rather than a guessed near-Fermi window.
It supplies all eight sector minima cheaply and exactly for the supplied quadratic
Hamiltonian. This is a useful independent oracle for the proposed seed routine.

The companion plan acknowledges the extra U=0 conservation laws, but its fixed
occupation window and forced maximum-unpairing rule do not solve them. An excited
determinant's natural occupations are already 0/2: occupation ordering can preserve
its wrong channel allocation indefinitely. Forcing open shells can also exclude
the exact closed-shell ground determinant when H is diagonal. Use energy-based
channel allocation at U=0, and multiple competitive channel/reference configurations
at finite U; do not make maximum unpairing an unconditional constraint.

## 3. Spin needs checks, but the proposed reasoning is too strong

`[H,S²]=0` does not by itself imply that determinant-based, preconditioned Davidson
or selected CI preserves S. Three mechanisms matter:

1. ASCI discards and regenerates the Davidson guess on each iteration, as above.
2. A selected determinant projector generally does not commute with S².
3. The Jacobi preconditioner `diag(H)−E` generally does not commute with S² either,
   even for a spin-complete space.

The numerical three-spin SU(2)-invariant example in the script has
`||[H,S²]|| ≈ 2.36e-16` but `||[diag(H),S²]|| ≈ 1.0198`. Thus the proposal's
spin-trapping claim is not a general Krylov invariance argument for this solver.
It remains entirely plausible for particular states to get stuck, or for selection
to favor the wrong spin; the reported high-U spin results are valuable evidence.

I would retain total-S² measurement and comparisons between Sz sectors, but remove
the categorical ban on spin-complete seed spaces. Spin completeness is normally a
way to control spin contamination; the issue is whether the selected roots and
search retain the competing states. See the primary-method paper
[Applencourt, Gasperich and Scemama, spin-adapted selected CI](https://arxiv.org/abs/1812.06902).

Other corrections:

* No singly occupied **impurity** orbitals in a chosen seed does not imply the full
  sector contains only S=0: bath electrons and other configurations can carry spin.
* An Sz=1 comparison can detect a missed higher-spin state, but agreement of two
  approximate energies is not proof that neither missed another state. Measure S²
  and compare additional Sz values if higher total spin is plausible. For odd N the
  corresponding first comparison is Sz=1/2 against 3/2.
* Applying S− preserves parity and K. It repairs the matching sector; it cannot
  produce the even-sector solution from an odd-sector higher-spin state. The proposed
  validation with parity enumeration disabled therefore exercises a different case
  from a fixed-parity spin repair.

## 4. Momentum works for the checked baths; parity does not for the raw SDP baths

For each degenerate bath pole I formed `Γ=VᵀV` and transformed it with the fixed
per-band 2×2 Fourier matrix. Off-diagonal entries between distinct K labels test
translation symmetry independently of band mixing.

| Archived input | Maximum cross-K residue | Maximum cross-band residue |
|---|---:|---:|
| `RUN_U4_Irrep_N18/It_2` | 2.78e-17 | 0 |
| `RUN_U0.5_SDP_GFall/It_1` | 0 | 3.8850e-4 |
| `TEST_D_SDP_b157/It_3` | 5.11e-15 | 1.1653e-2 |

This settles the previously unverified bath-translation check for these three
inputs. It does not establish that every future bath has the same symmetry.

An orthogonal rotation W within a degenerate bath pole preserves
`(WV)ᵀ(WV)=VᵀV`. Therefore **the nonzero cross-band residues above cannot be
removed by a bath rotation alone**. The SDP one-body orbitals are strongly mixed
partly because of gauge freedom, but their entire symmetry defect is not a gauge
artifact.

On 200 imaginary frequencies from 1e-3 to 100, projecting the residues to be
diagonal in `(band,K)` and equal between bands changes Δ by:

| Input | max absolute entry error | max relative Frobenius error over frequency |
|---|---:|---:|
| Irrep U=4 | 3.56e-17 | 8.51e-17 |
| SDP U=0.5 | 4.25e-4 | 6.79e-4 (0.068%) |
| SDP TEST_D | 3.63e-2 | 0.1169 (11.7%) |

These are explicitly defined frequency-grid metrics, not the same normalization
as the plan's quoted residue ratios. In particular, a residue error does not give
the same relative error in Δ; denominators and cancellations matter.

The canonicalizer should have distinct modes for exact basis adaptation and
explicit symmetry projection. The latter needs a recorded changed Hamiltonian,
the changed bath propagated into G0/self-energy evaluation and restart files, and
a frequency-dependent round-trip check. It should not silently continue as if it
had only rotated orbitals. Enforcing the intended symmetry in the fit would make
this distinction cleaner.

Two additional limitations:

* Projection can increase rank: the one-orbital residue `[[1,1],[1,1]]` has rank 1;
  zeroing its cross-band entries gives rank 2. The proposed refactorization can
  therefore change bath size. None of the three checked pole groups suffers a rank
  increase, but the general implementation must handle it.
* Dropping dark, zero-coupling bath orbitals preserves Δ, **not** the fixed-N many-body
  spectrum. Dark orbitals still carry electrons, energy and quantum numbers. Preserve
  them during validation, or account explicitly for their occupations and offsets.

Finally, translation symmetry does not imply C4 symmetry. The Irrep U=4 poles have
K=(π,0)/(0,π) diagonal-weight differences up to `4.31e-7`, exceeding the proposed
`1e-8` symmetry tolerance. This is small fit noise, but the six-versus-eight-sector
reduction is not exact at that tolerance. Verify C4 and band exchange separately;
retain all representatives unless the accepted Hamiltonian really has those symmetries.

## 5. What SECTOR_CHECK does and does not prove

The source documentation is more careful than the companion plan
(`include/macis/impurity_solver.hpp:92–129`). A negative Ritz difference proves
there exists a neighboring-charge trial state below the returned ASCI energy
(within numerical accuracy). It does **not** distinguish a wrong charge N from an
excited/truncated state within N. The difference is relative to the returned
`E_ASCI(N)`, not necessarily the exact `E_GS(N)`.

Consequently:

* Persistence with increasing determinant budgets is compelling evidence against
  ordinary slow convergence, but not a mathematical exclusion of truncation error
  or selection stagnation.
* Both neighbors lying lower does not, without a separate convexity argument,
  prove an excited state at fixed N. An exact charge-energy sequence can skip a
  charge sector. The U=2 acceptance criterion should allow changing N.
* A positive check cannot certify completeness of parity, momentum or spin search.
* The claim that a μ change of ~0.2 can alter adjacent-N energy differences by at
  most ~0.2 is not justified here. `set_impurity_diagonal` changes **only impurity**
  levels (`src/macis/doping/fix_mu.cpp:24`); bath levels are held fixed in that root
  search. For physical μ, `dE/dμ=−<N_imp>`, so the slope of an adjacent-N gap is
  the difference of impurity occupations, not identically ±1. If the fitted bath
  also changed, a simple μ correction cannot compare the two Hamiltonians at all.

I ran the existing read-only health check on `RUN_U4_Irrep_N18_400k`. It reported
the ~71 Ha reference penalty and oscillating dHyb. Its current run-level output
also says `symmetrize_solver_output=False`. It is not a substitute for explicitly
checking symmetry-sector energies.

## 6. Screening and μ search: useful heuristics, insufficient certification

I support screening for cost control, **after** all-sector validation. A sector's
100k energy is a variational upper bound, not a lower bound: a poor preliminary
solution can lie 30 mHa above the screening winner and later fall 100 mHa below it.
The observed agreement of different budgets in the old restricted search does not
bound the errors of previously unexplored sectors. NROTS=1 screening can introduce
another sector-dependent error when final solves require NROTS=3.

Keep `SCREEN_MARGIN` as a measured accuracy/cost policy with explicit uncertainty,
not a correctness guarantee. Initially compare every sector at two budgets and
with enough rotations. Preserve multiple candidate references/roots within each
sector so that a preliminary winner cannot delete a later winner during growth.

Solving μ in one branch and restarting once after a final sector switch may leave
the returned state on the wrong branch or at the wrong filling. Repeat until the
chosen branch and filling are mutually consistent, or report an unresolved crossing
or density jump. At a genuine jump there may be no pure-ground-state root at the
target density. Do not label such a result converged merely because one retry ran.

## 7. Integration details the wrapper must address

The wrapper location is sensible, but saving only `dets,C,occs,orb_rot,E` is not
enough once rotations are allowed:

* `HamiltonianGenerator` holds spans into `p.T_active`, `p.Td_active`, `p.V_active`.
  The rotation routines modify those arrays in place. Start each sector from an
  identical base Hamiltonian and restore a basis-consistent winner. Otherwise the
  next solve resets `orb_rot` to identity on already-rotated integrals.
* `SolveImpurityASCI_rot` writes `active_ordm.dat` and `rot_matrix.dat` itself
  (`impurity_solver.cpp:736,747`). Without isolated outputs/final republication,
  those files describe the last competitor rather than the winner, even if the
  wrapper correctly restores p. Archive per-sector diagnostics and publish the
  winner's files once.
* There is a small pre-existing closure inconsistency: cold-guess group closure at
  line 597 is immediately discarded by `if(iorb>0 or !have_guess)` at line 617.
  Later `symmetric_orbit_select` closes candidates, so this does not by itself prove
  a wrong energy, but the assertion that the initial cold guess remains closed is
  false. The sector changes should fix that lifecycle explicitly.
* `GROW_WITH_ROT` needs the same treatment or an explicit incompatibility check.
  Also the cheap μ path calls `SolveImpurityCheapASCI`, bypassing this wrapper
  (`fix_mu.cpp:250–265`); route it deliberately or disallow it in validated mode.
* Blockwise NO rotations preserve `(band,K)` labels, but do not automatically
  preserve the **permutation representation** used by `SYMMETRIZE_DETS`: unrelated
  rotations in symmetry-related blocks can turn a permutation into a dense unitary.
  Keep that feature disabled until rotations are chosen covariantly or symmetry
  operators are transformed correctly.
* For more than two bands, one `PARITY_ORBS` list is insufficient. Supply a full
  orbital-label mapping, account for frozen occupations if applicable, and enumerate
  only feasible parity vectors. Do not discard a still-valid momentum symmetry
  just because band parity failed validation, or vice versa.

## 8. Refine failure handling and GF ensembles

Cycle detection and retaining the lowest valid state are worthwhile. However,
returning a best-so-far state after `MAX_REFINE_ITER` needs a distinct nonconverged
status. Dropping a failed competing sector does not establish that the remaining
winner is the ground state. Report an incomplete search and retain restart data;
do not silently certify the survivor. Energy recurrence alone is weaker than a
verified recurrence of determinant spaces. A union of the two cycling spaces with
a proper lowest-root solve is another small-case recovery to test.

The spin-ensemble discussion also needs correction. For an SU(2)-invariant H,
the spin-summed single-particle resolvent is a spin scalar. Within one exact spin
multiplet, `(G↑+G↓)/2` is independent of m. For integer S at m=0, spin inversion
also gives `G↑=G↓`; either channel already equals the multiplet-averaged channel.
This does not apply to tensor observables such as `<Sz1 Sz2>` or the full RDM.

I verified this in a 16-state, two-orbital atom with ferromagnetic exchange:

* triplet energies: −2.25 for m=−1,0,+1;
* m=0 spin-up GF versus the equal-multiplet spin-up GF: max difference `5.55e-17`;
* `<Sz1 Sz2>`: −1/4 in m=0, +1/12 in the equal ensemble.

Thus the proposal's statement that choosing the triplet ensemble necessarily
changes the single-particle GF is incorrect under those assumptions. For odd-N
or polarized representatives, explicitly computing both spin channels and averaging
is needed; the current GF implementation uses the configured `IS_UP_COMP` list,
not an automatic spin average. Truncated GF spaces can themselves violate these
identities, so they remain useful numerical checks.

Spatial group averaging of G likewise gives the group-averaged density matrix
when the group is an exact symmetry of H. It is the full zero-temperature Gibbs
ensemble only if all ground-state degeneracies have been included, not just one
known orbit. Moreover, current `Solver.py:1352–1362` can reject a large projection
defect, and projection is optional. “No new code is needed” is too strong for the
actual current flow. A known orbit size is evidence to interpret a defect, not
permission to classify every large defect as legitimate degeneracy.

## 9. Recommended sequence

1. Fix result bookkeeping and introduce explicit convergence/coverage status;
   make cycle recovery preserve a valid state without claiming convergence.
2. Separate exact bath adaptation from optional physical symmetry projection.
   Verify parity, translations, band exchange and C4 independently on the actual
   FCIDUMP, and retain dark bath modes during initial validation.
3. Implement sector-preserving NOs and constrained seed construction at every
   macro iteration. Use the exact U=0 dynamic-programming oracle; test closed-shell
   as well as open-shell winners, and multiple channel configurations per sector.
4. At fixed Hamiltonian and fixed N, compare all relevant parity/K sectors with
   multiple starts or roots. Measure total S² and perform Sz consistency checks.
   Verify both the original high-U case and the U=0 failure before optimizing cost.
5. Repeat the charge-N scans on the same Hamiltonians. Only then calibrate screening,
   add branch-consistent μ search, and assess a restarted DMFT loop.

The central correction is to make the search cover competing states, not just to
give one improved starting determinant to a succession of one-root searches.
