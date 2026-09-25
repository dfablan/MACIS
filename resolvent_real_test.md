# Proposed real-calc tests: orbital-resolved resolvents in the single-site 2-band and 3-band symmetric models

Status: proposal. Scope: validate the `resolvent_testing` branch
(`gf/dynamical_properties.hpp`, `SPIN_ORB_MATRIX_RESOLVENT`) against the
`O(2)` and `SO(3)` selection rules on already-converged single-site DMFT
solutions, and produce the physics they were built for.

Sources read for this note:

- Repo plans: `PLAN_orbital_resolvent.md`, `PLAN_orbital_resolved_susceptibility.md`.
- Notes: `Orbital_Channels_and_Observable_Choice.md`, `Sz_Resolvent_Analysis.md`.
- Selection rules: `Selection_Rules_General_2band.tex`,
  `Selection_Rules_General_3band.tex`, `selection_rule.md`,
  `Orbital_Resolved_Resolvent_Plan_2Band_Degenerate.md`.

Target data:

- 2-band: `/g100/home/userexternal/dfloreza/scratch/dfloreza/Ulysses_move/multiband_singlesite/2bands/Doping/selected_calculations`
  (`J_0`, `J_0.1`, `J_0.2`, `J_0.25`, `J_0.3333`, `J_0.4`, `J_0.5`; each a `U` sweep).
- 3-band: `/g100/home/userexternal/dfloreza/scratch/dfloreza/Leonardo_move/3band/singlesite/Doping/selected_calculations`
  (`J_0`, `J_.05`, `J_.1`, `J_.2`, `J_.25`, `J_.3`, `J_.333`, `J_.35`, `J_.4`, `J_.5`; each a `U` sweep).

---

## 1. What the theory predicts

### 1.1 Two-band, single site — `SU(2)_spin × O(2)_orb`

The four longitudinal spin bilinears `S_μν` (and identically the charge
`N_μν`) transform as `2⊗2 = A_1 ⊕ A_2 ⊕ E_2`. With
`A = R_{11;11} = R_{22;22}`, `B = R_{11;22} = R_{22;11}`,
`C = R_{12;12} = R_{21;21}`, `D = R_{12;21} = R_{21;12}`, the resolvent in
the orbital-pair basis `(11,22,12,21)` is

```
R = [[A, B, 0, 0],
     [B, A, 0, 0],
     [0, 0, C, D],
     [0, 0, D, C]]
```

- **`O(2)` identity:** `A − B = C + D`.
- **Pair-hopping fingerprint:** `D ≠ 0` for full Kanamori; `D ≡ 0`
  identically for density–density-only interactions (no symmetry argument
  needed).
- Three independent functions `r_0, r_1, r_2` with
  `r_0 = A+B`, `r_1 = C−D`, `r_2 = A−B = C+D`.
- At `J = 0` the symmetry enlarges to `SU(4)`: `r_0 = r_1 = r_2 = q_1 = q_2`
  (`B = D = 0`, `A = C`). `q_0` does **not** join this collapse.

Det-diagonal representatives (weight-vector interface,
`O = Σ_i w_i (n_{i↑} ± n_{i↓})`):

| function | operator | channel | weights `w` |
|---|---|---|---|
| `r_0` | `S_z^1 + S_z^2` | SPIN | `(1, 1)` |
| `r_2` | `S_z^1 − S_z^2` | SPIN | `(+1, −1)` |
| `q_2` | `T^3 = ½(n_1 − n_2)` | CHARGE | `(+½, −½)` |
| `q_0` | `N_imp` (subtract `⟨N_imp⟩²/z`) | CHARGE | `(1, 1)` |

**Factor trap:** `make_orbital_cartan_weights` returns `(±½)`, correct for
CHARGE. The SPIN channel already carries its own `½`, so feeding that
vector to SPIN yields `½(S_z^1 − S_z^2)` and a resolvent smaller by `4`.

### 1.2 Three-band, single site — `SU(2)_spin × SO(3)_orb`

`3⊗3 = 1 ⊕ 3 ⊕ 5`, so in the scalar / vector / quadrupole basis

```
R_S = diag(r_0, r_1·I_3, r_2·I_5),   R_N = diag(q_0, q_1·I_3, q_2·I_5).
```

In the orbital-pair basis

```
r_0 = A + 2B,   r_1 = C − D,   r_2 = A − B = C + D,
```

with `A = R_{αα;αα}`, `B = R_{αα;ββ}`, `C = R_{αβ;αβ}`,
`D = R_{αβ;βα}` for `α ≠ β`. Structural consequences to test:

- 60 symmetry zeros (all diagonal↔interorbital cross elements, and the
  interorbital internal zeros).
- The two `E_g` copies are equal.
- `T^3` and `T^8` are members of the same `l = 2` quintet, so at
  `δ_CFS = 0`: `R_{T^3,T^3} = R_{T^8,T^8} = q_2/2`, `R_{T^3,T^8} = 0`.
- Total-spin contraction:
  `R_{S_z^imp,S_z^imp} = ¼[2k_{11} + k_{22} + √2(k_{12}+k_{21})]`.
- At `J = 0`, `SU(6)`: `r_0 = r_1 = r_2 = q_1 = q_2` (runnable subset
  `r_0 = r_2 = q_2`).

**Null control:** for a one-dimensional orbital ground manifold,
`W_0[T^3] = 0` exactly (and likewise `W_0[L^z] = W_0[δN] = 0`). A nonzero
value signals orbital degeneracy of the ground manifold, i.e. the
assumption under every selection rule in this family is broken.

---

## 2. Instrumentation available on the branch

Keywords live in the `[GF]` section of the impurity-solver `input.in`
(the driver reads them as `GF.<KEY>`).

| keyword | effect |
|---|---|
| `GF = TRUE` / `FALSE` | compute the full Green's function (usually `FALSE` for a measurement run) |
| `SZ_RESOLVENT = TRUE` | `r_0` (uniform SPIN), existing path |
| `TZ_RESOLVENT = TRUE` | `q_2` via `T^3`; adds `T^8` when `nbands == 3` |
| `SPIN_ORB_MATRIX_RESOLVENT = TRUE` | **new**: full `R_{μν;γδ}` (spin channel) in one band-Lanczos pass ⇒ `A, B, C, D` |
| `STAG_SZ_RESOLVENT = TRUE` | staggered `q = π` spin (requires `nsites == 2`; not used here) |
| `DELTA_RESOLVENT = TRUE` | connected `δO = O − ⟨O⟩`; writes `<label>_delta_resolvent.dat` |
| `USE_BANDLAN = ON` | band Lanczos (required for the matrix path) |
| `NLANITS` | total Krylov dimension; must be comfortably above the retained rank `r` |
| `ORB_DEFLATE_TOL` | Gram eigenvalue cutoff (default 1e-10; see review note — consider 1e-14) |
| `ORB_MIN_CAPTURE` | warn-only seed-capture floor (default 0.95) |
| `GF.NWS`, `GF.WMIN`, `GF.WMAX`, `GF.IMAG_FREQ`, `GF.ETA` | frequency grid |

Outputs:

- `<label>_resolvent.dat` — diagonal channels
  (`Re(w) Im(w) Re(R) Im(R)`).
- `<label>_orbital_resolvent.dat` — one row per `(iw, pair_k, pair_l)`:
  `Re(w) Im(w) μ ν γ δ Re(R) Im(R)`.
- `<label>_gram.dat` — the `M×M` Gram matrix, its spectrum, retained rank,
  and per-element capture fractions.

The matrix path is essential: the diagonal channels cannot see `B`, `C`,
`D`, which are exactly what the selection rules constrain.

---

## 3. Preconditions (verify before interpreting anything)

1. **Hamiltonian symmetry.**
   - Structure already confirmed:
     - 2-band (`locFCIDUMP.dat`, e.g. `J_0.25/U_26.00`): `ε₁ = ε₂ = −3.46`,
       `U = 26`, `U' = 13 = U − 2J` (`J = 6.5`); `1x1_TwoBands.latt` gives
       two identical bands.
     - 3-band (`locFCIDUMP.dat`, e.g. `J_.2/U_32.50`): `ε₁ = ε₂ = ε₃ = −19.80`,
       `U = 32.5`, `U' = 19.5 = U − 2J`; `dmft.input` sets
       `point_group = 'infinity'`, `impose_orb_symmetry = .true.`,
       `bath_struct = 'Irrep'`.
   - Still to check in the **fitted** bath (`Old_fit_params.dat`):
     `Δ_{12} = 0` and `Δ_{11} = Δ_{22}` (2-band); `Δ_{μν} = Δ δ_{μν}`
     (3-band). A nematic/unconverged fit breaks the extra rotation and
     mixes `A_1`/`A_2`.
2. **Unrotated basis.** Orbital-resolved operators are meaningless in the
   natural-orbital basis. Production runs use `ASCI.NROTS = 2`; measurement
   runs must use **`ASCI.NROTS = 0`** (or `CI.EXPANSION = CAS`). `S_z`
   uniform is exempt; `T^3`, `T^8`, `S_z^1 − S_z^2`, and every matrix element
   are not.
3. **Ground-manifold character.** The selection rules require a
   one-dimensional *orbital* irrep of the complete impurity–bath ground
   manifold. The `mean(Occs)` in `dHyb.dat` is ≈ 0.25 for the 2-band run and
   the fillings (`DOP NELECTRONS = 0.5` for 2-band, `0.666` for 3-band) sit
   **away from the clean `n = 2` limits**, where the atomic ground manifold
   is orbitally degenerate (`E_1` for two bands; `(S=1,L=1)` for three).
   Therefore the **diagnostic run comes first** (T1 / S1), and the
   ground-state `M_S` and orbital character must be recorded with every
   output.

---

## 4. Measurement-run recipe

Re-run the impurity solver on the converged bath, no DMFT self-consistency.
Reuse `Old_fit_params.dat`; set the `[CI]`/`[ASCI]`/`[GF]` sections as below.

```ini
[CI]
GF = FALSE
FCIDUMP = <...>/It_N/ASCI/FCIDUMP.dat
NIMP = 2                # 3 for the three-band tree
NALPHA = <as converged>
NBETA  = <as converged>
NBANDS = 2              # 3 for the three-band tree
DOPING = TRUE
EXPANSION = ASCI

[ASCI]
NTDETS_MAX = <as converged>
MAX_REFINE_ITER = <as converged>
REFINE_ETOL = 1E-4
NROTS = 0               # <-- required: unrotated for orbital-resolved operators

[DOP]
NELECTRONS = 0.500000   # 0.666000 for the three-band tree
INIT_MU = <from the converged It_ folder>

[GF]
WRITE = TRUE
NORBS = 16              # 18 for the three-band tree
TRUNC_SIZE = 100000000
TOT_SD = 1
GFSEEDTHRES = 1E-3
ASTHRES = 1E-4
USE_BANDLAN = ON
NLANITS = 3000          # >= a few x retained rank r
ORB_DEFLATE_TOL = 1E-10
ORB_MIN_CAPTURE = 0.95
BETA = 157
NWS = 1000
WMIN = 0.0
WMAX = 40.0
IMAG_FREQ = TRUE
ETA = 0.1

# channels
SZ_RESOLVENT = TRUE
TZ_RESOLVENT = TRUE
SPIN_ORB_MATRIX_RESOLVENT = TRUE
DELTA_RESOLVENT = TRUE
```

Notes:

- Watch stdout for `FOUND A ZERO VECTOR AT POSITION`; after correct deflation
  it should not fire. If it does, the deflation tolerance is too loose.
- On a truncated ASCI space, read `_gram.dat` capture fractions before
  interpreting any element with `μ ≠ ν`; on CAS they are identically 1.
- `DELTA_RESOLVENT` is effectively mandatory for the charge trace `q_0`
  (elastic piece `⟨N_imp⟩²/z`) and useful elsewhere.

---

## 5. Two-band test menu

Path: `.../2bands/Doping/selected_calculations`.

| # | test | observable / computation | pass condition |
|---|---|---|---|
| T1 | Ground-manifold diagnostic | `W_0[T^3]` from the `T^3` run; ground-state degeneracy and orbital character | consistent with 0 (orbital singlet); otherwise **stop** and record |
| T2 | Sum rule / static cross-check | zeroth moment of `q_2` vs `tauz_tauz.dat`; `⟨(S_z^imp)²⟩ = ¼ m_0^{(0)}` | agreement to grid accuracy |
| T3 | **O(2) identity** | from matrix resolvent: `A − B` vs `C + D` | equal to ~1e-6 at each ω |
| T4 | Pair-hopping fingerprint | `D = R_{12;21}` | `D ≠ 0` for Kanamori; `≈0` for a density–density control |
| T5 | `J = 0` collapse | `SPIN(1,1) = SPIN(1,−1) = CHARGE(½,−½)` on `J_0` | three curves coincide (catches the ×4 trap) |
| T6 | Hund's-locking order parameter | `w_0^B / w_0^A = (w_0^{(0)} − w_0^{(2)}) / (w_0^{(0)} + w_0^{(2)})` | in `[0,1]`; trend with `U` at fixed `J/U` |
| T7 | Trace regression | trace direction of matrix resolvent vs `Sz_resolvent.dat` | ratio 4 (convention), else bug |
| T8 | Seed capture | `_gram.dat` capture fractions | 1 on CAS; reported on ASCI |

Recommended points: `J_0` (T5), `J_0.25` (T1–T4, T6, T8), at
`U ∈ {1, 18.66, 26, 32, 44}`.

---

## 6. Three-band test menu

Path: `.../3band/singlesite/Doping/selected_calculations`.

| # | test | observable / computation | pass condition |
|---|---|---|---|
| S1 | Ground-manifold diagnostic | `W_0[T^3]`, `W_0[T^8]` | consistent with 0 for a one-dim orbital `Γ_0` |
| S2 | `T^3 = T^8` | `q_2` from `T^3` vs `T^8` | equal at `δ_CFS = 0` |
| S3 | **SO(3) projectors** | from `A,B,C,D`: `r_0 = A+2B`, `r_1 = C−D`, `r_2 = A−B = C+D` | identity holds piecewise in ω |
| S4 | Multiplet structure | equal `A`-entries (3), equal `B` (6), equal `C` (6), equal `D` (6); 60 zeros | zeros at noise floor; degeneracies exact |
| S5 | `J = 0` SU(6) | `r_0 = r_1 = r_2 = q_1 = q_2` on `J_0` | spread ~1e-6 (subset `r_0 = r_2 = q_2` diagonal-runnable) |
| S6 | Total-spin curve | `¼[2k_{11}+k_{22}+√2(k_{12}+k_{21})]` | matches direct `SZ_RESOLVENT` |
| S7 | Static partners | zeroth moments of `q_0`, `q_2` vs `db_occs_matrix`, `tauz_tauz` | agreement |

Recommended points: `J_0` (S5), `J_.2` (S1–S4, S6, S7), at
`U ∈ {2, 9, 18, 32.5, 52}`.

---

## 7. Implementation gaps to close first

1. **`GF.SPIN_QUAD_RESOLVENT`** (SPIN × traceless, weights `(+1,−1)`): listed
   as "available, not wired". ~5 lines; completes the det-diagonal 2-band
   sector and gives an independent `r_2` to compare against `A − B`.
2. **Charge uniform `q_0`**: a `CHARGE(1,1)` run plus the `δN` subtraction
   (`R[δN] = R[N] − ⟨N⟩²/z`) in post-processing — needed for T2/S7 on the
   charge side.
3. **Charge matrix resolvent** (`GF.CHARGE_ORB_MATRIX_RESOLVENT`): needed
   only for the `q_1` block and the charge-side `C_c, D_c`; not required for
   the tests above.
4. These runs are the **first real validation** of
   `evaluate_resolvent_orbital_matrix`. To date that path is exercised only
   by the 17 unit tests in `tests/dynamical_properties.cxx`
   (`macis_test "Dynamical properties*"` → all pass, 1447 assertions).

---

## 8. Caveats

- **Filling is the main risk.** At these dopings the ground manifold may be
  orbitally degenerate, so a failure of T3/S3 is not automatically a bug.
  Run T1/S1 first; record `M_S` and orbital character.
- Both peak-ratio observables are resolution-limited by `η = 0.1 eV`; use the
  exact `W_0` weights for the frozen-sector physics.
- `q_2` measures the *quadrupolar* orbital scale; the angular-momentum scale
  `q_1` differs for `J ≠ 0` and requires the off-diagonal machinery.
- Verify `Δ_{12} = 0` in the *fit*, not only in the target bath.
- The selection rules concern the complete impurity–bath ground state, not
  the atomic multiplet content of the local density matrix.
