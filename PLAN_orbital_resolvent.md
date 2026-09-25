# Plan: Generalized diagonal-operator resolvents (orbital isospin, staggered spin)

> **STATUS: IMPLEMENTED** — written 2026-09-16, implemented 2026-09-18. Extends the existing `Sz`
> resolvent machinery to arbitrary *diagonal* one-body impurity operators, so that the orbital
> (isospin) screening scale can be measured with the same Lanczos continued-fraction path already
> used for `Sz`. All line anchors below were verified against MACIS fork branch `feature/spin_dep`,
> HEAD `e2b5265`, at planning time; see the diff for the as-built line numbers.
>
> §3.1-3.3 are implemented as specified: `RunResolventDiagonal` / `weighted_imp_value` /
> `DiagChannel` / the `make_*_weights` builders in `dynamical_properties.hpp`; `RunResolventSz`
> is kept as an unchanged-signature thin wrapper. `evaluate_resolvent_diagonal` in
> `impurity_solver.hpp` adds the §6 identity-rotation guard; `evaluate_resolvent_sz` deliberately
> does **not** go through that guard (Sz_imp is rotation-invariant, so gating it would have been a
> regression on the production rotated-ASCI path). `GF.TZ_RESOLVENT` / `GF.STAG_SZ_RESOLVENT` are
> wired up in `main/run_asci_impsolv_dop.cxx`. §5.1 unit tests are in `tests/dynamical_properties.cxx`.
> §5.2 physics-level checks (T^3 vs T^8 agreement at `delta_CFS = 0`, sum rule against `tauz_tauz`,
> `omega_orb` vs `omega_sf` trends) are runtime DMFT-loop checks, not unit tests, and are left for a
> real run.
>
> Scope is deliberately restricted to the **unrotated** solver paths (`CAS`/ED, `ASCI_cheap`, or
> `ASCI` with `nrots = 0`). See §6 for why, and §9 for what the rotated path would additionally
> require.

---

## 1. Motivation

### 1.1 What we can currently measure

`RunResolventSz` (`include/macis/gf/dynamical_properties.hpp:196`) evaluates

```
R_S(z) = <psi_0| Sz_imp  1/(z - (H - E_0))  Sz_imp |psi_0>
```

Its Lehmann representation, with `W_n = |<n| Sz_imp |psi_0>|^2` and `omega_n = E_n - E_0`, is
`R_S(z) = sum_n W_n / (z - omega_n)`. Two derived quantities have proven useful:

* `omega_sf`, the position of the maximum of `S(omega) = -Im R_S(omega + i eta)/pi`, read as the
  energy scale of spin fluctuations;
* the **frozen-moment fraction** `W_0 / <Sz^2>`, where `W_0 = sum_{omega_n = 0} W_n` is the weight
  carried by the degenerate ground multiplet. Because
  `C(t) = <Sz(t) Sz(0)> = sum_n W_n exp(-omega_n t)` has every `omega_n > 0` term decay,
  `lim_{t->inf} C(t) = W_0`. This is exactly the long-time plateau that defines the spin-freezing
  order parameter of Werner, Gull, Troyer and Millis [1], measured spectrally rather than in
  imaginary time.

### 1.2 What is missing, and why it matters

In a Hund's metal the spin and orbital degrees of freedom are screened at **parametrically
different** scales, `T_K^orb >> T_K^spin` — the coherence/incoherence crossover of Haule and
Kotliar [4], made quantitative for the three-band model by the NRG study of spin-orbital separation
of Stadler *et al.* [5]. Between the two scales the orbital degrees of freedom are already screened
while the spin is not: this is precisely the regime of frozen local moments coexisting with a
metallic (if incoherent) charge response, and it is the microscopic content of the "Janus-faced"
role of `J` described by de' Medici, Mravlje and Georges [2,3] — at fillings away from half,
`J` *raises* the critical `U` while simultaneously suppressing the coherence scale.

Measuring `omega_sf` alone cannot distinguish "the spin scale is small" from "all scales are small."
The discriminating observable is the **ratio** `omega_orb / omega_sf`, which requires the orbital
partner of `R_S`.

This is sharp for the three-band, two-electron case this fork is mostly used for. For Kanamori
interactions with `N = 2` electrons in `M = 3` degenerate orbitals, the atomic ground multiplet is
`S = 1` combined with an orbital triplet (two electrons occupying two of three orbitals): a 9-fold
degenerate `3 x 3` manifold. Spin freezing is the statement that the *spin* factor of that multiplet
survives to `t -> inf` while the *orbital* factor does not. The two resolvents measure the two
factors separately:

| operator | probes | expected behaviour near the transition |
|---|---|---|
| `Sz_imp` | spin factor of the atomic multiplet | `W_0/<Sz^2> -> 1`, `omega_sf -> 0` |
| `T^3`, `T^8` | orbital factor of the same multiplet | `W_0^orb/<T^2>` stays near 0 well past the point where the spin fraction saturates |

If the orbital frozen fraction rose together with the spin one, the picture would be a conventional
Mott localization of the whole multiplet, not spin-orbital separation. The measurement is therefore
a genuine test, not a confirmation exercise.

### 1.3 Why this is cheap

The orbital Cartan generators are **diagonal in the occupation-number determinant basis**, exactly
like `Sz_imp`. For three orbitals, with `n_b = n_{b,up} + n_{b,dn}` the occupation of band `b`:

```
T^3 = (1/2)      (n_1 - n_2)
T^8 = (1/(2*sqrt(3))) (n_1 + n_2 - 2 n_3)
```

so applying them to `|psi_0>` is a rescaling of determinant coefficients — precisely what
`apply_diagonal_operator` (`dynamical_properties.hpp:115`) already does. No new Lanczos code, no
new Hamiltonian build, no determinant-space enlargement.

### 1.4 Tracelessness is not optional

The generators must satisfy `sum_b w_b = 0`. Using the raw occupations `n_b` instead mixes the
orbital channel with the impurity **charge** channel. The total impurity charge is a slow,
near-conserved quantity (it changes only through impurity-bath hybridization), so its spectral
weight piles up at low frequency and the resulting `W_0` fraction would trend to 1 for reasons that
have nothing to do with orbital freezing. A traceless generator is orthogonal to the charge channel
by construction and isolates orbital *polarization*. The existing static `tauz` correlator
(`src/macis/comp_observables.cpp:307`) already follows this convention with `w = (+1/2, -1/2)`; the
separate `compute_charge_charge_correlations` covers the charge channel.

---

## 2. Design: a weight-vector interface

Rather than hard-coding `tau`, generalize the operator to a **diagonal one-body impurity operator
specified by a per-orbital weight vector** `w` of length `n_imp`, in one of two channels:

```
channel = CHARGE : O = sum_i w_i ( n_{i,up} + n_{i,dn} )
channel = SPIN   : O = sum_i w_i ( n_{i,up} - n_{i,dn} ) / 2
```

This single interface covers every case of interest, including the existing one:

| observable | channel | weights `w_i` (impurity orbital index `i`) |
|---|---|---|
| `Sz_imp` (current behaviour) | SPIN | `w_i = 1` for all `i` |
| orbital `T^3`, 2 bands | CHARGE | `+1/2` on band 0, `-1/2` on band 1 |
| orbital `T^3`, 3 bands | CHARGE | `+1/2`, `-1/2`, `0` by band |
| orbital `T^8`, 3 bands | CHARGE | `(1,1,-2)/(2*sqrt(3))` by band |
| staggered spin, 2 sites | SPIN | `w_i = (-1)^{site(i)}` |
| site-resolved spin | SPIN | `w_i = 1` on one site, `0` elsewhere |

The generalization costs nothing over hard-coding `tau` and answers the multi-site question (§7) for
free.

### 2.1 Orbital index layout

`n_imp = nbands * nsites`, and the impurity orbital index is **band-major, site-minor**:

```
i = site + nsites * band        band = i / nsites        site = i % nsites
```

Verified in two independent places: `src/macis/doping/fix_mu.cpp:35-42` (`i / nsites` selects the
crystal-field level) and `src/macis/comp_observables.cpp:317-322`
(`int i = site_i + n_sites_ * band_i`). The weight-builder helpers must use this convention.

### 2.2 Bit-ordering gotcha

`decompose_det` (`include/macis/observables/impurity_rdm.hpp:42-45`) packs impurity occupations
**in reverse**:

```
if(alpha[n_imp - 1 - p]) out.imp_up |= (1ULL << p);
```

so impurity orbital `i` lives at bit `n_imp - 1 - i` of `imp_up` / `imp_dn`. `sz_imp_value`
(`dynamical_properties.hpp:151`) never noticed, because it only popcounts. Every operator in this
plan is orbital-resolved and *will* notice: getting it backwards silently flips `T^3 -> -T^3` (a sign
error invisible in `|R(z)|`, and invisible in the sum rule too, since both are quadratic in `O`).
The unit test in §5.1 must pin the orbital-to-bit map explicitly.

---

## 3. Implementation steps

### 3.1 `include/macis/gf/dynamical_properties.hpp`

**(a) Refactor `RunResolventSz` into a generic driver.** Extract the body into

```cpp
template <size_t nbits, typename index_t = int32_t>
std::vector<std::complex<double>> RunResolventDiagonal(
    const Eigen::VectorXd &wfn0, HamiltonianGenerator<nbits> &Hgen,
    const std::vector<std::bitset<nbits>> &base_dets, ScalarFn scalar_fn,
    double E0, const std::vector<std::complex<double>> &ws,
    const GFSettings &settings);
```

keeping the two existing guards verbatim: the `2 * n_imp > 64` packing check
(`dynamical_properties.hpp:203`) and the zero-norm early return
(`dynamical_properties.hpp:215-218`), which matters more here than for `Sz` — a traceless operator
annihilates an orbitally unpolarized determinant far more often than `Sz` annihilates a determinant.

Reduce `RunResolventSz` to a thin wrapper over it. **The existing signature and behaviour must not
change**; `tests/dynamical_properties.cxx:143,234` and
`include/macis/impurity_solver.hpp:291` both call it and must keep passing untouched.

**(b) Add the diagonal weight evaluator.**

```cpp
enum class DiagChannel { Charge, Spin };

template <size_t nbits>
inline double weighted_imp_value(const std::bitset<nbits> &det,
                                 const std::vector<double> &w,
                                 DiagChannel ch, size_t n_imp, size_t n_active);
```

Decomposes via `decompose_det`, walks impurity orbitals `i = 0 .. n_imp-1`, reads bit
`n_imp - 1 - i` of `imp_up`/`imp_dn`, and accumulates `w[i] * (n_up +/- n_dn)` with the `1/2`
prefactor in the Spin channel. `sz_imp_value` becomes the special case
`w = 1, ch = Spin` and should be re-expressed through it so there is one code path.

**(c) Add weight builders,** each validating its own preconditions:

```cpp
std::vector<double> make_orbital_cartan_weights(size_t nbands, size_t nsites, int which);
std::vector<double> make_staggered_spin_weights(size_t nbands, size_t nsites);
std::vector<double> make_uniform_spin_weights(size_t nbands, size_t nsites);
```

`make_orbital_cartan_weights` throws for `nbands == 1` (no traceless generator exists in a
one-dimensional orbital space — see §7.1), accepts `which = 3` for `nbands >= 2` and `which = 8`
only for `nbands == 3`. All builders assert `sum_i w_i == 0` for the orbital case before returning.

### 3.2 `include/macis/impurity_solver.hpp`

Add `evaluate_resolvent_diagonal(EASCI, p, ham_gen, gf_settings, w, channel, label)`, modelled
directly on `evaluate_resolvent_sz` (`impurity_solver.hpp:266-304`). Reuse verbatim:

* the frequency-grid construction (`impurity_solver.hpp:275-285`), including the Matsubara branch —
  the `chi(0)` and `omega_eff` analysis needs `imag_freq = true` runs;
* the reference-energy shift `E0 = EASCI - (p.E_core + p.E_inactive)` (`impurity_solver.hpp:289`);
* the output format, writing `<label>_resolvent.dat` with columns `Re(w) Im(w) Re(R) Im(R)`.

Rewrite `evaluate_resolvent_sz` as a call into it with uniform spin weights, preserving the
`Sz_resolvent.dat` filename so existing post-processing scripts keep working.

**Add the rotation guard here** (§6): throw unless the impurity block of `p.orb_rot` is the identity
within `1e-10`. Model the message on the existing `evaluate_GF` unitarity check
(`impurity_solver.hpp:209-221`), and name the configuration fix explicitly — set
`CI.EXPANSION = CAS` or `ASCI.NROTS = 0`.

### 3.3 `main/run_asci_impsolv_dop.cxx`

Add keywords next to `GF.SZ_RESOLVENT` (line 520) and calls next to line 585:

```
GF.TZ_RESOLVENT     bool          # run T^3 (and T^8 when nbands == 3)
GF.STAG_SZ_RESOLVENT bool         # run staggered Sz; requires nsites == 2
```

`nsites` is already in scope at line 135. Gate the whole block on the same
`if(testGF || sz_resolvent || ...)` condition at line 522.

---

## 4. Derived quantities (post-processing, no new C++)

Everything already computed for `Sz` carries over unchanged, because it depends only on the Lehmann
structure and not on which operator produced it:

* **Sum rule** `integral S_O(omega) d omega = <O^2>` — validates the run and supplies the
  normalization for the frozen fraction.
* **`omega_orb`**: position of the maximum of `-Im R(omega + i eta)/pi`. Subject to the same
  `eta = 0.1 eV` resolution floor and the same discrete-grid peak-picking staircase as `omega_sf`.
* **Orbital frozen fraction** `W_0^orb / <O^2>`.
* **`omega_eff`** from the Matsubara `chi(0)`, same definition.

The headline plot is `omega_orb / omega_sf` versus `U` at fixed `J/U`.

---

## 5. Verification

### 5.1 Unit tests (`tests/dynamical_properties.cxx`)

The file already has the scaffolding: `make_det` (line 24) builds determinants from explicit
alpha/beta occupation lists, and `dense_hamiltonian` (line 35) gives the exact Lehmann reference.

1. **Orbital-to-bit map.** Construct determinants with a single impurity orbital occupied, one per
   orbital index, and assert `weighted_imp_value` returns `w[i]` for each — this is the test that
   catches the §2.2 reversal.
2. **`sz_imp_value` regression.** Reproduce the three existing hand-computed values
   (`tests/dynamical_properties.cxx:68-75`) through the new generic path.
3. **Tracelessness.** Assert the builders sum to zero and that the builder throws for
   `nbands == 1`.
4. **Exact Lehmann cross-check** for `T^3`, mirroring the existing `RunResolventSz` test
   (line 93) with the orbital weights substituted.

### 5.2 Physics-level checks

* **`T^3` vs `T^8` agreement.** For three *degenerate* orbitals the two Cartan generators must give
  the same spectral function up to numerical noise. A free and stringent correctness check —
  disagreement means either an orbital-polarized (symmetry-broken) solution or an index bug. Note
  this check is only valid at `delta_CFS = 0`; a nonzero crystal field splits band 2 from bands 0,1
  (`fix_mu.cpp:59-61`) and the two generators then legitimately differ.
* **Sum rule against the static observable.** `<(T^3)^2>` from the resolvent must match the
  `nbands == 2` static `tauz_tauz` output (`comp_observables.cpp:307`). This cross-validates two
  independently written code paths and is the reason to run the 2-band case first.
* **Consistency with `Sz`.** At `J = 0` and small `U` both `omega_sf` and `omega_orb` should be of
  order the bandwidth and comparable; their separation should open up as `J/U` grows.

---

## 6. Why unrotated only

`sz_imp_value`'s docstring (`dynamical_properties.hpp:136-138`) argues that `Sz_imp` is invariant
under the block-diagonal, spin-conserving natural-orbital rotations used in `SolveImpurityASCI_rot`,
because those rotations only redistribute occupation among impurity orbitals and leave the per-spin
impurity electron counts unchanged. **That argument does not extend to any orbital-resolved
operator.** `T^3` distinguishes orbital 0 from orbital 1 — which is exactly the distinction the
natural-orbital rotation scrambles. Evaluating it on rotated determinants yields a rotated-basis
isospin with no physical meaning.

`evaluate_GF` handles the analogous problem by checking that the impurity block of `orb_rot` is
unitary and back-rotating with `rotMat * G * rotMat^dagger` (`impurity_solver.hpp:198-234`). That
works because `G` carries two free orbital indices. It does **not** work here: in
`R = <psi| O (z - H + E_0)^{-1} O |psi>` the operator is contracted on both sides, so there is no
surviving index to rotate back through. `O` has to be correct *before* it is applied.

The production path `main/run_asci_impsolv_dop.cxx:413` uses `SolveImpurityASCI_rot`, so the guard
in §3.2 is a real constraint, not a formality. For the three-band single-site problem the unrotated
CAS/ED path is expected to be tractable at the `U` values of interest, which is why this plan takes
that route.

**Optional later extension (not in scope):** if `orb_rot`'s impurity block turns out to be a *signed
permutation* rather than a general rotation, the weight vector can simply be permuted to match and
the rotated path becomes usable at no further cost. Worth checking numerically before assuming the
expensive route below is needed.

---

## 7. Compatibility

### 7.1 `nbands = 2`

Fully supported. `T^3` with weights `(+1/2, -1/2)` per band reproduces the convention of the
existing static `tauz` correlator, which is what makes the §5.2 sum-rule cross-check possible. `T^8`
does not exist and the builder throws for `which = 8`.

### 7.2 `nbands = 3`

Fully supported, and the primary target. Both `T^3` and `T^8` are available, and their agreement is
the §5.2 correctness check. Note that the static `compute_tz_tz_correlations` explicitly refuses
three bands (`comp_observables.cpp:313-317`) because its `+/-1`-eigenvalue `tau_z` only exists for
a two-band manifold — the weight-vector formulation has no such restriction, so **this plan also
supplies the three-band static orbital correlator that the codebase currently lacks**, as the
zeroth moment of the resolvent.

### 7.3 `nbands = 1`

The spin channel works (it is the existing `Sz` behaviour). There is no traceless orbital generator
in a one-dimensional orbital space, so `make_orbital_cartan_weights` throws. This is correct
behaviour, not a limitation: with one band there is no orbital degree of freedom to screen.

### 7.4 `nsites = 2` — supported, with a caveat that matters

The index layout `i = site + nsites * band` (§2.1) is already respected by the weight builders, so
the orbital generators work unchanged for a two-site cluster: band weights are applied uniformly
across sites, giving the total cluster orbital polarization.

Two points deserve emphasis.

**(a) The existing `Sz` resolvent measures only the uniform channel.** `sz_imp_value` sums over
*all* impurity orbitals, so it computes `Sz(site 0) + Sz(site 1)` — the `q = 0` response. The
staggered combination `Sz(site 0) - Sz(site 1)`, i.e. the `q = pi` response where inter-site
antiferromagnetic correlations and singlet formation live, is **currently invisible**. This is a
limitation of the code as it stands today, not one introduced here; the weight-vector design fixes
it for free via `make_staggered_spin_weights`.

**(b) The uniform frozen fraction is misleading for a cluster.** If two sites lock into an
inter-site singlet, the total `Sz` has no elastic weight at all — the singlet is non-degenerate, so
`W_0 -> 0` in the uniform channel — even though each site still carries a large local moment. Read
naively, the uniform-channel frozen fraction would report "no frozen moments" for the most strongly
correlated state in the problem. **For `nsites > 1`, the uniform and staggered channels must both be
run, and the frozen fraction interpreted from the pair.** This is worth stating in whatever
post-processing script consumes the output.

**(c) Cost.** `nsites = 2` doubles the impurity orbital count: `nbands = 3, nsites = 2` gives
`n_imp = 6` impurity orbitals plus bath, where a full CAS is likely out of reach. The `ASCI_cheap`
path or `ASCI` with `NROTS = 0` both satisfy the §6 guard and remain available. The
`2 * n_imp <= 64` packing constraint inherited from `decompose_det` allows `n_imp <= 32` and is not
a practical limit here.

### 7.5 Summary

| configuration | orbital `T^3` | orbital `T^8` | uniform spin | staggered spin |
|---|---|---|---|---|
| `nbands=1`, `nsites=1` | throws (no orbital dof) | throws | yes (existing) | n/a |
| `nbands=2`, `nsites=1` | yes | throws | yes | n/a |
| `nbands=3`, `nsites=1` | yes | yes | yes | n/a |
| `nbands=2`, `nsites=2` | yes | throws | yes | yes |
| `nbands=3`, `nsites=2` | yes | yes | yes | yes |

Every cell marked "yes" requires the §6 unrotated condition.

---

## 8. Effort estimate

| step | scope |
|---|---|
| §3.1 refactor + weight evaluator + builders | ~90 lines, one header |
| §3.2 solver entry point + rotation guard | ~50 lines |
| §3.3 driver keywords | ~10 lines |
| §5.1 unit tests | ~120 lines, mostly adapted from existing cases |

No changes to the Lanczos, the Hamiltonian build, the CSR layer, or MPI. Half a day of work; the
cross-checks in §5.2 are the part worth spending real time on.

---

## 9. Out of scope

* **Off-diagonal orbital generators** (`T^1`, `T^2`, ...). These are genuine one-body operators
  mixing orbitals, so `O|psi>` leaves `base_dets` and would need an `apply_one_body_operator` with
  fermionic sign handling plus a determinant-space enlargement analogous to the `gf_dets`
  construction in the Green's-function path. The Cartan generators suffice to define an orbital
  screening scale, for the same reason `Sz` suffices for the spin one.
* **The rotated-ASCI path** beyond the signed-permutation shortcut noted in §6.
* **Total spin `S^2`** rather than `Sz`. Not diagonal; same obstacle as off-diagonal orbital
  generators.
* Any change to the DMFT self-consistency loop. This is a measurement on a converged solution.

---

## References

1. P. Werner, E. Gull, M. Troyer, A. J. Millis, *Spin freezing transition and non-Fermi-liquid
   self-energy in a three-orbital model*, Phys. Rev. Lett. **101**, 166405 (2008).
2. L. de' Medici, J. Mravlje, A. Georges, *Janus-faced influence of Hund's rule coupling in strongly
   correlated materials*, Phys. Rev. Lett. **107**, 256401 (2011).
3. A. Georges, L. de' Medici, J. Mravlje, *Strong correlations from Hund's coupling*, Annu. Rev.
   Condens. Matter Phys. **4**, 137 (2013).
4. K. Haule, G. Kotliar, *Coherence-incoherence crossover in the normal state of iron oxypnictides
   and importance of Hund's rule coupling*, New J. Phys. **11**, 025021 (2009).
5. K. M. Stadler, Z. P. Yin, J. von Delft, G. Kotliar, A. Weichselbaum, *Dynamical mean-field theory
   plus numerical renormalization-group study of spin-orbital separation in a three-band Hund
   metal*, Phys. Rev. Lett. **115**, 136401 (2015).

*(Reference details transcribed from memory — verify volume/page before citing in a paper.)*

---

## Critical files

| file | role |
|---|---|
| `include/macis/gf/dynamical_properties.hpp` | `RunResolventSz:196`, `apply_diagonal_operator:115`, `sz_imp_value:151` — all changes in §3.1 |
| `include/macis/impurity_solver.hpp` | `evaluate_resolvent_sz:266`, rotation guard pattern at `:209-221` |
| `include/macis/observables/impurity_rdm.hpp` | `decompose_det:35` — reversed bit packing at `:42-45` |
| `src/macis/comp_observables.cpp` | `compute_tz_tz_correlations:307` — static cross-check, band/site index convention at `:317-322` |
| `src/macis/doping/fix_mu.cpp` | `set_impurity_diagonal:24` — independent confirmation of band-major layout at `:35-42` |
| `main/run_asci_impsolv_dop.cxx` | `GF.SZ_RESOLVENT:520`, call site `:585`, `nsites` in scope at `:135` |
| `tests/dynamical_properties.cxx` | existing Lehmann cross-check scaffolding at `:24`, `:35`, `:93` |
