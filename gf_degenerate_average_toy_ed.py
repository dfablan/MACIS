#!/usr/bin/env python3
"""Toy ED for gf_band_average_full_group.md: which band average of a single degenerate member's
Green's function reproduces the T -> 0+ ensemble GF of a J = 0 multiband impurity?

Model: 3 equivalent bands, Kanamori at J = 0 (= (U/2) N(N-1)), one bath level per band, coupled
band-diagonally and identically. H is U(6)-invariant. The bath level sits below the Fermi level, so
the N = 8 ground manifold is the 15-fold d^2 multiplet on top of a closed-shell bath (a single U(6)
irrep), which is the situation of .../Nb15/J_0/U_60.00 It_19.

Each flavour count (impurity + bath of one band and one spin) is conserved, so every member of the
manifold is labelled by the two flavours holding the extra electrons. The script compares, for the
up-spin impurity GF that MACIS computes (GF.ORBS_COMP = 0 1 2, IS_UP_COMP = TRUE):

  raw            : the GF of one member, as the solver returns it today for an (e,e,e) winner
  coset average  : 1/m sum_r G(g_r^-1 a, g_r^-1 b), one group element per image of the parity key
                   (band_orbit_average_gf, impurity_solver.hpp:256-278), every choice of
                   representatives
  full average   : 1/|S3| sum_g G(g^-1 a, g^-1 b) over all six band permutations

against the ensemble GF (equal mixture of all 15 members, both spins).

Usage: python gf_degenerate_average_toy_ed.py
"""

import itertools

import numpy as np

NB = 3                     # bands
NORB = 2 * NB              # spatial orbitals: impurity 0..2, bath 3..5 (bath 3+a couples to band a)
NMODE = 2 * NORB           # fermion modes: m = orbital + NORB * spin
U, MU, EPS, V = 8.0, 12.0, -1.5, 0.6
BETA, NW = 50.0, 200
IW = 1j * (2 * np.arange(NW) + 1) * np.pi / BETA


def mode(orb, spin):
    return orb + NORB * spin


def apply_cdag(m, state):
    """c^dag_m |state> as (sign, new_state), or None. Jordan-Wigner order: modes below m."""
    if state >> m & 1:
        return None
    sign = -1 if bin(state & ((1 << m) - 1)).count("1") % 2 else 1
    return sign, state | (1 << m)


def apply_c(m, state):
    if not state >> m & 1:
        return None
    sign = -1 if bin(state & ((1 << m) - 1)).count("1") % 2 else 1
    return sign, state & ~(1 << m)


def flavour_counts(state):
    """(impurity + bath) electrons per flavour (band, spin): conserved at J = 0."""
    return tuple(((state >> mode(a, s)) & 1) + ((state >> mode(NB + a, s)) & 1)
                 for s in range(2) for a in range(NB))


def hamiltonian(states):
    index = {st: i for i, st in enumerate(states)}
    H = np.zeros((len(states), len(states)))
    for i, st in enumerate(states):
        nimp = sum((st >> mode(a, s)) & 1 for a in range(NB) for s in range(2))
        nbath = sum((st >> mode(NB + a, s)) & 1 for a in range(NB) for s in range(2))
        H[i, i] += 0.5 * U * nimp * (nimp - 1) - MU * nimp + EPS * nbath
        for a in range(NB):
            for s in range(2):
                for src, dst in ((mode(NB + a, s), mode(a, s)), (mode(a, s), mode(NB + a, s))):
                    r1 = apply_c(src, st)
                    if r1 is None:
                        continue
                    r2 = apply_cdag(dst, r1[1])
                    if r2 is None:
                        continue
                    H[index[r2[1]], i] += V * r1[0] * r2[0]
    return H


def sector(pred):
    states = [st for st in range(1 << NMODE) if pred(st)]
    E, X = np.linalg.eigh(hamiltonian(states))
    return states, E, X


def popcount(st):
    return bin(st).count("1")


# Charge sectors around the ground state
sectors = {n: sector(lambda st, n=n: popcount(st) == n) for n in range(5, 12)}
emin = {n: sectors[n][1][0] for n in sectors}
N0 = min(emin, key=emin.get)
E0 = emin[N0]
deg = int(np.sum(sectors[N0][1] < E0 + 1e-9))
print(f"ground sector N = {N0}, E0 = {E0:.10f}, degeneracy = {deg}  (d^2 of SU(6): 15)")
print(f"gaps to N-1 / N+1: {emin[N0 - 1] - E0:.4f} / {emin[N0 + 1] - E0:.4f}")
assert deg == 15


def gf(psi, states, flav_mode):
    """G_m(iw) for the normalised state psi (amplitudes on `states`) at the impurity mode m."""
    out = np.zeros(NW, dtype=complex)
    for op, dn, sgn in ((apply_cdag, +1, +1), (apply_c, -1, -1)):
        tstates, tE, tX = sectors[N0 + dn]
        tindex = {st: i for i, st in enumerate(tstates)}
        vec = np.zeros(len(tstates))
        for amp, st in zip(psi, states):
            r = op(flav_mode, st)
            if r is not None:
                vec[tindex[r[1]]] += r[0] * amp
        w = (tX.T @ vec) ** 2
        out += np.sum(w[None, :] / (IW[:, None] - sgn * (tE[None, :] - E0)), axis=1)
    return out


# One member per flavour pair: the ground state of each flavour-count sector at E0
flavours = [(a, s) for s in range(2) for a in range(NB)]
members = {}
for pair in itertools.combinations(range(2 * NB), 2):
    target = tuple(2 if f in pair else 1 for f in range(2 * NB))
    states, E, X = sector(lambda st, t=target: popcount(st) == N0 and flavour_counts(st) == t)
    assert abs(E[0] - E0) < 1e-9, (pair, E[0] - E0)
    members[pair] = (states, X[:, 0])

G = {p: np.array([gf(psi, st, mode(a, s)) for (a, s) in flavours]) for p, (st, psi) in members.items()}
G_ens = np.mean([G[p] for p in G], axis=0)            # (6 flavours, NW)
g_ens = G_ens.mean(axis=0)
scale = np.abs(g_ens).max()
print(f"ensemble GF is flavour-diagonal-uniform: max_f |G_ens,f - g| / max|g| = "
      f"{np.abs(G_ens - g_ens).max() / scale:.2e}")

# S3 on bands; the parity key of a member and its images
group = list(itertools.permutations(range(NB)))       # g[a] = image band of a


def parity_key(pair):
    n = [2 * 1 + sum(1 for f in pair if flavours[f][0] == a) for a in range(NB)]  # bath 2 + extra
    return tuple(x % 2 for x in n)


def image_key(key, g):
    out = [0] * NB
    for a in range(NB):
        out[g[a]] = key[a]
    return tuple(out)


def band_average(Gup, elems):
    """1/m sum_r G(g_r^-1 a, g_r^-1 a) for the up-spin diagonal Gup[a]."""
    inv = [{g[a]: a for a in range(NB)} for g in elems]
    return np.array([np.mean([Gup[iv[a]] for iv in inv], axis=0) for a in range(NB)])


def rel(x):
    return np.abs(x).max() / scale


print()
print("member (extra electrons)   key        raw: max_a|G_a - g|   coset avg (all rep choices)   full S3 avg")
for pair in [(0, 3), (0, 4), (1, 5), (0, 1)]:
    Gup = G[pair][:NB]                               # up-spin impurity GF, bands 0..2
    key = parity_key(pair)
    images = {}
    for g in group:
        images.setdefault(image_key(key, g), []).append(g)
    coset_errs = [rel(band_average(Gup, list(choice)) - g_ens)
                  for choice in itertools.product(*images.values())]
    full = rel(band_average(Gup, group) - g_ens)
    label = " ".join(f"{'ud'[flavours[f][1]]}{flavours[f][0]}" for f in pair)
    keystr = "(" + ",".join("eo"[k] for k in key) + ")"
    n_exact = sum(e < 1e-10 for e in coset_errs)
    span = (f"{min(coset_errs):.2e} .. {max(coset_errs):.2e} ({n_exact}/{len(coset_errs)} exact)"
            if len(images) > 1 else "not averaged (m = 1)")
    print(f"{label:26s} {keystr:9s}  {rel(Gup - g_ens):18.2e}   {span:28s}  {full:.2e}")

print("\nSz = +1 member (u0 u1) is outside the Sz = 0 solve MACIS runs; listed for completeness.")
