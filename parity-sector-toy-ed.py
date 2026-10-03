#!/usr/bin/env python3
"""Exact per-parity-sector energies of the two-band toy of tests/charge_sectors.cxx.

Same model as make_model() there, but with the cross-band bath coupling set to 0
(band-diagonal bath), so the band parities (-1)^{N_A}, (-1)^{N_B} are conserved.
Prints the full ground state, the minimum of each parity sector, the largest
Hamiltonian element between sectors (must be 0), and the sector of the legacy
ASCI seed (canonical_hf_determinant: first nalpha/nbeta raw indices).

Usage: parity-sector-toy-ed.py U J [eps_d] [nalpha nbeta]   (eps_d defaults to -U/2)
Reference output (U=6, J=0.8, eps_d=-3):
  (3,3): (e,e) -8.512442367  (o,o) -8.551448456  <- GS; legacy seed in (e,e)
  (3,2): (e,o) -8.700141181  (o,e) -8.865079189  <- GS; legacy seed in (o,e)
"""
import itertools
import sys

import numpy as np

n, nimp = 6, 2
BANDS = ([0, 2, 4], [1, 3, 5])  # impurity orbital b, then its two bath levels


def model(U, J, ed, cross=0.0):
    T = np.zeros((n, n))
    V = np.zeros((n, n, n, n))  # chemist (pq|rs), as in the FCIDUMP
    eps = [-2.0, -0.7, 0.6, 1.9]
    for k in range(4):
        b, band = nimp + k, k % 2
        T[b, b] = eps[k]
        T[band, b] = T[b, band] = 0.5
        T[1 - band, b] = T[b, 1 - band] = cross
    for m in range(nimp):
        T[m, m] = ed
        V[m, m, m, m] = U
        for mp in range(nimp):
            if mp != m:
                V[m, m, mp, mp] = U - 2 * J
                V[m, mp, mp, m] = J
                V[m, mp, m, mp] = J
    return T, V


def apply(ops, d):
    """Apply a product of (is_creation, spin_orbital) right to left."""
    sgn = 1
    for cre, q in reversed(ops):
        occ = (d >> q) & 1
        if occ == cre:
            return None, 0
        if bin(d & ((1 << q) - 1)).count("1") % 2:
            sgn = -sgn
        d ^= 1 << q
    return d, sgn


def build(T, V, na, nb):
    dets = []
    for a in itertools.combinations(range(n), na):
        for b in itertools.combinations(range(n), nb):
            dets.append(sum(1 << 2 * p for p in a) | sum(1 << 2 * p + 1 for p in b))
    idx = {d: i for i, d in enumerate(dets)}
    terms = [(T[p, q], [(1, 2 * p + s), (0, 2 * q + s)])
             for p in range(n) for q in range(n) for s in range(2) if T[p, q]]
    for p, q, r, s_ in itertools.product(range(n), repeat=4):
        if V[p, q, r, s_]:
            for s in range(2):
                for t in range(2):
                    terms.append((0.5 * V[p, q, r, s_],
                                  [(1, 2 * p + s), (1, 2 * r + t), (0, 2 * s_ + t), (0, 2 * q + s)]))
    H = np.zeros((len(dets), len(dets)))
    for j, d in enumerate(dets):
        for c, ops in terms:
            d2, sg = apply(ops, d)
            if d2 is not None:
                H[idx[d2], j] += c * sg
    return dets, H


def key(d):
    """Band-parity vector (N_A mod 2, N_B mod 2), spins summed."""
    return tuple(sum(((d >> 2 * p) & 1) + ((d >> 2 * p + 1) & 1) for p in orbs) % 2
                 for orbs in BANDS)


def main():
    U, J = float(sys.argv[1]), float(sys.argv[2])
    ed = float(sys.argv[3]) if len(sys.argv) > 3 else -U / 2
    na, nb = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (3, 3)
    dets, H = build(*model(U, J, ed), na, nb)
    keys = [key(d) for d in dets]
    print(f"(nalpha, nbeta) = ({na}, {nb}), full GS = {np.linalg.eigvalsh(H)[0]:.9f}")
    for k in sorted(set(keys)):
        m = np.array([kk == k for kk in keys])
        print(f"  sector {k}: dim {m.sum():4d}  E_min = {np.linalg.eigvalsh(H[np.ix_(m, m)])[0]:.9f}"
              f"  max |H| to other sectors = {abs(H[np.ix_(m, ~m)]).max():.1e}")
    legacy = sum(1 << 2 * p for p in range(na)) | sum(1 << 2 * p + 1 for p in range(nb))
    print(f"  legacy seed sector: {key(legacy)}")


if __name__ == "__main__":
    main()
