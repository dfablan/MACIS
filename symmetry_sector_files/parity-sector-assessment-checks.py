"""Small, read-only checks for the parity-sector proposal; no solver runs.

Run in classy-dmft-dev, with OPENBLAS_NUM_THREADS=1.
"""
from pathlib import Path
import io
import json
import tarfile
import numpy as np

ROOT = Path('/leonardo/home/userexternal/dfloreza/g100_move_Collura/2band/Nimp2x2/Doping/SDR/Nb16/J_0.2')


def read_files(path, names):
    found = {}
    for name in names:
        for p in (path / name, path / 'ASCI' / name):
            if p.exists():
                found[name] = p.read_text()
                break
    if len(found) < len(names) and (path / 'ASCI.tar.gz').exists():
        with tarfile.open(path / 'ASCI.tar.gz', 'r|gz') as tf:
            for m in tf:
                name = Path(m.name).name
                if name in names and name not in found and m.isfile():
                    found[name] = tf.extractfile(m).read().decode()
                if len(found) == len(names):
                    break
    return found


def one_body(text):
    rows = np.loadtxt(io.StringIO(text))
    n = int(rows[:, :4].max())
    h = np.zeros((n, n))
    for i, j, k, l, v in rows:
        if k == l == 0 and i > 0 and j > 0:
            h[int(i)-1, int(j)-1] = h[int(j)-1, int(i)-1] = v
    return h


F = np.array([[(-1)**((s//2)*(k//2)+(s%2)*(k%2))/2
               for k in range(4)] for s in range(4)])
Q = np.kron(np.eye(2), F)


def bath_check(rel):
    data = read_files(ROOT / rel, ['FCIDUMP.dat'])
    h = one_body(data['FCIDUMP.dat'])
    eps = np.diag(h)[8:]
    v = h[8:, :8]
    assert np.max(np.abs(h[8:, 8:] - np.diag(eps))) < 1e-12
    groups = []
    for j in np.argsort(eps):
        if not groups or abs(eps[j]-eps[groups[-1][0]]) >= 1e-10:
            groups.append([int(j)])
        else:
            groups[-1].append(int(j))
    poles = []
    errs = []
    for g in groups:
        residue = v[g].T @ v[g]
        rk = Q.T @ residue @ Q
        scale = np.max(np.abs(rk))
        band_cross = np.max(np.abs(residue[:4, 4:]))
        kmask = np.array([[i % 4 != j % 4 for j in range(8)] for i in range(8)])
        k_cross = np.max(np.abs(rk[kmask]))
        c4_difference = max(abs(rk[1,1]-rk[2,2]), abs(rk[5,5]-rk[6,6]))
        # Projection in this representation: diagonal in (band,K), equal bands.
        diagonal = np.diag(rk).reshape(2, 4).mean(axis=0)
        projected = Q @ np.diag(np.tile(diagonal, 2)) @ Q.T
        defect = np.max(np.abs(residue - projected))
        vals = np.linalg.eigvalsh(residue)
        pvals = np.linalg.eigvalsh(projected)
        poles.append(dict(eps=float(eps[g[0]]), multiplicity=len(g),
                          rank=int(np.sum(vals > 1e-10)),
                          projected_rank=int(np.sum(pvals > 1e-10)),
                          band_cross_abs=float(band_cross),
                          momentum_cross_abs=float(k_cross),
                          c4_weight_difference=float(c4_difference),
                          band_k_scale=float(scale),
                          projection_max_abs=float(defect)))
        errs.append((eps[g[0]], residue-projected, residue))
    # An explicit frequency-grid test, distinct from the residue metric.
    ws = np.geomspace(1e-3, 100, 200)
    max_delta = max_rel = 0.0
    for w in ws:
        delta = sum(d / (1j*w-e) for e, d, _ in errs)
        original = sum(r / (1j*w-e) for e, _, r in errs)
        max_delta = max(max_delta, float(np.max(np.abs(delta))))
        max_rel = max(max_rel, float(np.linalg.norm(delta)/np.linalg.norm(original)))
    return dict(path=rel, poles=poles, delta_max_abs=max_delta,
                delta_max_relative_frobenius=max_rel)


def noninteracting_check():
    rel = 'SOLVER_TESTS/U_ladder/U_0.0'
    data = read_files(ROOT / rel, ['FCIDUMP.dat', 'active_ordm.dat', 'input.in'])
    h = one_body(data['FCIDUMP.dat'])
    gamma = np.loadtxt(io.StringIO(data['active_ordm.dat']))
    ev, u = np.linalg.eigh(h)
    occ = np.diag(u.T @ gamma @ u)
    na = nb = 14  # archived input.in; total N is 28, not 14
    exact = float(2*sum(ev[:na]))
    actual = float(np.trace(h @ gamma))
    # Invariant subspaces in the ORIGINAL basis: infer bath action from residues.
    # If each pole residue is diagonal in band,K, normalized columns of VQ are
    # the corresponding bath orbital eigenvectors. Complete dark directions via SVD.
    eps = np.diag(h)[8:]
    vq = h[8:, :8] @ Q
    ub = np.zeros((len(eps), len(eps)))
    labels = list(range(8))
    jnew = 0
    for e in sorted(set(eps)):
        g = np.flatnonzero(abs(eps-e) < 1e-10)
        cols = []
        for label in range(8):
            col = vq[g, label]
            if np.linalg.norm(col) > 1e-10:
                col = col / np.linalg.norm(col)
                cols.append(col)
                ub[g, jnew] = col
                labels.append(label)
                jnew += 1
        # This frozen input is full-rank at each pole; refuse invented dark labels.
        assert len(cols) == len(g), (e, len(cols), len(g))
    assert np.max(np.abs(ub.T@ub-np.eye(len(eps)))) < 1e-6
    transform = np.zeros_like(h)
    transform[:8, :8] = Q
    transform[8:, 8:] = ub
    hh = transform.T@h@transform
    gg = transform.T@gamma@transform
    labels = np.array(labels)
    off = labels[:, None] != labels[None, :]
    # Solve each one-body channel, then dynamic programming over all spin orbitals.
    levels = []
    channel_counts = []
    for label in range(8):
        inds = np.flatnonzero(labels == label)
        ee = np.linalg.eigvalsh(hh[np.ix_(inds, inds)])
        levels.extend((float(e), label//4, label%4) for e in ee)
        channel_counts.append(float(np.trace(gg[np.ix_(inds, inds)])))
    dp = {(0, 0, 0, 0): 0.0}
    for spin in (0, 1):
        for en, band, k in levels:
            nxt = dp.copy()
            for (a, b, parity, momentum), energy in dp.items():
                aa, bb = a+(spin == 0), b+(spin == 1)
                if aa > na or bb > nb:
                    continue
                key = (aa, bb, parity ^ (band == 0), momentum ^ k)
                nxt[key] = min(nxt.get(key, float('inf')), energy+en)
            dp = nxt
    sectors = {f'p{p}_K{k}': dp[(na,nb,p,k)] for p in (0,1) for k in range(4)}
    # For an idempotent spin-summed gamma with spin equality, take its occupied
    # spatial subspace and compute many-body symmetry expectations (one determinant
    # per spin, hence square of the single-spin determinant).
    gval, gu = np.linalg.eigh(gg/2)
    occupied = gu[:, gval > .5]
    sym_exp = {}
    for name, signs in [('band_parity', np.where(labels//4 == 0, -1., 1.)),
                        ('Tx', (-1.)**((labels%4)//2)),
                        ('Ty', (-1.)**((labels%4)%2))]:
        sym_exp[name] = float(np.linalg.det(occupied.T@(signs[:,None]*occupied))**2)
    return dict(path=rel, nalpha=na, nbeta=nb, exact=exact, archived_rdm_energy=actual,
                error=actual-exact, channel_offdiag_h=float(np.max(abs(hh[off]))),
                channel_counts_spin_summed=channel_counts,
                idempotency_error=float(np.max(abs(gamma@gamma-2*gamma))),
                symmetry_expectations=sym_exp,
                exact_by_parity_momentum=sectors,
                one_body_eigenvalues=ev.tolist(), spin_summed_occupations=occ.tolist())


def toy_checks():
    # Two disconnected sectors: min diagonal lies in A, true minimum in B.
    h = np.array([[0., 0., 0.], [0., 1., -2.], [0., -2., 1.]])
    x = np.array([1.,0.,0.])
    # A symmetry projection can INCREASE the rank of a bath pole.
    v = np.array([[1., 1.]])
    r = v.T@v
    # An isolated closed shell is the true noninteracting ground state; forcing
    # maximum unpairing loses it in a diagonal one-body eigenbasis.
    return dict(disconnected_min_diag_energy=float(x@h@x),
                disconnected_residual=float(np.linalg.norm(h@x-(x@h@x)*x)),
                disconnected_true_energy=float(np.linalg.eigvalsh(h)[0]),
                pole_rank_before=int(np.linalg.matrix_rank(r)),
                pole_rank_after_band_projection=int(np.linalg.matrix_rank(np.diag(np.diag(r)))),
                two_level_closed_shell_energy=0., two_level_forced_open_shell_energy=1.)


def spin_checks():
    # Exact 16-state atom: U*double occupancy - J*S1.S2 - mu*N.
    annihilators = []
    for p in range(4):
        c = np.zeros((16, 16))
        for d in range(16):
            if (d >> p) & 1:
                c[d ^ (1 << p), d] = (-1)**((d & ((1 << p)-1)).bit_count())
        annihilators.append(c)
    n = [c.T@c for c in annihilators]
    sz = [(n[2*i]-n[2*i+1])/2 for i in range(2)]
    sp = [annihilators[2*i].T@annihilators[2*i+1] for i in range(2)]
    h = 4*(n[0]@n[1]+n[2]@n[3])-sum(n)
    h -= sz[0]@sz[1] + .5*(sp[0]@sp[1].T+sp[0].T@sp[1])
    triplets = []
    for a, b in [(2,0), (1,1), (0,2)]:
        inds = [d for d in range(16) if ((d&1)>0)+((d&4)>0) == a
                and ((d&2)>0)+((d&8)>0) == b]
        ev, u = np.linalg.eigh(h[np.ix_(inds, inds)])
        psi = np.zeros(16)
        psi[inds] = u[:, 0]
        triplets.append((float(ev[0]), psi))
    gf = np.zeros((3, 2, 3), dtype=complex)
    correlations = []
    for m, (e, psi) in enumerate(triplets):
        correlations.append(float(psi@sz[0]@sz[1]@psi))
        for s in range(2):
            a = annihilators[s].T@psi
            b = annihilators[s]@psi
            for iw, w in enumerate([.1, 1., 10.]):
                gf[m,s,iw] = a@np.linalg.solve((1j*w+e)*np.eye(16)-h,a)
                gf[m,s,iw] += b@np.linalg.solve((1j*w-e)*np.eye(16)+h,b)
    # Three distinguishable spins give a minimal SU(2)-invariant Hamiltonian
    # whose determinant/spin-product diagonal does not commute with total S^2.
    paulis = [np.array([[0,1],[1,0]])/2,
              np.array([[0,-1j],[1j,0]])/2, np.diag([1,-1])/2]
    spins = []
    for site in range(3):
        spins.append([np.kron(np.kron(s if site==0 else np.eye(2),
                                     s if site==1 else np.eye(2)),
                                     s if site==2 else np.eye(2)) for s in paulis])
    total = [sum(spins[i][a] for i in range(3)) for a in range(3)]
    s2 = sum(s@s for s in total)
    hh = sum(j*sum(spins[i][a]@spins[k][a] for a in range(3))
             for i,k,j in [(0,1,1.),(1,2,.4),(0,2,.2)])
    diag = np.diag(np.diag(hh))
    return dict(triplet_energies=[e for e,_ in triplets],
                m0_up_vs_multiplet_up_max_error=float(np.max(abs(gf[1,0]-gf[:,0].mean(axis=0)))),
                spin_average_max_variation=float(np.max(abs(gf.mean(axis=1)-gf.mean(axis=(0,1))))),
                sz1sz2_m_plus1_0_minus1=correlations,
                sz1sz2_ensemble=float(np.mean(correlations)),
                h_s2_commutator=float(np.linalg.norm(hh@s2-s2@hh)),
                diagonal_s2_commutator=float(np.linalg.norm(diag@s2-s2@diag)))


if __name__ == '__main__':
    results = {'baths': [bath_check(p) for p in [
        'RUN_U4_Irrep_N18/It_2', 'RUN_U0.5_SDP_GFall/It_1', 'TEST_D_SDP_b157/It_3']],
        'u0': noninteracting_check(), 'toy': toy_checks(), 'spin': spin_checks()}
    print(json.dumps(results, indent=2))
