#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <macis/asci/grow.hpp>
#include <macis/asci/parity_labels.hpp>
#include <macis/asci/refine.hpp>
#include <macis/gf/dynamical_properties.hpp>
#include <macis/gf/gf.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/hamiltonian_generator/sd_build.hpp>
#include <macis/util/cas.hpp>
#include <macis/util/detail/rdm_files.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/util/fock_matrices.hpp>
#include <macis/util/general_io.hpp>
#include <macis/util/memory.hpp>
#include <macis/util/moller_plesset.hpp>
#include <macis/util/mpi.hpp>
#include <macis/util/transform.hpp>
#include <macis/wavefunction_io.hpp>
#include <map>
#include <sparsexx/io/write_dist_mm.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>

using macis::NumActive;
using macis::NumCanonicalOccupied;
using macis::NumCanonicalVirtual;
using macis::NumElectron;
using macis::NumInactive;
using macis::NumOrbital;
using macis::NumVirtual;

enum class CIExpansion { CAS, ASCI, ASCI_cheap };

namespace macis {

/**
 * @brief Structure to hold the parameters of the impurity problem.
 */

template <size_t N>
struct impurity_params {
  size_t n_active;
  size_t nbeta;
  size_t nalpha;
  size_t n_inactive;
  size_t norb;
  size_t n_imp;
  size_t nbands;

  double nel_target;
  std::vector<double> orb_rot;

  double dstep;
  double abs_tol;
  size_t maxiter;
  size_t mu_cost_counter;
  bool print_doping;
  double init_shift;
  bool cheap_mode;
  double delta_CFS;
  bool spin_dep;

  CIExpansion ci_exp;
  std::string asci_wfn_fname;
  bool compute_asci_E0;
  double asci_E0;

  macis::MCSCFSettings mcscf_settings;
  macis::ASCISettings asci_settings;
  std::vector<double> occs;
  std::vector<double> C;
  std::vector<macis::wfn_t<N>> dets;

  double E_core;
  double E;
  double E_inactive;
  std::vector<double> T;
  std::vector<double> Td;
  std::vector<double> V;
  std::vector<double> F_inactive;
  std::vector<double> Fd_inactive;
  std::vector<double> T_active;
  std::vector<double> Td_active;
  std::vector<double> V_active;
  bool just_singles;

  // Band-parity sector solve (ASCI.PARITY_SOLVE,
  // parity-sector-solve-simple.md). Set by setup_parity_sectors; null = off.
  // When set, SolveImpurityED, SolveImpurityASCI and SolveImpurityASCI_rot
  // solve every band-parity sector separately and return the lowest.
  std::shared_ptr<const ParityLabels> parity_labels;
  double parity_etol = 1e-6;  // ASCI.PARITY_ETOL: near-tie report threshold
  std::vector<int>
      parity_only;  // ASCI.PARITY_ONLY: one 0/1 per band; empty = all
  // Outcome of the last parity solve, read by evaluate_GF's band average:
  // the winner's key and the sectors that tied with it within PARITY_ETOL
  // (PARITY_TIE)
  uint32_t parity_winner_key = 0;
  std::vector<uint32_t> parity_tied_keys;
};

/**
 * @brief Prints the charge-sector check that falls out of the GF calculation.
 *
 * The FCIDUMP carries -mu, so the solved Hamiltonian is H - mu*N and a GF pole
 * sits at E(N+1) - E(N) (particle) or E(N-1) - E(N) (hole). If (NALPHA, NBETA)
 * is the ground-state sector both are >= 0. The band Lanczos yields upper
 * bounds on them, so a NEGATIVE value proves an N+-1 state lies below the
 * ASCI state -- either the sector is wrong, or E(N) is off by more than |dE|
 * (ASCI truncation). A positive value proves nothing: the bound comes from a
 * truncated GF space and only reaches states connected to psi0 by one c/c+.
 *
 * Greppable line: "SECTOR_CHECK dE_add = <x> dE_rem = <y> [OK|FAIL]".
 */
template <size_t N>
void report_sector_check(double dE_add, double dE_rem,
                         const macis::impurity_params<N> &p) {
  const bool fail = dE_add < 0. or dE_rem < 0.;
  const auto flags = std::cout.flags();
  const auto prec = std::cout.precision();
  std::cout << std::scientific << std::setprecision(6) << std::showpos;
  std::cout << "GF SECTOR CHECK for (NALPHA, NBETA) = (" << std::noshowpos
            << p.nalpha << ", " << p.nbeta << std::showpos
            << "), upper bounds from the band-Lanczos Ritz values:"
            << std::endl;
  std::cout << "  E(N+1) - E(N) <= " << dE_add << "   (particle)" << std::endl;
  std::cout << "  E(N-1) - E(N) <= " << dE_rem << "   (hole)" << std::endl;
  std::cout << "SECTOR_CHECK dE_add = " << dE_add << " dE_rem = " << dE_rem
            << (fail ? " FAIL" : " OK") << std::noshowpos << std::endl;
  if(fail)
    std::cout << "WARNING: an N" << (dE_add < 0. ? "+1" : "-1")
              << " state lies below the ASCI ground state. Either (NALPHA, "
                 "NBETA) is not the ground-state sector of this FCIDUMP, or "
                 "E(N) is off by more than |dE| (ASCI truncation). Compare "
                 "sectors with explore_charge_sectors.py."
              << std::endl;
  if(std::isnan(dE_add) or std::isnan(dE_rem))
    std::cout
        << "  (NaN: no electron could be added/removed, or GF.USE_BANDLAN "
           "is off -- the bound comes from the band Lanczos only)"
        << std::endl;
  std::cout.flags(flags);
  std::cout.precision(prec);
}

/**
 * @brief Averages the impurity GF over the orbit of the solved state under the
 *        SYMMETRIZE_DETS orbital-permutation group, when the state's
 *        band-parity sector is not invariant under that group
 *        (GF.BAND_AVERAGE; symmetry-sector-solve.md, sec. 3.9).
 *
 * In a parity sector the determinant space is closed only under the stabilizer
 * of the sector's key (parity_stabilizer). At odd N with a band swap the keys
 * (e,o) and (o,e) are exchanged, the stabilizer is trivial, and the solved
 * state is ONE of two exactly degenerate partners, psi and U_g psi, with
 * U_g c_q U_g^+ = c_{g(q)}. Its GF is band-polarized; the ensemble GF at
 * T = 0 is the orbit average
 *
 *   G(a, b) = 1/m sum_r G_psi(g_r^-1 a, g_r^-1 b),   m = |orbit of the key|,
 *
 * over one group element g_r per distinct image key g_r.key (identity
 * included). Elements of the stabilizer are NOT averaged over: there the
 * state is already invariant, and averaging would hide a real symmetry
 * breaking of the solver. When the stabilizer is the whole group (m = 1,
 * e.g. every even-N (e,e) / (o,o) sector with a band swap) nothing is done.
 *
 * The GF must be in the ORIGINAL basis (call after the back-rotation): the
 * permutation acts on orbital labels. Every g_r must map GF.ORBS_COMP onto
 * itself with the spin kept; otherwise the average is skipped and reported.
 * H is invariant under the group by construction (prepare_det_symmetry
 * checks T, Td and V), so the partners are degenerate.
 *
 * Never silent when the returned GF may break a degeneracy:
 *  - PARITY_SOLVE on but no group (SYMMETRIZE_DETS off, e.g. any NROTS > 0
 *    run): warns if the parity solve reported a PARITY_TIE with the winner;
 *  - a tied sector outside the orbit of the winner's key: warned;
 *  - SYMMETRIZE_DETS on without PARITY_SOLVE (no labels): the state is
 *    checked against each group element through the sign-free overlap
 *    sum_d |C_d| |C_{g(d)}| (= 1 for a state invariant up to signs, ~0 for an
 *    orthogonal partner), and a broken symmetry is warned, not averaged.
 */
template <size_t N>
void band_orbit_average_gf(std::vector<std::vector<std::complex<double>>> &GF,
                           const macis::impurity_params<N> &p,
                           const macis::GFSettings &s) {
  const auto &stg = p.asci_settings;
  const bool have_group =
      stg.symmetrize_dets and stg.sym_group and !stg.sym_group->empty();
  if(p.dets.empty() or p.dets.size() != p.C.size()) return;

  // No parity labels: nothing to average by key. With a group, check that
  // the state is invariant under it, and warn otherwise.
  if(!p.parity_labels) {
    if(!have_group) return;
    std::unordered_map<macis::wfn_t<N>, size_t> index;
    index.reserve(p.dets.size());
    for(size_t i = 0; i < p.dets.size(); ++i) index.emplace(p.dets[i], i);
    double norm = 0.;
    for(auto c : p.C) norm += c * c;
    double min_ovl = 1.;
    for(const auto &g : *stg.sym_group) {
      double ovl = 0.;
      for(size_t i = 0; i < p.dets.size(); ++i) {
        const auto it = index.find(macis::permute_orbitals(p.dets[i], g));
        if(it != index.end()) ovl += std::abs(p.C[i] * p.C[it->second]);
      }
      min_ovl = std::min(min_ovl, ovl / norm);
    }
    if(min_ovl < 1. - 1e-6)
      std::cout << "WARNING: GF_BAND_AVERAGE none: the solved state is not "
                   "invariant under the SYMMETRIZE_DETS group (min_g sum_d "
                   "|C_d||C_g(d)| = "
                << min_ovl
                << "), so its GF breaks that symmetry. It is one of several "
                   "degenerate partners (e.g. a band-parity sector at odd N); "
                   "set ASCI.PARITY_SOLVE = TRUE to average the GF over them"
                << std::endl;
    return;
  }

  const auto &L = *p.parity_labels;
  const macis::ParityMasks<N> pm(L);
  // Parity key of the solved state: the same for every determinant of a
  // parity-sector solve
  const uint32_t key = pm.key(p.dets.front());
  const auto keystr = macis::parity_key_string(key, L.ngroups);
  // Ties reported by the parity solve that produced this state
  const std::vector<uint32_t> tied =
      p.parity_winner_key == key ? p.parity_tied_keys : std::vector<uint32_t>{};
  auto keys_string = [&](const std::vector<uint32_t> &ks) {
    std::string out;
    for(auto k : ks) out += " " + macis::parity_key_string(k, L.ngroups);
    return out;
  };

  if(!have_group) {
    if(!tied.empty())
      std::cout << "WARNING: GF_BAND_AVERAGE none: parity sector " << keystr
                << " tied with" << keys_string(tied)
                << " (PARITY_TIE), but no symmetry group is available "
                   "(ASCI.SYMMETRIZE_DETS off; it requires NROTS = 0) to "
                   "relate them. The GF is that of one of the tied states and "
                   "may be band-polarized; symmetrize it on the DMFT side"
                << std::endl;
    return;
  }
  const auto &group = *stg.sym_group;

  // One representative per distinct image key
  std::map<uint32_t, const std::vector<uint32_t> *> reps;
  for(const auto &g : group) {
    uint32_t img = 0;
    for(size_t a = 0; a < L.ngroups; ++a) {
      const int b = L.group_of[g[L.group_orbs[a].front()]];
      if(b >= 0 and ((key >> a) & 1u)) img |= (1u << b);
    }
    reps.emplace(img, &g);
  }

  // Ties the group does not explain are not averaged over
  std::vector<uint32_t> unexplained;
  for(auto k : tied)
    if(!reps.count(k)) unexplained.push_back(k);
  if(!unexplained.empty())
    std::cout << "WARNING: GF_BAND_AVERAGE: parity sector " << keystr
              << " tied with" << keys_string(unexplained)
              << " (PARITY_TIE), which the SYMMETRIZE_DETS group does not map "
                 "it onto; the GF is not averaged over "
              << (unexplained.size() > 1 ? "those sectors" : "that sector")
              << std::endl;

  if(reps.size() <= 1) return;  // sector invariant: the state is symmetric

  const auto &comp = s.GF_orbs_comp;
  const auto &up = s.is_up_comp;
  const size_t n = comp.size();

  if(!s.band_average) {
    std::cout << "WARNING: GF_BAND_AVERAGE off (GF.BAND_AVERAGE = FALSE) in "
                 "parity sector "
              << keystr << ": the GF is that of one of " << reps.size()
              << " degenerate partners, band-polarized" << std::endl;
    return;
  }
  if(up.size() != n or GF.empty() or GF[0].size() != n * n) {
    std::cout << "WARNING: GF_BAND_AVERAGE none: GF.ORBS_COMP / IS_UP_COMP do "
                 "not match the computed GF size; the GF of parity sector "
              << keystr << " is returned band-polarized" << std::endl;
    return;
  }

  // idx_r[a] = position in the list of the entry g_r^-1 maps entry a to
  std::vector<std::vector<size_t>> idx;
  for(const auto &kv : reps) {
    const auto &g = *kv.second;
    std::vector<uint32_t> ginv(g.size());
    for(size_t q = 0; q < g.size(); ++q) ginv[g[q]] = q;
    std::vector<size_t> ix(n);
    for(size_t a = 0; a < n; ++a) {
      bool found = false;
      for(size_t b = 0; b < n and !found; ++b)
        if(uint32_t(comp[b]) == ginv[comp[a]] and up[b] == up[a]) {
          ix[a] = b;
          found = true;
        }
      if(!found) {
        std::cout << "WARNING: GF_BAND_AVERAGE none: the band permutation "
                     "maps GF.ORBS_COMP entry "
                  << comp[a] << " outside the list; the GF of parity sector "
                  << keystr << " is returned band-polarized" << std::endl;
        return;
      }
    }
    idx.push_back(std::move(ix));
  }

  double bias = 0., scale = 0.;
  const double w = 1. / double(idx.size());
  for(auto &Gw : GF) {
    const auto G0 = Gw;
    for(auto &x : Gw) x = 0.;
    for(const auto &ix : idx)
      for(size_t a = 0; a < n; ++a)
        for(size_t b = 0; b < n; ++b) {
          const auto v = G0[ix[a] * n + ix[b]];
          bias = std::max(bias, std::abs(G0[a * n + b] - v));
          Gw[a * n + b] += w * v;
        }
    for(const auto &x : G0) scale = std::max(scale, std::abs(x));
  }
  const auto flags = std::cout.flags();
  const auto prec = std::cout.precision();
  std::cout << std::scientific << std::setprecision(3)
            << "GF_BAND_AVERAGE parity sector " << keystr
            << ": G = 1/m sum_r G(g_r^-1 a, g_r^-1 b), m = " << idx.size()
            << " degenerate partners; max |G - G_partner| = " << bias
            << " (max |G| = " << scale << ")" << std::endl;
  std::cout.flags(flags);
  std::cout.precision(prec);
}

template <size_t N>
auto evaluate_GF(double EASCI, macis::impurity_params<N> &p,
                 macis::SDBuildHamiltonianGenerator<N> &ham_gen,
                 macis::GFSettings &gf_settings) {
  Eigen::VectorXd psi0 =
      Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(p.C.data(), p.C.size());

  std::vector<double> active_ordm(p.n_active * p.n_active);
  std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

  ham_gen.form_rdms(
      p.dets.begin(), p.dets.end(), p.dets.begin(), p.dets.end(), p.C.data(),
      macis::matrix_span<double>(active_ordm.data(), p.n_active, p.n_active),
      macis::rank4_span<double>(active_trdm.data(), p.n_active, p.n_active,
                                p.n_active, p.n_active));

  std::vector<double> occs(p.n_active, 0.);
  for(int i = 0; i < p.n_active; i++)
    occs[i] += active_ordm[i + i * p.n_active] / 2.;

  std::cout << "Orbital Occupations in the rotated basis (nrots = "
            << p.asci_settings.nrots << "):" << std::endl;
  std::cout << "(rotated)Occs: ";
  for(const auto oc : occs) std::cout << oc << ", ";
  std::cout << std::endl;

  std::cout << "EASCI = " << EASCI << std::endl;

  // Frequency grid
  std::vector<std::complex<double>> ws(gf_settings.nws,
                                       std::complex<double>(0., 0.));

  for(int i = 0; i < gf_settings.nws; i++)
    if(gf_settings.imag_freq) {
      //  MATSUBARA GRID
      ws[i] = std::complex<double>(0., (2 * i + 1) * M_PI / gf_settings.beta);
    } else {
      std::complex<double> w0(gf_settings.wmin, gf_settings.eta);
      std::complex<double> wf(gf_settings.wmax, gf_settings.eta);
      ws[i] = w0 + (wf - w0) / double(gf_settings.nws - 1) * double(i);
    }

  using gf_t = std::vector<std::vector<std::complex<double>>>;

  EASCI -= (p.E_core + p.E_inactive);

  // Particle + hole GF of psi0 for the (orbital, spin) list of `s`, in the
  // rotated basis, and the lowest N+1 / N-1 energies the band Lanczos reaches
  // (upper bounds)
  auto particle_plus_hole = [&](const macis::GFSettings &s, double &E_add,
                                double &E_rem) {
    gf_t GF(s.nws, std::vector<std::complex<double>>(
                       p.n_active * p.n_active, std::complex<double>(0., 0.)));
    gf_t GF_tmp = GF;
    std::vector<int> todelete_p, todelete_h;
    E_add = E_rem = std::numeric_limits<double>::quiet_NaN();
    // The N+/-1 GF spaces can exceed INT32_MAX Hamiltonian nonzeros, so the
    // GF path uses 64-bit CSR indices (the ASCI Hamiltonian stays 32-bit).
    macis::RunGFCalc<N, int64_t>(GF_tmp, psi0, ham_gen, p.dets, EASCI, true, ws,
                                 occs, s, todelete_p, &E_add);
    macis::RunGFCalc<N, int64_t>(GF, psi0, ham_gen, p.dets, EASCI, false, ws,
                                 occs, s, todelete_h, &E_rem);
    // Both sectors return full GF_orbs_comp^2 matrices: RunGFCalc pads the
    // dropped (vanishing add/remove vector) rows/cols with zeros. They may
    // have dropped different orbitals (e.g. a fully occupied orbital is
    // dropped from the particle sector while an empty one is dropped from the
    // hole sector), but each sector's contribution to a dropped orbital is
    // zero, so the two matrices can simply be added elementwise.
    for(size_t iw = 0; iw < s.nws; iw++)
      for(size_t k = 0; k < GF[iw].size(); k++) GF[iw][k] += GF_tmp[iw][k];
    return GF;
  };

  double E_add, E_rem;
  gf_t GF = particle_plus_hole(gf_settings, E_add, E_rem);

  // Spin average (symmetry-sector-solve.md, sec. 3.9). With NALPHA != NBETA
  // the solved state is the m = Sz member of a spin multiplet (at odd N, one
  // member of a doublet), and its G_up != G_down. Its partner with m = -Sz,
  // the spin flip of psi0, is degenerate with it whenever H is spin-flip
  // symmetric, and its GF is G_{-m}(a, b) = G_m(flip a, flip b), where flip
  // swaps the spin of an (orbital, spin) entry of GF.ORBS_COMP/IS_UP_COMP.
  // The ensemble GF is the average of the two. With SU(2) symmetry this is
  // also the average over the whole multiplet: sum_sigma G_sigma(m) does not
  // depend on m (Wigner-Eckart), so it holds for any S. At NALPHA == NBETA
  // the state is its own spin flip and nothing changes.
  if(p.nalpha != p.nbeta) {
    // H is spin-flip symmetric unless the spin-down one-body terms differ
    // (V is spin-free here)
    double max_dT = 0.;
    if(p.spin_dep)
      for(size_t k = 0; k < p.T.size() and k < p.Td.size(); ++k)
        max_dT = std::max(max_dT, std::abs(p.T[k] - p.Td[k]));
    const auto &comp = gf_settings.GF_orbs_comp;
    const auto &up = gf_settings.is_up_comp;
    if(!gf_settings.spin_average) {
      std::cout << "WARNING: GF_SPIN_AVERAGE off (GF.SPIN_AVERAGE = FALSE) at "
                   "(NALPHA, NBETA) = ("
                << p.nalpha << ", " << p.nbeta
                << "): the GF is that of the single m = Sz state, spin-biased"
                << std::endl;
    } else if(max_dT > 1e-10) {
      std::cout << "GF_SPIN_AVERAGE none: spin-dependent one-body terms "
                   "(max |T - Td| = "
                << max_dT
                << "), so the spin-flipped state is not degenerate with the "
                   "solved one; the GF of the solved state is returned"
                << std::endl;
    } else {
      if(up.size() != comp.size())
        throw std::runtime_error(
            "evaluate_GF: GF.IS_UP_COMP must have one entry per GF.ORBS_COMP "
            "entry for the spin average");
      // flipped[a] = position of (comp[a], !up[a]) in the list, if present
      std::vector<int> flipped(comp.size(), -1);
      for(size_t a = 0; a < comp.size(); ++a)
        for(size_t b = 0; b < comp.size(); ++b)
          if(comp[b] == comp[a] and up[b] != up[a]) flipped[a] = int(b);
      const bool closed =
          std::find(flipped.begin(), flipped.end(), -1) == flipped.end();
      const size_t n = comp.size();
      gf_t GF_flip;
      if(closed) {
        // Both spins of every orbital are in the list: the partner's GF is a
        // permutation of this one
        GF_flip = GF;
        for(size_t iw = 0; iw < GF.size(); ++iw)
          for(size_t a = 0; a < n; ++a)
            for(size_t b = 0; b < n; ++b)
              GF_flip[iw][a * n + b] = GF[iw][flipped[a] * n + flipped[b]];
      } else {
        // Otherwise compute the opposite spin channel of every entry on the
        // same state
        auto fs = gf_settings;
        fs.is_up_comp.flip();
        fs.is_up_basis.flip();
        fs.writeGF = false;
        double E_add_f, E_rem_f;
        GF_flip = particle_plus_hole(fs, E_add_f, E_rem_f);
        // Both runs bound the same N+1 / N-1 energies; keep the tighter
        E_add = std::fmin(E_add, E_add_f);
        E_rem = std::fmin(E_rem, E_rem_f);
      }
      // How spin-biased the single state was, before averaging
      double bias = 0., scale = 0.;
      for(size_t iw = 0; iw < GF.size(); ++iw)
        for(size_t k = 0; k < GF[iw].size(); ++k) {
          bias = std::max(bias, std::abs(GF[iw][k] - GF_flip[iw][k]));
          scale = std::max(scale, std::abs(GF[iw][k]));
          GF[iw][k] = 0.5 * (GF[iw][k] + GF_flip[iw][k]);
        }
      const auto flags = std::cout.flags();
      const auto prec = std::cout.precision();
      std::cout << std::scientific << std::setprecision(3)
                << "GF_SPIN_AVERAGE (NALPHA, NBETA) = (" << p.nalpha << ", "
                << p.nbeta << "): G = [G_m(a,b) + G_m(flip a, flip b)] / 2, "
                << (closed ? "both spins requested, no extra GF run"
                           : "opposite spin channel computed in a second run")
                << "; max |G_m - G_-m| = " << bias << " (max |G| = " << scale
                << ")" << std::endl;
      std::cout.flags(flags);
      std::cout.precision(prec);
    }
  }

  report_sector_check(E_add - EASCI, E_rem - EASCI, p);

  EASCI += p.E_core + p.E_inactive;

  // Rotate the GF back to original basis

  size_t G_n_orbs = sqrt(GF[0].size());

  // The loops below index rows/columns 0..n_imp-1 of the returned GF, so the
  // GF must be at least that large. It is not if fewer than n_imp orbitals
  // were requested in GF_orbs_comp, or if any of them were dropped.
  if(G_n_orbs < p.n_imp)
    throw std::runtime_error(
        "In evaluate_GF: the computed Green's function is smaller than n_imp, "
        "cannot rotate the impurity block back to the original basis. Check "
        "GF.ORBS_COMP and the list of dropped orbitals.");

  // The back-rotation assumes GF row j corresponds to impurity orbital j. That
  // mapping now always holds: RunGFCalc pads the GF back to the full
  // GF_orbs_comp index space (with zeros where an orbital's add/remove vector
  // vanished), so rows map onto GF_orbs_comp order, and hence onto impurity
  // orbitals 0..n_imp-1 whenever GF_orbs_comp is the full impurity set.
  // A zero row/col here means that orbital contributes nothing from the
  // corresponding sector, which is the correct physical limit.

  Eigen::MatrixXd rotMat = Eigen::MatrixXd::Identity(p.n_imp, p.n_imp);
  for(int j = 0; j < p.n_imp; j++)
    for(int k = 0; k < p.n_imp; k++)
      rotMat(j, k) = p.orb_rot[j + k * p.n_active];

  // rotMat is the top-left n_imp x n_imp block of the n_active x n_active
  // orb_rot. That block is only unitary on its own if orb_rot is block-diagonal
  // in imp/bath, which rotate_hamiltonian_ordm_imp_bath guarantees but the full
  // rotate_hamiltonian_ordm (asci_grow's grow_with_rot path) does not. Without
  // this check a non-block-diagonal orb_rot would silently apply a
  // non-unitary truncation to the GF.
  if(p.n_imp > 0) {
    const Eigen::MatrixXd dev = rotMat * rotMat.transpose() -
                                Eigen::MatrixXd::Identity(p.n_imp, p.n_imp);
    const double unitarity_tol = 1.e-8;
    if(dev.cwiseAbs().maxCoeff() > unitarity_tol)
      throw std::runtime_error(
          "In evaluate_GF: the impurity block of orb_rot is not unitary (max "
          "deviation of R R^T from the identity is " +
          std::to_string(dev.cwiseAbs().maxCoeff()) +
          "). orb_rot is not block-diagonal in imp/bath, so the impurity "
          "Green's function cannot be rotated back with its impurity block "
          "alone.");
  }

  for(int iw = 0; iw < gf_settings.nws; iw++) {
    Eigen::MatrixXcd G = Eigen::MatrixXcd::Zero(p.n_imp, p.n_imp);
    for(int j = 0; j < p.n_imp; j++)
      for(int k = 0; k < p.n_imp; k++) {
        G(j, k) = GF[iw][j + k * G_n_orbs];
      }

    //  Eigen::MatrixXcd rotG  = rotMat.adjoint() * G * rotMat;
    Eigen::MatrixXcd rotG = rotMat * G * rotMat.adjoint();

    for(int j = 0; j < p.n_imp; j++)
      for(int k = 0; k < p.n_imp; k++) GF[iw][j + k * G_n_orbs] = rotG(j, k);
  }

  // Band-orbit average, in the original basis where the permutation group of
  // SYMMETRIZE_DETS acts on orbital labels (see band_orbit_average_gf)
  band_orbit_average_gf<N>(GF, p, gf_settings);

  if(gf_settings.writeGF_singlef)
    macis::write_GF(GF, ws, gf_settings.GF_orbs_comp, std::vector<int>{});

  return GF;
}

/**
 * @brief Evaluates the dynamical impurity Sz-Sz response, i.e. the retarded
 *        resolvent of the impurity Sz operator on the ASCI ground state:
 *
 *          R(w) = <psi0| Sz_imp  1/(w - (H - E0))  Sz_imp |psi0>
 *
 *        Sz_imp is applied to the ground state by rescaling each determinant
 *        coefficient (see macis::RunResolventSz / macis::sz_imp_value), and the
 *        single-vector resolvent of the resulting state is evaluated over the
 *        same real grid as evaluate_GF but on a bosonic Matsubara grid
 *        (even multiples of pi/beta) on the imaginary axis, since the Sz-Sz
 *        response is a bosonic correlator. The grid starts at nu_1 = 2*pi/beta
 *        and deliberately EXCLUDES nu_0 = 0: see the comment on the grid
 *        construction below. The result R(w) is written to
 *        "Sz_resolvent.dat" (columns: Re(w) Im(w) Re(R) Im(R)) when
 *        gf_settings.writeGF_singlef is set, and returned to the caller.
 *
 * @param[in] double EASCI: ASCI ground-state energy (including core/inactive).
 * @param[in] macis::impurity_params<N> &p: Impurity problem parameters.
 * @param[in] macis::SDBuildHamiltonianGenerator<N> &ham_gen: Hamiltonian
 *            generator.
 * @param[in] macis::GFSettings &gf_settings: GF/resolvent settings (frequency
 *            grid, nLanIts, saveGFmats, writeGF_singlef).
 *
 * @returns std::vector<std::complex<double>>: R(w) along the frequency grid.
 */
template <size_t N>
auto evaluate_resolvent_sz(double EASCI, macis::impurity_params<N> &p,
                           macis::SDBuildHamiltonianGenerator<N> &ham_gen,
                           macis::GFSettings &gf_settings) {
  Eigen::VectorXd psi0 =
      Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(p.C.data(), p.C.size());

  std::cout << "EASCI = " << EASCI << std::endl;

  // Frequency grid: real axis as in evaluate_GF, but on the imaginary axis
  // use a BOSONIC Matsubara grid (even multiples of pi/beta): the Sz-Sz
  // resolvent is a spin (bosonic) correlator, unlike the fermionic GF which
  // lives on odd multiples (2*i+1)*pi/beta.
  //
  // The grid starts at nu_1 = 2*pi/beta, NOT at nu_0 = 0. w = 0 is a point on
  // the REAL axis, which is exactly where the poles of the continued fraction
  // sit, and there is no i*eta broadening on the Matsubara axis to keep it
  // away from them. E0 comes from Davidson at CI_RES_TOL, while the smallest
  // eigenvalue of the Lanczos tridiagonal matrix is the exact ground state of
  // H in the Krylov space, so the two differ by O(CI_RES_TOL) with an
  // essentially arbitrary sign. That plants a spurious pole at w ~ delta -> 0
  // whose contribution to R is -weight/delta at w = 0 and only
  // -weight*delta/(nu^2 + delta^2) ~ 0 at every nu >> delta. The nu = 0 value
  // was therefore the only unusable point on the grid (observed: static
  // susceptibilities wrong by orders of magnitude, and negative), while
  // nu >= nu_1 is smooth. Recover the static limit by extrapolating R(i*nu)
  // from the lowest few Matsubara points instead.
  std::vector<std::complex<double>> ws(gf_settings.nws,
                                       std::complex<double>(0., 0.));
  for(int i = 0; i < gf_settings.nws; i++)
    if(gf_settings.imag_freq) {
      //  BOSONIC MATSUBARA GRID, STARTING AT nu_1 = 2*pi/beta
      ws[i] = std::complex<double>(
          0., 2. * double(i + 1) * M_PI / gf_settings.beta);
    } else {
      std::complex<double> w0(gf_settings.wmin, gf_settings.eta);
      std::complex<double> wf(gf_settings.wmax, gf_settings.eta);
      ws[i] = w0 + (wf - w0) / double(gf_settings.nws - 1) * double(i);
    }

  // Reference energy relative to the active-space Hamiltonian (matching the
  // shift used for the Green's function in evaluate_GF).
  double E0 = EASCI - (p.E_core + p.E_inactive);

  std::vector<std::complex<double>> R = macis::RunResolventSz<N>(
      psi0, ham_gen, p.dets, p.n_imp, p.n_active, E0, ws, gf_settings);

  if(gf_settings.writeGF_singlef) {
    // Guard the file write so only rank 0 touches the shared filename
    // (all ranks compute identical R(w), so one write is sufficient).
    bool write_file = true;
    MACIS_MPI_CODE(write_file = (macis::comm_rank(MPI_COMM_WORLD) == 0);)
    if(write_file) {
      using dbl = std::numeric_limits<double>;
      std::ofstream ofile("Sz_resolvent.dat");
      ofile.precision(dbl::max_digits10);
      for(size_t iii = 0; iii < ws.size(); iii++)
        ofile << std::scientific << real(ws[iii]) << " " << imag(ws[iii]) << " "
              << real(R[iii]) << " " << imag(R[iii]) << std::endl;
    }
  }

  return R;
}

template <size_t N>
auto evaluate_ordm(std::vector<macis::wfn_t<N>> &dets,
                   std::vector<double> &X_local,
                   macis::SDBuildHamiltonianGenerator<N> &ham_gen,
                   std::vector<double> &orb_rot) {
  // Get Parameters
  size_t n_active = sqrt(orb_rot.size());

  // Compute the 1-rdm
  typename std::vector<macis::wfn_t<N>>::iterator det_st = dets.begin();
  typename std::vector<macis::wfn_t<N>>::iterator det_en = dets.end();

  std::vector<double> active_ordm(n_active * n_active);
  std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

  ham_gen.form_rdms(
      dets.begin(), dets.end(), dets.begin(), dets.end(), X_local.data(),
      macis::matrix_span<double>(active_ordm.data(), n_active, n_active),
      macis::rank4_span<double>(active_trdm.data(), n_active, n_active,
                                n_active, n_active));

  std::cout << "Orbital Occupations in the rotated basis:" << std::endl;
  std::cout << "(rotated)Occs: ";
  for(int i = 0; i < n_active; i++)
    std::cout << active_ordm[i + i * n_active] / 2. << ", ";
  std::cout << std::endl;

  //   //print ordm DEBUG
  //  macis::util::write_matrix(active_ordm.data(), n_active, n_active,
  //                            "active_ordm_rotated.dat", true);

  // Rotate the 1-RDM back to original basis
  // Eigen::MatrixXd roto = orb_rot * o * orb_rot.adjoint();
  std::vector<double> tmp(n_active * n_active, 0.);
  std::vector<double> comp(n_active * n_active, 0.);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
             n_active, n_active, n_active, 1.0, active_ordm.data(), n_active,
             orb_rot.data(), n_active, 0.0, tmp.data(), n_active);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             n_active, n_active, n_active, 1.0, orb_rot.data(), n_active,
             tmp.data(), n_active, 0.0, comp.data(), n_active);

  return comp;
}

template <size_t N>
double SolveImpurityED(impurity_params<N> &params);

template <size_t N>
double SolveImpurityASCI(impurity_params<N> &params);

template <size_t N>
double SolveImpurityASCI_rot(impurity_params<N> &params);

template <size_t N>
double SolveImpurityCheapASCI(impurity_params<N> &params);

/**
 * @brief Turn on the band-parity sector solve (ASCI.PARITY_SOLVE).
 *
 * Detects the band groups, verifies that the Hamiltonian conserves every band
 * parity, and zeroes parity-breaking integrals up to `tol` in p.T, p.Td and
 * p.V (larger ones throw); see build_parity_labels. Call it before the active
 * integrals (T_active, V_active, ...) are built from p.T / p.V, so that they
 * see the cleaned integrals. Requires GROW_WITH_ROT = FALSE, NINACTIVE = 0
 * and NACTIVE = NORB. NROTS > 0 is allowed: inside a sector the natural
 * orbitals are taken per band group, so the labels survive the rotations.
 */
template <size_t N>
void setup_parity_sectors(impurity_params<N> &params, double tol);

}  // namespace macis
