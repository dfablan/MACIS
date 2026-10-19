#include "macis/doping/charge_sectors.hpp"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>

#include "macis/doping/fix_mu.hpp"

namespace fs = std::filesystem;

namespace macis {
namespace charge_sectors {

namespace {

bool is_root_rank() {
#ifdef MACIS_ENABLE_MPI
  return macis::comm_rank(MPI_COMM_WORLD) == 0;
#else
  return true;
#endif
}

void barrier() { MACIS_MPI_CODE(MPI_Barrier(MPI_COMM_WORLD);) }

// Gather a Davidson eigenvector distributed as in selected_ci_diag (block rows,
// remainder on the last rank) into a full, replicated vector. Same scheme as
// asci_iter.
std::vector<double> gather_C(std::vector<double> C_local, size_t ndets) {
#ifdef MACIS_ENABLE_MPI
  auto world_size = macis::comm_size(MPI_COMM_WORLD);
  auto world_rank = macis::comm_rank(MPI_COMM_WORLD);
  if(world_size > 1) {
    std::vector<double> C(ndets);
    const size_t local_count = ndets / world_size;
    MPI_Allgather(C_local.data(), local_count, MPI_DOUBLE, C.data(),
                  local_count, MPI_DOUBLE, MPI_COMM_WORLD);
    if(ndets % world_size) {
      const size_t nrem = ndets % world_size;
      auto* C_rem = C.data() + world_size * local_count;
      if(world_rank == world_size - 1)
        std::copy_n(C_local.data() + local_count, nrem, C_rem);
      MPI_Bcast(C_rem, nrem, MPI_DOUBLE, world_size - 1, MPI_COMM_WORLD);
    }
    return C;
  }
#endif
  (void)ndets;
  return C_local;
}

// ---- seeds -----------------------------------------------------------------

template <size_t N>
using weight_map_t =
    std::map<wfn_t<N>, double, macis::bitset_less_comparator<N>>;

// Apply c^dagger_{i,spin} (create) or c_{i,spin} to the top @p nparents
// determinants of (dets, C) (ranked by |C|) for every active orbital i that
// allows it. Each image collects sqrt(sum C_parent^2) over the parents that
// reach it. Only the ranking is used afterwards (Davidson recomputes C), so
// fermionic signs are dropped: summing signed amplitudes could cancel an image
// that two different i reach.
template <size_t N>
weight_map_t<N> apply_ladder(const std::vector<wfn_t<N>>& dets,
                             const std::vector<double>& C, int spin,
                             bool create, size_t nparents, size_t n_active) {
  std::vector<size_t> order(dets.size());
  std::iota(order.begin(), order.end(), 0);
  nparents = std::min(nparents, dets.size());
  std::partial_sort(
      order.begin(), order.begin() + nparents, order.end(),
      [&](size_t x, size_t y) { return std::abs(C[x]) > std::abs(C[y]); });

  const size_t off = spin ? N / 2 : 0;
  weight_map_t<N> w;
  for(size_t k = 0; k < nparents; ++k) {
    const auto& d = dets[order[k]];
    const double c2 = C[order[k]] * C[order[k]];
    for(size_t i = 0; i < n_active; ++i) {
      const bool occ = d[i + off];
      if(occ == create) continue;
      auto e = d;
      e.flip(i + off);
      w[e] += c2;
    }
  }
  for(auto& [d, x] : w) x = std::sqrt(x);
  return w;
}

template <size_t N>
std::pair<std::vector<wfn_t<N>>, std::vector<double>> unzip(
    const weight_map_t<N>& w) {
  std::vector<wfn_t<N>> d;
  std::vector<double> c;
  d.reserve(w.size());
  c.reserve(w.size());
  for(auto& [k, v] : w) {
    d.push_back(k);
    c.push_back(v);
  }
  return {d, c};
}

// One ladder step from (pa, pb) to (ta, tb), or two for a move in both spins.
template <size_t N>
weight_map_t<N> ladder_to(const std::vector<wfn_t<N>>& dets,
                          const std::vector<double>& C, size_t pa, size_t pb,
                          size_t ta, size_t tb, size_t nparents,
                          size_t n_active) {
  if(pa == ta and pb == tb) {
    weight_map_t<N> w;
    for(size_t i = 0; i < dets.size(); ++i) w[dets[i]] += C[i] * C[i];
    for(auto& [d, x] : w) x = std::sqrt(x);
    return w;
  }
  // Beta first, so an S_z flip (a+1, b-1) removes a down electron before adding
  // an up one.
  if(pb != tb) {
    const bool create = tb > pb;
    if((create ? tb - pb : pb - tb) != 1)
      throw std::logic_error("ladder_to: beta count changes by more than one");
    auto w = apply_ladder<N>(dets, C, 1, create, nparents, n_active);
    auto [d, c] = unzip<N>(w);
    return ladder_to<N>(d, c, pa, tb, ta, tb, pa == ta ? nparents : d.size(),
                        n_active);
  }
  const bool create = ta > pa;
  if((create ? ta - pa : pa - ta) != 1)
    throw std::logic_error("ladder_to: alpha count changes by more than one");
  return apply_ladder<N>(dets, C, 0, create, nparents, n_active);
}

// Keep the @p nkeep heaviest determinants, and check they all sit in (na, nb).
template <size_t N>
std::vector<wfn_t<N>> truncate_seed(const weight_map_t<N>& w, size_t nkeep,
                                    size_t na, size_t nb) {
  std::vector<std::pair<double, wfn_t<N>>> v;
  v.reserve(w.size());
  for(auto& [d, x] : w) v.push_back({x, d});
  nkeep = std::min(nkeep, v.size());
  // Ties broken on the bitset so every rank keeps the same determinants.
  std::partial_sort(v.begin(), v.begin() + nkeep, v.end(),
                    [](const auto& x, const auto& y) {
                      if(x.first != y.first) return x.first > y.first;
                      return macis::bitset_less(x.second, y.second);
                    });
  std::vector<wfn_t<N>> dets(nkeep);
  for(size_t i = 0; i < nkeep; ++i) {
    dets[i] = v[i].second;
    const size_t a = macis::bitset_lo_word(dets[i]).count();
    const size_t b = macis::bitset_hi_word(dets[i]).count();
    if(a != na or b != nb)
      throw std::logic_error("seed determinant in " + sector_str(a, b) +
                             ", expected " + sector_str(na, nb));
  }
  return dets;
}

template <size_t N>
void restore(SectorContext<N>& ctx) {
  auto& p = *ctx.p;
  p.T_active = ctx.pristine.T_active;
  p.V_active = ctx.pristine.V_active;
  p.Td_active = ctx.pristine.Td_active;
  p.asci_settings = ctx.pristine.asci_settings;
  p.just_singles = ctx.pristine.just_singles;
  p.asci_wfn_fname.clear();
  p.compute_asci_E0 = true;
  p.asci_E0 = 0.0;
}

// Diagonalize in the seed space and hand the result to SolveImpurityASCI_rot as
// a guess. Returns E_seed (total energy).
template <size_t N>
double solve_from_seed(SectorContext<N>& ctx, std::vector<wfn_t<N>> dets,
                       const std::string& fname) {
  auto& p = *ctx.p;
  macis::SDBuildHamiltonianGenerator<N> ham_gen(
      macis::matrix_span<double>(p.T_active.data(), p.n_active, p.n_active),
      macis::rank4_span<double>(p.V_active.data(), p.n_active, p.n_active,
                                p.n_active, p.n_active));
  if(p.spin_dep)
    ham_gen.ReadTdo(
        macis::matrix_span<double>(p.Td_active.data(), p.n_active, p.n_active));
  ham_gen.SetJustSingles(p.just_singles);
  ham_gen.SetNimp(p.n_imp);

  double E_act;
  std::vector<double> C;
  if(dets.size() == 1) {
    E_act = ham_gen.matrix_element(dets[0], dets[0]);
    C = {1.0};
  } else {
    std::vector<double> C_local;
    E_act = macis::selected_ci_diag(
        dets.begin(), dets.end(), ham_gen, p.mcscf_settings.ci_matel_tol,
        p.mcscf_settings.ci_max_subspace, p.mcscf_settings.ci_res_tol,
        C_local MACIS_MPI_CODE(, MPI_COMM_WORLD), true);
    C = gather_C(std::move(C_local), dets.size());
  }
  const double E_seed = E_act + p.E_core + p.E_inactive;

  if(is_root_rank()) macis::write_wavefunction(fname, p.n_active, dets, C);
  barrier();

  p.asci_wfn_fname = fname;
  p.compute_asci_E0 = false;
  p.asci_E0 = E_seed;
  return E_seed;
}

template <size_t N>
SectorResult<N> solve_sector_once(SectorContext<N>& ctx, size_t na, size_t nb,
                                  const SeedSource<N>& src) {
  auto& p = *ctx.p;
  auto& out = *ctx.out;
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();

  SectorResult<N> r;
  r.na = na;
  r.nb = nb;
  r.seed = src.label;

  out << "\n"
      << std::string(90, '=') << "\n"
      << "SECTOR " << sector_str(na, nb) << "  N = " << na + nb
      << "  start: " << src.label << "\n"
      << std::string(90, '=') << std::endl;

  try {
    restore(ctx);
    p.nalpha = na;
    p.nbeta = nb;
    // A seeded sector is solved with NROTS = 0 in the basis of its seed (the
    // guess is only meaningful in that basis, and load_asci_guess refuses
    // NROTS > 0); a cold one with the NROTS of the mode. A CAS solve has no
    // seed and no rotation.
    const bool seeded = src.kind != SeedSource<N>::Cold and !ctx.use_ed;
    const size_t nrots = seeded ? 0 : ctx.nrots_cold;
    p.asci_settings.nrots = nrots;
    if(seeded and src.U and !src.U->empty()) {
      rotate_active<N>(p, *src.U);
      out << "basis       : natural orbitals inherited from " << src.label
          << " (NROTS = 0 in that basis)" << std::endl;
    }

    if(seeded) {
      std::vector<wfn_t<N>> seed;
      if(src.kind == SeedSource<N>::Parent) {
        auto w = ladder_to<N>(*src.dets, *src.C, src.pa, src.pb, na, nb,
                              ctx.seed_parents, p.n_active);
        seed = truncate_seed<N>(w, ctx.seed_size, na, nb);
      } else {
        std::vector<wfn_t<N>> d;
        std::vector<double> c;
        macis::read_wavefunction(src.fname, d, c, true);
        weight_map_t<N> w;
        for(size_t i = 0; i < d.size(); ++i) w[d[i]] += c[i] * c[i];
        seed = truncate_seed<N>(w, ctx.seed_size, na, nb);
      }
      if(seed.empty())
        throw std::runtime_error(
            "empty seed: no determinant of the parent maps into " +
            sector_str(na, nb));
      const std::string fname = ctx.opt.workdir + "/seed_Na" +
                                std::to_string(na) + "_Nb" +
                                std::to_string(nb) + ".wfn";
      r.E_seed = solve_from_seed<N>(ctx, std::move(seed), fname);
      out << "seed        : " << p.asci_wfn_fname << "  E_seed = " << std::fixed
          << std::setprecision(10) << r.E_seed << std::endl;
    }

    r.E = ctx.use_ed ? macis::SolveImpurityED<N>(p)
                     : macis::SolveImpurityASCI_rot<N>(p);
    r.converged = std::isfinite(r.E);
    r.status = r.converged ? "OK" : "NONFINITE";
    r.ndets = p.dets.size();
    double s = 0;
    for(size_t i = 0; i < p.n_imp; ++i) s += p.occs[i];
    r.n_band = 2.0 * s / p.n_imp;
    // A CAS solution is the whole Hilbert space: never used as a seed.
    if(r.converged and !ctx.use_ed) {
      r.dets = p.dets;
      r.C = p.C;
    }
    // The basis the solution is written in. A seeded solve runs with NROTS = 0
    // (the solver's orb_rot stays the identity) in its seed's basis; a cold one
    // starts from the original orbitals and ends in the solver's orb_rot.
    // n_band above needs no back-rotation either way: every rotation here is
    // block-diagonal in (impurity, bath), so the impurity trace is invariant.
    if(seeded)
      r.U = src.U ? *src.U : basis_t{};
    else if(nrots > 0 and !ctx.use_ed)
      r.U = p.orb_rot;
  } catch(const std::exception& e) {
    r.converged = false;
    std::string what = e.what();
    r.status = what.find("did not converge") != std::string::npos
                   ? "UNCONVERGED"
                   : "FAILED";
    out << "WARNING: sector " << sector_str(na, nb) << " " << r.status << ": "
        << what << std::endl;
  }
  r.time_s = std::chrono::duration<double>(clock::now() - t0).count();
  return r;
}

}  // namespace

std::string sector_str(size_t a, size_t b) {
  return "(" + std::to_string(a) + "," + std::to_string(b) + ")";
}

template <size_t N>
void rebuild_active(impurity_params<N>& p) {
  macis::active_hamiltonian(NumOrbital(p.norb), NumActive(p.n_active),
                            NumInactive(p.n_inactive), p.T.data(), p.norb,
                            p.V.data(), p.norb, p.F_inactive.data(), p.norb,
                            p.T_active.data(), p.n_active, p.V_active.data(),
                            p.n_active);
  if(p.spin_dep)
    macis::active_hamiltonian(
        NumOrbital(p.norb), NumActive(p.n_active), NumInactive(p.n_inactive),
        p.Td.data(), p.norb, p.V.data(), p.norb, p.Fd_inactive.data(), p.norb,
        p.Td_active.data(), p.n_active, p.V_active.data(), p.n_active);
}

// The rotation mixes impurity orbitals among themselves, so the interaction is
// no longer density-density: singles-only must be turned off, and it must be
// turned off in p, because SolveImpurityASCI_rot re-applies p.just_singles to
// its own generator.
template <size_t N>
void rotate_active(impurity_params<N>& p, const basis_t& U) {
  // Under PARITY_SOLVE the seed's determinants carry band-parity labels only
  // if the basis keeps every orbital in its band group. A parity solve's own
  // orb_rot does (per-band natural orbitals); one from a run without the
  // parity solve generally does not.
  if(p.parity_labels) {
    const double off = macis::max_off_group(U.data(), *p.parity_labels);
    if(off > p.parity_labels->tol) {
      std::ostringstream os;
      os << std::scientific << std::setprecision(2)
         << "PARITY_SOLVE: the inherited orbital basis mixes band groups "
            "(largest cross-band element "
         << off << " > PARITY_TOL = " << p.parity_labels->tol
         << "), so its determinants have no band parity. It was not written "
            "by a parity solve.";
      throw std::runtime_error(os.str());
    }
  }
  macis::SDBuildHamiltonianGenerator<N> ham_gen(
      macis::matrix_span<double>(p.T_active.data(), p.n_active, p.n_active),
      macis::rank4_span<double>(p.V_active.data(), p.n_active, p.n_active,
                                p.n_active, p.n_active));
  if(p.spin_dep)
    ham_gen.ReadTdo(
        macis::matrix_span<double>(p.Td_active.data(), p.n_active, p.n_active));
  basis_t Ucopy = U;
  ham_gen.rotate_hamiltonian_rotmat_imp_bath(Ucopy.data(), p.spin_dep);
  p.just_singles = false;
}

// A seeded solve that fails for any reason other than refinement not converging
// (e.g. the seed already holds NTDETS_MAX determinants and the ASCI search
// cannot reproduce that many, which asci_refine refuses) is retried cold, so
// the scan does not lose the sector.
template <size_t N>
SectorResult<N> solve_sector(SectorContext<N>& ctx, size_t na, size_t nb,
                             const SeedSource<N>& src) {
  auto r = solve_sector_once<N>(ctx, na, nb, src);
  if(src.kind == SeedSource<N>::Cold or ctx.use_ed or r.converged or
     r.status == "UNCONVERGED")
    return r;
  *ctx.out << "retrying " << sector_str(na, nb)
           << " cold after the seeded solve failed" << std::endl;
  SeedSource<N> cold;
  cold.label = "cold(seed failed)";
  auto rc = solve_sector_once<N>(ctx, na, nb, cold);
  rc.E_seed = r.E_seed;
  rc.time_s += r.time_s;
  return rc;
}

// ---- search ------------------------------------------------------------------

template <size_t N>
void SectorScan<N>::solve(size_t n, std::optional<size_t> Np) {
  const size_t a = split_alpha(n, ctx_.beta_heavy);
  const size_t b = split_beta(n, ctx_.beta_heavy);
  SeedSource<N> src;
  if(ctx_.use_ed) {
    src.label = "ED";
  } else if(ctx_.opt.warm and Np and res.count(*Np) and
            !res.at(*Np).dets.empty()) {
    const auto& par = res.at(*Np);
    src.kind = SeedSource<N>::Parent;
    src.dets = &par.dets;
    src.C = &par.C;
    src.pa = par.na;
    src.pb = par.nb;
    src.U = &par.U;
    src.label = sector_str(par.na, par.nb);
  } else if(ctx_.opt.warm and Np) {
    src.label = "cold(parent failed)";
  }
  res[n] = solve_sector<N>(ctx_, a, b, src);
  prune();
}

template <size_t N>
void SectorScan<N>::solve_reference(const SeedSource<N>& src) {
  SeedSource<N> s = src;
  if(ctx_.use_ed) s.label = "ED";
  res[N0_] = solve_sector<N>(ctx_, split_alpha(N0_, ctx_.beta_heavy),
                             split_beta(N0_, ctx_.beta_heavy), s);
}

template <size_t N>
void SectorScan<N>::set(size_t n, SectorResult<N> r) {
  res[n] = std::move(r);
}

template <size_t N>
std::optional<size_t> SectorScan<N>::argmin() const {
  std::optional<size_t> m;
  for(auto& [n, r] : res)
    if(r.converged and (!m or r.E < res.at(*m).E)) m = n;
  return m;
}

template <size_t N>
void SectorScan<N>::walk(long margin) {
  if(in_range(long(N0_) - 1)) solve(N0_ - 1, N0_);
  if(in_range(long(N0_) + 1)) solve(N0_ + 1, N0_);
  while(true) {
    auto m = argmin();
    if(!m)
      throw std::runtime_error("no sector converged; nothing to walk from");
    if(long(*m) - long(lo()) < margin and lo() > 0) {
      solve(lo() - 1, lo());
      continue;
    }
    if(long(hi()) - long(*m) < margin and hi() < Nmax_) {
      solve(hi() + 1, hi());
      continue;
    }
    break;
  }
}

template <size_t N>
void SectorScan<N>::window(long W) {
  for(long d = 1; d <= W; ++d) {
    if(in_range(long(N0_) - d)) solve(N0_ - d, N0_ - d + 1);
    if(in_range(long(N0_) + d)) solve(N0_ + d, N0_ + d - 1);
  }
}

// Wavefunctions are needed only by sectors that can still seed a neighbour: the
// two ends of the solved range, and the current minimum (for --check-spin, and
// as the guess of a sector switch).
template <size_t N>
void SectorScan<N>::prune() {
  auto m = argmin();
  for(auto& [n, r] : res)
    if(n != lo() and n != hi() and (!m or n != *m)) {
      r.dets = {};
      r.C = {};
    }
}

// ---- reporting ---------------------------------------------------------------

template <size_t N>
void write_row(std::ostream& os, size_t n, const SectorResult<N>& r,
               double Emin) {
  auto num = [&](double x, int prec) {
    std::ostringstream s;
    if(std::isfinite(x))
      s << std::fixed << std::setprecision(prec) << x;
    else
      s << "-";
    return s.str();
  };
  os << std::setw(4) << n << std::setw(8) << r.na << std::setw(7) << r.nb
     << std::setw(19) << num(r.E, 10) << std::setw(14)
     << num(r.converged ? r.E - Emin : NAN, 8) << std::setw(10)
     << num(r.n_band, 5) << std::setw(19) << num(r.E_seed, 10) << std::setw(10)
     << r.ndets << std::setw(10) << num(r.time_s, 1) << "  " << std::setw(20)
     << std::left << r.seed << std::right << "  " << r.status << "\n";
}

void write_header(std::ostream& os, const std::string& lead) {
  os << lead << std::setw(3 - long(lead.size()) + 1) << "N" << std::setw(8)
     << "NALPHA" << std::setw(7) << "NBETA" << std::setw(19) << "E(CI)=Omega"
     << std::setw(14) << "E-E_min" << std::setw(10) << "n/band" << std::setw(19)
     << "E_seed" << std::setw(10) << "ndets" << std::setw(10) << "time[s]"
     << "  " << std::setw(20) << std::left << "seed" << std::right
     << "  status\n";
}

}  // namespace charge_sectors

// ---- ground-sector file --------------------------------------------------------

template <size_t N>
void write_ground_sector_file(const std::string& fname,
                              const impurity_params<N>& p, double mu,
                              const std::string& extra) {
  if(!charge_sectors::is_root_rank()) return;
  std::ofstream f(fname);
  if(!f) throw std::runtime_error("cannot write " + fname);
  f << std::scientific << std::setprecision(12);
  f << "GROUND_SECTOR NALPHA = " << p.nalpha << " NBETA = " << p.nbeta
    << " N = " << p.nalpha + p.nbeta << " E = " << p.E << " MU = " << mu
    << "\n";
  if(p.occs.size() >= p.n_imp && p.n_imp > 0) {
    double s = 0;
    for(size_t i = 0; i < p.n_imp; ++i) s += p.occs[i];
    f << "# electrons per impurity orbital = " << 2.0 * s / p.n_imp
      << "  (target " << p.nel_target << ")\n";
  }
  f << extra;
}

// ---- mu search + sector search -------------------------------------------------

template <size_t N>
double Fix_Mu_sectors(const std::string& method_name, bool deriv,
                      double& init_mu, impurity_params<N>* params,
                      const ChargeSectorSettings& cs) {
  using namespace charge_sectors;
  auto& p = *params;
  const bool root = is_root_rank();
  std::ostream null_stream(nullptr);
  std::ostream& out = root ? *cs.out : null_stream;
  // The driver prints with its own stream format afterwards: leave it as found.
  struct FormatGuard {
    std::ostream& o;
    std::ios::fmtflags flags;
    std::streamsize precision;
    ~FormatGuard() {
      o.flags(flags);
      o.precision(precision);
    }
  } format_guard{out, out.flags(), out.precision()};

  const CIExpansion ci_orig = p.ci_exp;
  const bool use_ed = ci_orig == CIExpansion::CAS;
  const fs::path cwd0 = fs::current_path();
  const std::string gs_file = (cwd0 / "GS_charge_sector.dat").string();

  // The mu search restarts from the input's expansion every time: cheap mode
  // switches p.ci_exp to ASCI_cheap and leaves it there, and the cheap solver
  // reuses the determinants of the previous solve, which belong to the old
  // sector after a switch.
  auto run_search = [&](double mu0) {
    p.ci_exp = ci_orig;
    return deriv ? Fix_Mu_der<N>(method_name, mu0, params)
                 : Fix_Mu_noder<N>(method_name, mu0, params);
  };
  auto filling = [&]() {
    double s = 0;
    for(size_t i = 0; i < p.n_imp; ++i) s += p.occs[i];
    return 2.0 * s / p.n_imp;
  };

  const long dSz = long(p.nalpha) - long(p.nbeta);
  if(!cs.enabled or std::abs(dSz) > 1) {
    if(cs.enabled)
      out << "WARNING: sector search skipped: (NALPHA, NBETA) = "
          << sector_str(p.nalpha, p.nbeta)
          << " is not a minimal-|S_z| sector, so it is not comparable with "
             "the sectors the search would scan. Fixing mu in the input "
             "sector only."
          << std::endl;
    const double mu = run_search(init_mu);
    write_ground_sector_file<N>(
        gs_file, p, mu,
        "# sector search skipped: the sector is the one of input.in, not "
        "verified\n");
    return mu;
  }

  const bool beta_heavy = p.nbeta > p.nalpha;
  const size_t Nmax = 2 * p.n_active;
  const size_t nrots_input = p.asci_settings.nrots;
  const bool warm = cs.warm and !use_ed and p.asci_settings.max_refine_iter > 0;
  if(cs.warm and !use_ed and !warm)
    out << "note        : sector search: warm start needs ASCI.MAX_REFINE_ITER "
           "> 0; neighbours are solved cold\n";
  if(p.spin_dep)
    out << "note        : spin-dependent input: the +-S_z mirrors are not "
           "equivalent; only the minimal-|S_z| sector of each N is scanned\n";

  ChargeSectorSettings opt = cs;
  {
    fs::path wd = fs::absolute(cs.workdir);
    opt.workdir = wd.lexically_normal().string();
  }
  opt.warm = warm;

  struct Visit {
    double mu, n;
  };
  std::map<size_t, Visit> searched;  // sector N -> (mu, filling) of its mu search
  size_t switches = 0;
  double mu_start = init_mu;
  bool guess_active = false;

  while(true) {
    const size_t N0 = p.nalpha + p.nbeta;
    const std::string sec = sector_str(p.nalpha, p.nbeta);
    out << "\n" << std::string(90, '#') << "\nMU SEARCH in sector " << sec
        << "  N = " << N0 << "\n" << std::string(90, '#') << std::endl;

    double mu = 0;
    for(int attempt = 0;; ++attempt) {
      try {
        mu = run_search(mu_start);
        break;
      } catch(const std::exception& e) {
        if(guess_active and attempt == 0) {
          out << "WARNING: mu search seeded from the scan failed (" << e.what()
              << "); retrying cold" << std::endl;
          p.asci_wfn_fname.clear();
          p.compute_asci_E0 = true;
          p.asci_E0 = 0.0;
          guess_active = false;
          continue;
        }
        if(switches > 0)
          throw std::runtime_error(
              "mu search in sector " + sec + " (reached by switching sector) "
              "failed: " + e.what() +
              ". The target filling may not be reachable in that sector.");
        throw;
      }
    }
    guess_active = false;
    const double n_fill = filling();
    searched[N0] = {mu, n_fill};
    // The bracketing search stops on the width of the mu interval, so it also
    // "converges" on a discontinuity of n(mu) (e.g. two solutions of different
    // character at essentially the same mu). Say so: the sector comparison below
    // is then made at a mu that does not give the target filling.
    if(std::abs(n_fill - p.nel_target) > 1e-3) {
      std::ostringstream w;
      w << std::fixed << std::setprecision(6)
        << "WARNING     : the mu search in sector " << sec << " ended at n = "
        << n_fill << " electrons per orbital, not the target " << p.nel_target
        << " (mu = " << mu
        << "): n(mu) is discontinuous there, or the search stopped early.\n";
      out << w.str();
    }

    // ---- scan the neighbouring sectors at mu
    impurity_params<N> saved = p;
    rebuild_active<N>(p);  // the solver leaves the active integrals rotated

    SectorContext<N> ctx{
        &p,
        Pristine<N>{p.T_active, p.V_active, p.Td_active, saved.asci_settings,
                    saved.just_singles},
        opt,
        cs.seed_parents > 0 ? cs.seed_parents : p.asci_settings.ncdets_max,
        std::min<size_t>(cs.seed_size > 0 ? cs.seed_size
                                          : p.asci_settings.ntdets_max,
                         p.asci_settings.ntdets_max),
        nrots_input,
        &out};
    ctx.use_ed = use_ed;
    ctx.beta_heavy = beta_heavy;

    if(root) fs::create_directories(opt.workdir);
    barrier();

    out << "\nSECTOR SCAN at mu = " << std::setprecision(10) << mu
        << " (charge_sectors: margin = " << cs.margin << ", etol = "
        << std::scientific << std::setprecision(2) << cs.etol << std::fixed
        << std::setprecision(10) << ", " << (warm ? "warm" : "cold")
        << " neighbours, " << (use_ed ? "ED" : "ASCI") << ")" << std::endl;

    SectorScan<N> scan(ctx, N0, Nmax);
    {
      // Solver side files (active_ordm.dat, rot_matrix*.dat) of the
      // neighbours land in the scan directory, not next to the accepted
      // solution's.
      if(chdir(opt.workdir.c_str()) != 0)
        throw std::runtime_error("cannot chdir into " + opt.workdir);
      try {
        if(p.cheap_mode and !use_ed) {
          // The mu search's last solve was a cheap one (fixed determinants):
          // less accurate than the ASCI neighbours it is compared with.
          out << "note        : cheap mode: the current sector is re-solved "
                 "with full ASCI for the comparison\n";
          scan.solve_reference(SeedSource<N>{});
        } else {
          SectorResult<N> ref;
          ref.na = saved.nalpha;
          ref.nb = saved.nbeta;
          ref.E = saved.E;
          ref.converged = std::isfinite(saved.E);
          ref.status = ref.converged ? "OK" : "NONFINITE";
          ref.seed = "mu search";
          ref.ndets = saved.dets.size();
          ref.n_band = n_fill;
          if(!use_ed) {
            ref.dets = saved.dets;
            ref.C = saved.C;
            if(nrots_input > 0) ref.U = saved.orb_rot;
          }
          scan.set(N0, std::move(ref));
        }
        scan.walk(long(cs.margin));
      } catch(...) {
        if(chdir(cwd0.c_str()) != 0) {}
        throw;
      }
      if(chdir(cwd0.c_str()) != 0)
        throw std::runtime_error("cannot chdir back into " + cwd0.string());
    }

    const auto mopt = scan.argmin();
    if(!mopt or !scan.res.count(N0) or !scan.res.at(N0).converged)
      throw std::runtime_error("sector search: the sector " + sec +
                               " has no converged energy at mu = " +
                               std::to_string(mu));
    const size_t m = *mopt;
    const double Emin = scan.res.at(m).E;
    const double E_N = scan.res.at(N0).E;

    std::ostringstream table;
    write_header(table, "#");
    for(auto& [n, r] : scan.res) write_row<N>(table, n, r, Emin);
    out << "\n" << std::string(90, '=') << "\nCHARGE SECTOR SCAN at mu = "
        << std::setprecision(10) << mu << "\n";
    {
      std::ostringstream t2;
      write_header(t2, "");
      for(auto& [n, r] : scan.res) write_row<N>(t2, n, r, Emin);
      out << t2.str() << std::endl;
    }
    for(auto& [n, r] : scan.res)
      if(!r.converged)
        out << "WARNING     : N = " << n << " " << r.status
            << "; it is left out of the comparison\n";

    const bool do_switch = m != N0 and (E_N - Emin) > cs.etol;
    if(!do_switch) {
      for(auto& [n, r] : scan.res)
        if(n != N0 and r.converged and std::abs(r.E - E_N) < cs.etol)
          out << "WARNING     : N = " << n << " lies within etol = "
              << std::scientific << std::setprecision(2) << cs.etol
              << std::fixed << std::setprecision(10)
              << " of the accepted sector (E - E_N = " << r.E - E_N
              << "): the sector is not resolved at this accuracy\n";
      out << "sector      : N = " << N0 << " " << sec
          << " is the ground-state sector (lowest among the scanned, "
             "switches = " << switches << ")\n";
      p = saved;
      std::ostringstream extra;
      extra << "# sector search: on, margin = " << cs.margin
            << ", etol = " << cs.etol << ", switches = " << switches << "\n"
            << "# scan at the final mu (E(CI) = Omega):\n" << table.str();
      write_ground_sector_file<N>(gs_file, p, mu, extra.str());
      out << std::setprecision(12) << std::scientific
          << "GROUND_SECTOR NALPHA = " << p.nalpha << " NBETA = " << p.nbeta
          << " N = " << N0 << " E = " << p.E << " MU = " << mu << std::endl;
      return mu;
    }

    // ---- another sector is lower: switch
    const auto& r = scan.res.at(m);
    out << "sector      : " << sector_str(r.na, r.nb) << " (N = " << m
        << ") lies below " << sec << " by " << E_N - Emin << " at mu = " << mu
        << "; switching\n";
    if(searched.count(m)) {
      const auto& v = searched.at(m);
      std::ostringstream msg;
      msg << std::setprecision(10)
          << "the target filling n = " << p.nel_target
          << " electrons per orbital lies in a jump of the ground-state "
             "filling: at mu = " << mu << " the sector N = " << N0
          << " reaches it but N = " << m << " is lower (by " << E_N - Emin
          << "), while in N = " << m << " the mu search (mu = " << v.mu
          << ", n = " << v.n << ") ends where N = " << N0
          << " (or another sector) is lower again. No ground state has this "
             "filling. Crossing between sectors is not handled.";
      throw std::runtime_error(msg.str());
    }
    if(++switches > cs.max_switch)
      throw std::runtime_error(
          "sector search: more than DOP.SECTOR_MAX_SWITCH = " +
          std::to_string(cs.max_switch) + " sector switches");

    p = saved;
    p.nalpha = r.na;
    p.nbeta = r.nb;
    p.ci_exp = ci_orig;
    p.asci_wfn_fname.clear();  // ASCI.WFN_FILE belongs to the input sector
    p.compute_asci_E0 = true;
    p.asci_E0 = 0.0;
    // The sector's solution at this mu can start the next mu search, but only
    // in the original orbital basis: load_asci_guess needs NROTS = 0 and the
    // integrals the mu search builds are unrotated.
    if(warm and nrots_input == 0 and r.U.empty() and !r.dets.empty()) {
      const std::string gfile = opt.workdir + "/guess_Na" +
                                std::to_string(r.na) + "_Nb" +
                                std::to_string(r.nb) + ".wfn";
      if(root) macis::write_wavefunction(gfile, p.n_active, r.dets, r.C);
      barrier();
      p.asci_wfn_fname = gfile;
      p.compute_asci_E0 = false;
      p.asci_E0 = r.E;
      guess_active = true;
      out << "guess       : " << gfile << "\n";
    }
    mu_start = mu;
  }
}

// Explicit template instantiations
namespace charge_sectors {
template class SectorScan<64>;
template void rebuild_active<64>(impurity_params<64>&);
template void rotate_active<64>(impurity_params<64>&, const basis_t&);
template SectorResult<64> solve_sector<64>(SectorContext<64>&, size_t, size_t,
                                           const SeedSource<64>&);
template void write_row<64>(std::ostream&, size_t, const SectorResult<64>&,
                            double);
}  // namespace charge_sectors
template void write_ground_sector_file<64>(const std::string&,
                                           const impurity_params<64>&, double,
                                           const std::string&);
template double Fix_Mu_sectors<64>(const std::string&, bool, double&,
                                   impurity_params<64>*,
                                   const ChargeSectorSettings&);

}  // namespace macis
