// Find the charge sector holding the ground state of an impurity problem, in
// one process.
//
// The FCIDUMP carries -mu on the impurity diagonal, so E(CI) of a sector
// already is the grand potential Omega(N) at that mu. Every sector here is
// solved with the same integrals at fixed mu (no mu search), so the energies
// compare directly and the lowest one is the ground state. Omega(N) need not be
// convex (it can have several local minima), so the search keeps going until
// the minimum has --margin higher sectors on each side (--search walk), or
// solves a fixed window around the reference (--search window).
//
// Usage: explore_charge_sectors {It_N/ASCI.tar.gz | It_N/ | ASCI/ | input.in}
//            [--workdir DIR] [--warm-start [--warm-nrots0]]
//            [--search walk|window] [--margin M]
//            [--window W]
//            [--nalpha A --nbeta B | --half] [--scale x] [--seed-parents P]
//            [--seed-size S]
//            [--check-spin] [--etol T]
//
// Input: an archived iteration (ASCI.tar.gz is unpacked into --workdir, default
// ./sector_scan_It<N>), an ASCI/ directory or a plain input.in. If
// locFCIDUMP.dat sits next to input.in, its impurity one-body block (the mu the
// search ENDED on) replaces the one in CI.FCIDUMP (the mu it started from).
// DOPING and GF are ignored: the scan is at fixed mu.
//
// --warm-start: every new sector is seeded from its solved neighbour by one
// c^dagger / c on the neighbour's leading determinants, diagonalized in that
// seed space and handed to the production ASCI path as a guess wavefunction.
// The seeded sector is solved in the neighbour's orbital basis: its integrals
// are rotated by the neighbour's cumulative orbital rotation, and it runs with
// NROTS = 0 in that basis. A sector started cold (the reference, or a fallback)
// uses the input's NROTS, so with NROTS > 0 every sector ends up in the
// natural-orbital basis of the cold sector it descends from. --warm-nrots0
// instead forces NROTS = 0 everywhere (all sectors in the original basis).
// Without --warm-start every sector is solved cold with the input's NROTS.
//
// Only the minimal-|S_z| sector ((N+1)/2, N/2) is solved per N: with SU(2) it
// contains every multiplet. --check-spin also solves (a+1, b-1) at the final N
// to flag a high-spin ground state. Solver side files (active_ordm.dat,
// rot_matrix*.dat, seed_*.wfn) go to --workdir.
//
// Output: a table on stdout and <workdir>/sector_scan.dat, verdict lines, and
//   GROUND_SECTOR NALPHA = a NBETA = b N = n E = e

#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <macis/impurity_solver.hpp>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "../tests/ini_input.hpp"

namespace fs = std::filesystem;

constexpr size_t nwfn_bits = 64;
using wfn_type = macis::wfn_t<nwfn_bits>;
using params_t = macis::impurity_params<nwfn_bits>;

std::map<std::string, CIExpansion> ci_exp_map = {
    {"CAS", CIExpansion::CAS},
    {"ASCI", CIExpansion::ASCI},
    {"ASCI_cheap", CIExpansion::ASCI_cheap}};

namespace {

int world_rank = 0;
bool is_root = true;

void barrier() { MACIS_MPI_CODE(MPI_Barrier(MPI_COMM_WORLD);) }

struct Options {
  std::string source, workdir;
  std::string search = "walk";
  long margin = 2, window = 2;
  bool warm = false, warm_nrots0 = false, check_spin = false, half = false;
  long nalpha = -1, nbeta = -1;
  double scale = 1.0, etol = 1e-4;
  long seed_parents = -1, seed_size = -1;
};

const char* usage =
    "usage: explore_charge_sectors {It_N/ASCI.tar.gz | It_N/ | ASCI/ | "
    "input.in} "
    "[--workdir DIR] [--warm-start [--warm-nrots0]] [--search walk|window] "
    "[--margin M] "
    "[--window W] "
    "[--nalpha A --nbeta B | --half] [--scale x] [--seed-parents P] "
    "[--seed-size S] "
    "[--check-spin] [--etol T]";

Options parse_args(int argc, char** argv) {
  Options o;
  auto need = [&](int& i) -> std::string {
    if(i + 1 >= argc)
      throw std::runtime_error(std::string(argv[i]) + " needs a value");
    return argv[++i];
  };
  for(int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if(a == "--workdir")
      o.workdir = need(i);
    else if(a == "--warm-start")
      o.warm = true;
    else if(a == "--warm-nrots0")
      o.warm = o.warm_nrots0 = true;
    else if(a == "--search")
      o.search = need(i);
    else if(a == "--margin")
      o.margin = std::stol(need(i));
    else if(a == "--window")
      o.window = std::stol(need(i));
    else if(a == "--nalpha")
      o.nalpha = std::stol(need(i));
    else if(a == "--nbeta")
      o.nbeta = std::stol(need(i));
    else if(a == "--half")
      o.half = true;
    else if(a == "--scale")
      o.scale = std::stod(need(i));
    else if(a == "--seed-parents")
      o.seed_parents = std::stol(need(i));
    else if(a == "--seed-size")
      o.seed_size = std::stol(need(i));
    else if(a == "--check-spin")
      o.check_spin = true;
    else if(a == "--etol")
      o.etol = std::stod(need(i));
    else if(a == "-h" or a == "--help")
      throw std::runtime_error(usage);
    else if(a.rfind("--", 0) == 0)
      throw std::runtime_error("unknown option " + a);
    else if(o.source.empty())
      o.source = a;
    else
      throw std::runtime_error("unexpected argument " + a);
  }
  if(o.source.empty()) throw std::runtime_error(usage);
  if(o.search != "walk" and o.search != "window")
    throw std::runtime_error("--search must be walk or window");
  if(o.margin < 1) throw std::runtime_error("--margin must be >= 1");
  if(o.window < 0) throw std::runtime_error("--window must be >= 0");
  if((o.nalpha < 0) != (o.nbeta < 0))
    throw std::runtime_error("--nalpha and --nbeta go together");
  if(o.half and o.nalpha >= 0)
    throw std::runtime_error("--half and --nalpha/--nbeta are exclusive");
  if(!(o.scale > 0)) throw std::runtime_error("--scale must be > 0");
  if(o.seed_parents == 0 or o.seed_size == 0)
    throw std::runtime_error("--seed-parents and --seed-size must be >= 1");
  return o;
}

// ---- input resolution (helpers as in charge_sector_estimate.cxx)
// ------------------------

bool file_exists(const std::string& f) { return std::ifstream(f).good(); }

bool looks_like_ini(const std::string& fname) {
  std::ifstream f(fname);
  if(!f) throw std::runtime_error("cannot open " + fname);
  std::string line;
  while(std::getline(f, line)) {
    auto i = line.find_first_not_of(" \t");
    if(i != std::string::npos and line[i] == '[') return true;
  }
  return false;
}

std::string dir_of(const std::string& f) {
  auto i = f.rfind('/');
  return i == std::string::npos ? "." : f.substr(0, i);
}

// Resolve a path written in input.in the way the solver would see it (cwd = its
// directory).
std::string resolve(const std::string& path, const std::string& ini_dir,
                    std::ostream& out) {
  if(path.empty() or path[0] != '/') return ini_dir + "/" + path;
  if(file_exists(path)) return path;
  std::string local = ini_dir + "/" + path.substr(path.rfind('/') + 1);
  if(!file_exists(local))
    throw std::runtime_error(path + " does not exist, and neither does " +
                             local);
  out << "note        : " << path << " does not exist; using " << local << "\n";
  return local;
}

bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() and
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// "It_7" -> "7", searched over the path components of @p p (last match wins).
std::string iteration_tag(const fs::path& p) {
  std::string tag;
  for(const auto& c : fs::absolute(p).lexically_normal()) {
    auto s = c.string();
    if(s.rfind("It_", 0) == 0 and s.size() > 3) tag = s.substr(3);
  }
  return tag;
}

std::string shell_quote(const std::string& s) {
  std::string r = "'";
  for(char c : s) r += (c == '\'') ? std::string("'\\''") : std::string(1, c);
  return r + "'";
}

// Returns the absolute path of input.in, unpacking a tarball into @p workdir if
// needed. Also fills in a default workdir.
std::string resolve_source(Options& o, std::ostream& out) {
  fs::path src = fs::absolute(o.source).lexically_normal();
  if(!fs::exists(src)) throw std::runtime_error(o.source + " does not exist");

  fs::path tarball;
  fs::path ini;
  if(fs::is_directory(src)) {
    if(fs::exists(src / "ASCI.tar.gz"))
      tarball = src / "ASCI.tar.gz";
    else if(fs::exists(src / "input.in"))
      ini = src / "input.in";
    else if(fs::exists(src / "ASCI" / "input.in"))
      ini = src / "ASCI" / "input.in";
    else
      throw std::runtime_error(
          o.source + " holds neither ASCI.tar.gz, input.in nor ASCI/input.in");
  } else if(ends_with(src.string(), ".tar.gz") or
            ends_with(src.string(), ".tgz")) {
    tarball = src;
  } else {
    if(!looks_like_ini(src.string()))
      throw std::runtime_error(o.source +
                               " is not an input.in (no [SECTION] headers)");
    ini = src;
  }

  if(o.workdir.empty()) {
    auto tag = iteration_tag(src);
    o.workdir = tag.empty() ? "./sector_scan" : "./sector_scan_It" + tag;
  }
  o.workdir = fs::absolute(o.workdir).lexically_normal().string();
  if(is_root) fs::create_directories(o.workdir);

  if(!tarball.empty()) {
    out << "archive     : " << tarball.string() << "\n";
    out << "unpacking   : into " << o.workdir << "\n";
    int rc = 0;
    if(is_root) {
      std::string cmd = "tar -xzf " + shell_quote(tarball.string()) + " -C " +
                        shell_quote(o.workdir);
      rc = std::system(cmd.c_str());
    }
    MACIS_MPI_CODE(MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);)
    if(rc != 0)
      throw std::runtime_error("tar failed to unpack " + tarball.string());
    barrier();
    fs::path w(o.workdir);
    if(fs::exists(w / "ASCI" / "input.in"))
      ini = w / "ASCI" / "input.in";
    else if(fs::exists(w / "input.in"))
      ini = w / "input.in";
    else
      throw std::runtime_error(
          tarball.string() +
          " unpacked, but neither ASCI/input.in nor input.in appeared");
  }
  barrier();
  return ini.string();
}

// ---- small utilities
// --------------------------------------------------------------------

size_t split_alpha(size_t N) { return (N + 1) / 2; }
size_t split_beta(size_t N) { return N / 2; }

std::string sector_str(size_t a, size_t b) {
  return "(" + std::to_string(a) + "," + std::to_string(b) + ")";
}

// Gather a Davidson eigenvector distributed as in selected_ci_diag (block rows,
// remainder on the last rank) into a full, replicated vector. Same scheme as
// asci_iter.
std::vector<double> gather_C(std::vector<double> C_local, size_t ndets) {
#ifdef MACIS_ENABLE_MPI
  auto world_size = macis::comm_size(MPI_COMM_WORLD);
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

// ---- seeds
// ------------------------------------------------------------------------------

using weight_map_t =
    std::map<wfn_type, double, macis::bitset_less_comparator<nwfn_bits>>;

// Apply c^dagger_{i,spin} (create) or c_{i,spin} to the top @p nparents
// determinants of (dets, C) (ranked by |C|) for every active orbital i that
// allows it. Each image collects sqrt(sum C_parent^2) over the parents that
// reach it. Only the ranking is used afterwards (Davidson recomputes C), so
// fermionic signs are dropped: summing signed amplitudes could cancel an image
// that two different i reach.
weight_map_t apply_ladder(const std::vector<wfn_type>& dets,
                          const std::vector<double>& C, int spin, bool create,
                          size_t nparents, size_t n_active) {
  std::vector<size_t> order(dets.size());
  std::iota(order.begin(), order.end(), 0);
  nparents = std::min(nparents, dets.size());
  std::partial_sort(
      order.begin(), order.begin() + nparents, order.end(),
      [&](size_t x, size_t y) { return std::abs(C[x]) > std::abs(C[y]); });

  const size_t off = spin ? nwfn_bits / 2 : 0;
  weight_map_t w;
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

std::pair<std::vector<wfn_type>, std::vector<double>> unzip(
    const weight_map_t& w) {
  std::vector<wfn_type> d;
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
weight_map_t ladder_to(const std::vector<wfn_type>& dets,
                       const std::vector<double>& C, size_t pa, size_t pb,
                       size_t ta, size_t tb, size_t nparents, size_t n_active) {
  if(pa == ta and pb == tb) {
    weight_map_t w;
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
    auto w = apply_ladder(dets, C, 1, create, nparents, n_active);
    auto [d, c] = unzip(w);
    return ladder_to(d, c, pa, tb, ta, tb, pa == ta ? nparents : d.size(),
                     n_active);
  }
  const bool create = ta > pa;
  if((create ? ta - pa : pa - ta) != 1)
    throw std::logic_error("ladder_to: alpha count changes by more than one");
  return apply_ladder(dets, C, 0, create, nparents, n_active);
}

// Keep the @p nkeep heaviest determinants, and check they all sit in (na, nb).
std::vector<wfn_type> truncate_seed(const weight_map_t& w, size_t nkeep,
                                    size_t na, size_t nb) {
  std::vector<std::pair<double, wfn_type>> v;
  v.reserve(w.size());
  for(auto& [d, x] : w) v.push_back({x, d});
  nkeep = std::min(nkeep, v.size());
  // Ties broken on the bitset so every rank keeps the same determinants.
  std::partial_sort(v.begin(), v.begin() + nkeep, v.end(),
                    [](const auto& x, const auto& y) {
                      if(x.first != y.first) return x.first > y.first;
                      return macis::bitset_less(x.second, y.second);
                    });
  std::vector<wfn_type> dets(nkeep);
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

// ---- solving one sector
// -----------------------------------------------------------------

struct Pristine {
  std::vector<double> T_active, V_active, Td_active;
  macis::ASCISettings asci_settings;
  bool just_singles;
};

// An orbital basis, as the cumulative rotation from the original active
// orbitals (n_active x n_active, column-major, the convention of
// impurity_params::orb_rot). Empty means the original orbitals.
using basis_t = std::vector<double>;

// Where a sector's starting wavefunction comes from.
struct SeedSource {
  enum Kind { Cold, Parent, File } kind = Cold;
  std::string label = "cold";
  const std::vector<wfn_type>* dets =
      nullptr;  // Parent: the parent's wavefunction
  const std::vector<double>* C = nullptr;
  size_t pa = 0, pb = 0;
  std::string fname;  // File
  // Orbital basis the seed determinants are written in (Parent, File)
  const basis_t* U = nullptr;
};

struct SectorResult {
  size_t na = 0, nb = 0;
  bool converged = false;
  std::string status, seed;
  double E = NAN, E_seed = NAN, n_band = NAN, time_s = 0;
  size_t ndets = 0;
  std::vector<wfn_type>
      dets;  // kept while the sector may still seed a neighbour
  std::vector<double> C;
  basis_t U;  // orbital basis dets/C are written in
};

struct Context {
  params_t* p;
  Pristine pristine;
  Options opt;
  size_t seed_parents, seed_size;
  size_t nrots_cold;  // NROTS of a solve that starts cold
  std::ostream* out;
};

void restore(Context& ctx) {
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

// Rotate the active integrals in place into the basis @p U, the way the
// solver's macro iterations and the GF path do (T <- U^T T U, same for Td and
// all four indices of V). The rotation mixes impurity orbitals among
// themselves, so the interaction is no longer density-density: singles-only
// must be turned off, and it must be turned off in p, because
// SolveImpurityASCI_rot re-applies p.just_singles to its own generator.
void rotate_active(params_t& p, const basis_t& U) {
  macis::SDBuildHamiltonianGenerator<nwfn_bits> ham_gen(
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

// Read a rot_matrix.dat written by SolveImpurityASCI_rot (row i holds
// orb_rot(i, 0..n-1)) and check it is a rotation that keeps impurity and bath
// apart. Returns an empty basis and sets @p why if it is not usable.
basis_t read_rot_matrix(const std::string& fname, size_t n, size_t n_imp,
                        std::string& why) {
  std::ifstream f(fname);
  if(!f) {
    why = "cannot open " + fname;
    return {};
  }
  std::vector<double> v;
  double x;
  while(f >> x) v.push_back(x);
  if(v.size() != n * n) {
    why = fname + " holds " + std::to_string(v.size()) + " numbers, expected " +
          std::to_string(n * n);
    return {};
  }
  basis_t U(n * n);
  for(size_t i = 0; i < n; ++i)
    for(size_t j = 0; j < n; ++j) U[i + j * n] = v[i * n + j];
  double orth = 0, mix = 0;
  for(size_t i = 0; i < n; ++i)
    for(size_t j = 0; j < n; ++j) {
      double d = 0;
      for(size_t k = 0; k < n; ++k) d += U[k + i * n] * U[k + j * n];
      orth = std::max(orth, std::abs(d - (i == j ? 1.0 : 0.0)));
      if((i < n_imp) != (j < n_imp))
        mix = std::max(mix, std::abs(U[i + j * n]));
    }
  if(orth > 1e-8) {
    std::ostringstream o;
    o << fname << " is not orthogonal (max |U^T U - 1| = " << orth << ")";
    why = o.str();
    return {};
  }
  if(mix > 1e-10) {
    std::ostringstream o;
    o << fname << " mixes impurity and bath orbitals (max |U_ib| = " << mix
      << ")";
    why = o.str();
    return {};
  }
  return U;
}

// Diagonalize in the seed space and hand the result to SolveImpurityASCI_rot as
// a guess. Returns E_seed (total energy).
double solve_from_seed(Context& ctx, std::vector<wfn_type> dets,
                       const std::string& fname) {
  auto& p = *ctx.p;
  macis::SDBuildHamiltonianGenerator<nwfn_bits> ham_gen(
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

  if(is_root) macis::write_wavefunction(fname, p.n_active, dets, C);
  barrier();

  p.asci_wfn_fname = fname;
  p.compute_asci_E0 = false;
  p.asci_E0 = E_seed;
  return E_seed;
}

SectorResult solve_sector_once(Context& ctx, size_t na, size_t nb,
                               const SeedSource& src) {
  auto& p = *ctx.p;
  auto& out = *ctx.out;
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();

  SectorResult r;
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
    // NROTS > 0); a cold one with the NROTS of the mode.
    const bool seeded = src.kind != SeedSource::Cold;
    const size_t nrots = seeded ? 0 : ctx.nrots_cold;
    p.asci_settings.nrots = nrots;
    if(seeded and src.U and !src.U->empty()) {
      rotate_active(p, *src.U);
      out << "basis       : natural orbitals inherited from " << src.label
          << " (NROTS = 0 in that basis)" << std::endl;
    }

    if(seeded) {
      std::vector<wfn_type> seed;
      if(src.kind == SeedSource::Parent) {
        auto w = ladder_to(*src.dets, *src.C, src.pa, src.pb, na, nb,
                           ctx.seed_parents, p.n_active);
        seed = truncate_seed(w, ctx.seed_size, na, nb);
      } else {
        std::vector<wfn_type> d;
        std::vector<double> c;
        macis::read_wavefunction(src.fname, d, c, true);
        weight_map_t w;
        for(size_t i = 0; i < d.size(); ++i) w[d[i]] += c[i] * c[i];
        seed = truncate_seed(w, ctx.seed_size, na, nb);
      }
      if(seed.empty())
        throw std::runtime_error(
            "empty seed: no determinant of the parent maps into " +
            sector_str(na, nb));
      const std::string fname = ctx.opt.workdir + "/seed_Na" +
                                std::to_string(na) + "_Nb" +
                                std::to_string(nb) + ".wfn";
      r.E_seed = solve_from_seed(ctx, std::move(seed), fname);
      out << "seed        : " << p.asci_wfn_fname << "  E_seed = " << std::fixed
          << std::setprecision(10) << r.E_seed << std::endl;
    }

    r.E = macis::SolveImpurityASCI_rot<nwfn_bits>(p);
    r.converged = std::isfinite(r.E);
    r.status = r.converged ? "OK" : "NONFINITE";
    r.ndets = p.dets.size();
    double s = 0;
    for(size_t i = 0; i < p.n_imp; ++i) s += p.occs[i];
    r.n_band = 2.0 * s / p.n_imp;
    if(r.converged) {
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
    else if(nrots > 0)
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

// A seeded solve that fails for any reason other than refinement not converging
// (e.g. the seed already holds NTDETS_MAX determinants and the ASCI search
// cannot reproduce that many, which asci_refine refuses) is retried cold, so
// the scan does not lose the sector.
SectorResult solve_sector(Context& ctx, size_t na, size_t nb,
                          const SeedSource& src) {
  auto r = solve_sector_once(ctx, na, nb, src);
  if(src.kind == SeedSource::Cold or r.converged or r.status == "UNCONVERGED")
    return r;
  *ctx.out << "retrying " << sector_str(na, nb)
           << " cold after the seeded solve failed" << std::endl;
  SeedSource cold;
  cold.label = "cold(seed failed)";
  auto rc = solve_sector_once(ctx, na, nb, cold);
  rc.E_seed = r.E_seed;
  rc.time_s += r.time_s;
  return rc;
}

// ---- search
// -----------------------------------------------------------------------------

class Scan {
 public:
  Scan(Context& ctx, size_t N0, size_t Nmax)
      : ctx_(ctx), N0_(N0), Nmax_(Nmax) {}

  std::map<size_t, SectorResult> res;

  bool in_range(long N) const { return N >= 0 and N <= long(Nmax_); }

  // Solve N from the solved neighbour @p Np (or cold).
  void solve(size_t N, std::optional<size_t> Np) {
    const size_t a = split_alpha(N), b = split_beta(N);
    SeedSource src;
    if(ctx_.opt.warm and Np and res.count(*Np) and !res.at(*Np).dets.empty()) {
      const auto& par = res.at(*Np);
      src.kind = SeedSource::Parent;
      src.dets = &par.dets;
      src.C = &par.C;
      src.pa = par.na;
      src.pb = par.nb;
      src.U = &par.U;
      src.label = sector_str(par.na, par.nb);
    } else if(ctx_.opt.warm and Np) {
      src.label = "cold(parent failed)";
    }
    res[N] = solve_sector(ctx_, a, b, src);
    prune();
  }

  void solve_reference(const SeedSource& src) {
    res[N0_] = solve_sector(ctx_, split_alpha(N0_), split_beta(N0_), src);
  }

  // Lowest converged N.
  std::optional<size_t> argmin() const {
    std::optional<size_t> m;
    for(auto& [N, r] : res)
      if(r.converged and (!m or r.E < res.at(*m).E)) m = N;
    return m;
  }

  size_t lo() const { return res.begin()->first; }
  size_t hi() const { return res.rbegin()->first; }

  void walk(long margin) {
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

  void window(long W) {
    for(long d = 1; d <= W; ++d) {
      if(in_range(long(N0_) - d)) solve(N0_ - d, N0_ - d + 1);
      if(in_range(long(N0_) + d)) solve(N0_ + d, N0_ + d - 1);
    }
  }

 private:
  // Wavefunctions are needed only by sectors that can still seed a neighbour:
  // the two ends of the solved range, and the current minimum (for
  // --check-spin).
  void prune() {
    auto m = argmin();
    for(auto& [N, r] : res)
      if(N != lo() and N != hi() and (!m or N != *m)) {
        r.dets = {};
        r.C = {};
      }
  }

  Context& ctx_;
  size_t N0_, Nmax_;
};

// ---- reporting
// --------------------------------------------------------------------------

void write_row(std::ostream& os, size_t N, const SectorResult& r, double Emin) {
  auto num = [&](double x, int prec) {
    std::ostringstream s;
    if(std::isfinite(x))
      s << std::fixed << std::setprecision(prec) << x;
    else
      s << "-";
    return s.str();
  };
  os << std::setw(4) << N << std::setw(8) << r.na << std::setw(7) << r.nb
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

}  // namespace

int main(int argc, char** argv) {
  MACIS_MPI_CODE(MPI_Init(&argc, &argv);)
  MACIS_MPI_CODE(world_rank = macis::comm_rank(MPI_COMM_WORLD);)
  is_root = world_rank == 0;
  int rc = 0;
  {
    std::ostream null_stream(nullptr);
    std::ostream& out = is_root ? std::cout : null_stream;
    spdlog::cfg::load_env_levels();
    spdlog::set_pattern("[%n] %v");
    auto console = world_rank
                       ? spdlog::null_logger_mt("explore_charge_sectors")
                       : spdlog::stdout_color_mt("explore_charge_sectors");
    try {
      auto opt = parse_args(argc, argv);

      // ---- 1. input resolution
      // -------------------------------------------------------
      const std::string input_file = resolve_source(opt, out);
      const std::string ini_dir = dir_of(input_file);
      out << "input.in    : " << input_file << "\n";
      out << "workdir     : " << opt.workdir << "\n";
      INIFile input(input_file);

      params_t params;
      auto& p = params;

#define OPT_KEYWORD(STR, RES, DTYPE) \
  if(input.containsData(STR)) {      \
    RES = input.getData<DTYPE>(STR); \
  }

      // ---- 2. setup, copied from run_asci_impsolv_dop.cxx
      // ---------------------------- Required Keywords
      auto fcidump_fname =
          resolve(input.getData<std::string>("CI.FCIDUMP"), ini_dir, out);
      p.nalpha = input.getData<size_t>("CI.NALPHA");
      p.nbeta = input.getData<size_t>("CI.NBETA");

      // Read FCIDUMP File
      p.norb = macis::read_fcidump_norb(fcidump_fname);
      size_t norb2 = p.norb * p.norb;
      size_t norb4 = norb2 * norb2;

      p.T.resize(norb2);
      p.V.resize(norb4);
      p.E_core = macis::read_fcidump_core(fcidump_fname);
      macis::read_fcidump_1body(fcidump_fname, p.T.data(), p.norb);
      macis::read_fcidump_2body(fcidump_fname, p.V.data(), p.norb);
      p.just_singles = macis::is_2body_diagonal(fcidump_fname);

      bool just_singles_ = true;
      OPT_KEYWORD("CI.JUST_SINGLES", just_singles_, bool);
      if(not just_singles_) p.just_singles = just_singles_;

      // Possibility of hoppings for the spin-down orbitals
      std::string fcidump_do_fname = "NONE";
      p.Td.resize(norb2);
      p.spin_dep = false;
      OPT_KEYWORD("CI.FCIDUMP_DO", fcidump_do_fname, std::string);
      if(fcidump_do_fname != "NONE") {
        fcidump_do_fname = resolve(fcidump_do_fname, ini_dir, out);
        macis::read_fcidump_1body(fcidump_do_fname, p.Td.data(), p.norb);
        p.spin_dep = true;
      }

      // Set up job
      std::string ciexp_str = "ASCI";
      OPT_KEYWORD("CI.EXPANSION", ciexp_str, std::string);
      try {
        p.ci_exp = ci_exp_map.at(ciexp_str);
      } catch(...) {
        throw std::runtime_error("CI Expansion Not Recognized");
      }
      if(p.ci_exp != CIExpansion::ASCI)
        out << "note        : CI.EXPANSION = " << ciexp_str
            << " ignored; every sector is solved with ASCI\n";
      p.ci_exp = CIExpansion::ASCI;

      // Set up active space
      p.n_inactive = 0;
      OPT_KEYWORD("CI.NINACTIVE", p.n_inactive, size_t);
      if(p.n_inactive >= p.norb) throw std::runtime_error("NINACTIVE >= NORB");

      p.n_active = p.norb - p.n_inactive;
      OPT_KEYWORD("CI.NACTIVE", p.n_active, size_t);

      if(p.n_inactive + p.n_active > p.norb)
        throw std::runtime_error("NINACTIVE + NACTIVE > NORB");

      p.n_imp = p.norb;
      OPT_KEYWORD("CI.NIMP", p.n_imp, size_t);

      p.nbands = 1;
      OPT_KEYWORD("CI.NBANDS", p.nbands, size_t);

      if(p.n_active > nwfn_bits / 2)
        throw std::runtime_error("Not Enough Bits");

      // MCSCF Settings
      OPT_KEYWORD("MCSCF.MAX_MACRO_ITER", p.mcscf_settings.max_macro_iter,
                  size_t);
      OPT_KEYWORD("MCSCF.MAX_ORB_STEP", p.mcscf_settings.max_orbital_step,
                  double);
      OPT_KEYWORD("MCSCF.MCSCF_ORB_TOL", p.mcscf_settings.orb_grad_tol_mcscf,
                  double);
      OPT_KEYWORD("MCSCF.ENABLE_DIIS", p.mcscf_settings.enable_diis, bool);
      OPT_KEYWORD("MCSCF.DIIS_START_ITER", p.mcscf_settings.diis_start_iter,
                  size_t);
      OPT_KEYWORD("MCSCF.DIIS_NKEEP", p.mcscf_settings.diis_nkeep, size_t);
      OPT_KEYWORD("MCSCF.CI_RES_TOL", p.mcscf_settings.ci_res_tol, double);
      OPT_KEYWORD("MCSCF.CI_MAX_SUB", p.mcscf_settings.ci_max_subspace, size_t);
      OPT_KEYWORD("MCSCF.CI_MATEL_TOL", p.mcscf_settings.ci_matel_tol, double);
      OPT_KEYWORD("MCSCF.CI_NSTATES", p.mcscf_settings.ci_nstates, size_t);

      // ASCI Settings
      std::string asci_wfn_out_fname;
      p.asci_E0 = 0.0;
      p.compute_asci_E0 = true;
      OPT_KEYWORD("ASCI.NTDETS_MAX", p.asci_settings.ntdets_max, size_t);
      OPT_KEYWORD("ASCI.NTDETS_MIN", p.asci_settings.ntdets_min, size_t);
      OPT_KEYWORD("ASCI.NCDETS_MAX", p.asci_settings.ncdets_max, size_t);
      OPT_KEYWORD("ASCI.HAM_EL_TOL", p.asci_settings.h_el_tol, double);
      OPT_KEYWORD("ASCI.RV_PRUNE_TOL", p.asci_settings.rv_prune_tol, double);
      OPT_KEYWORD("ASCI.PAIR_MAX_LIM", p.asci_settings.pair_size_max, size_t);
      OPT_KEYWORD("ASCI.GROW_FACTOR", p.asci_settings.grow_factor, int);
      OPT_KEYWORD("ASCI.MAX_REFINE_ITER", p.asci_settings.max_refine_iter,
                  size_t);
      OPT_KEYWORD("ASCI.REFINE_ETOL", p.asci_settings.refine_energy_tol,
                  double);
      OPT_KEYWORD("ASCI.GROW_WITH_ROT", p.asci_settings.grow_with_rot, bool);
      OPT_KEYWORD("ASCI.NROTS", p.asci_settings.nrots, size_t);
      OPT_KEYWORD("ASCI.ROT_SIZE_START", p.asci_settings.rot_size_start,
                  size_t);
      OPT_KEYWORD("ASCI.CONSTRAINT_LVL", p.asci_settings.constraint_level, int);
      OPT_KEYWORD("ASCI.SYMMETRIZE_DETS", p.asci_settings.symmetrize_dets,
                  bool);
      OPT_KEYWORD("ASCI.SYM_TOL", p.asci_settings.sym_tol, double);
      if(p.asci_settings.symmetrize_dets) {
        size_t nperm = 0;
        OPT_KEYWORD("ASCI.SYM_NPERM", nperm, size_t);
        if(nperm == 0)
          throw std::runtime_error(
              "SYMMETRIZE_DETS=TRUE requires SYM_NPERM >= 1");
        auto gens = std::make_shared<std::vector<std::vector<uint32_t>>>();
        for(size_t i = 1; i <= nperm; ++i) {
          std::string key = "ASCI.SYM_PERM_" + std::to_string(i);
          if(!input.containsData(key))
            throw std::runtime_error("Missing " + key);
          auto v = input.getData<std::vector<int>>(key);
          gens->emplace_back(v.begin(), v.end());
        }
        p.asci_settings.sym_group = gens;
      }
      p.asci_settings.just_singles = p.just_singles;
      // WFN_FILE / E0_WFN seed the production run from the previous DMFT
      // iteration, in the reference sector only. Here every sector sets its own
      // guess (or none), so they are read to be reported and then dropped.
      OPT_KEYWORD("ASCI.WFN_FILE", p.asci_wfn_fname, std::string);
      OPT_KEYWORD("ASCI.WFN_OUT_FILE", asci_wfn_out_fname, std::string);
      if(p.asci_wfn_fname.size())
        out << "note        : ASCI.WFN_FILE = " << p.asci_wfn_fname
            << " ignored; each sector sets its own starting wavefunction\n";
      p.asci_wfn_fname.clear();

      const size_t nrots_input = p.asci_settings.nrots;
      if(opt.warm and p.asci_settings.max_refine_iter == 0)
        throw std::runtime_error(
            "--warm-start requires ASCI.MAX_REFINE_ITER > 0: a seed that "
            "already holds "
            "NTDETS_MAX determinants skips asci_grow, and refinement is then "
            "the only stage "
            "that improves it");

      bool doping = false, testGF = false;
      OPT_KEYWORD("CI.DOPING", doping, bool);
      OPT_KEYWORD("CI.GF", testGF, bool);
      if(doping or testGF)
        out << "note        : CI.DOPING / CI.GF ignored: the scan is at fixed "
               "mu\n";

      // --scale: cheaper screening runs
      if(opt.scale != 1.0) {
        auto sc = [&](size_t x) {
          return std::max<size_t>(1,
                                  size_t(std::llround(double(x) * opt.scale)));
        };
        auto& s = p.asci_settings;
        out << "scale       : x" << opt.scale << "  NTDETS_MAX " << s.ntdets_max
            << " -> " << sc(s.ntdets_max) << ", NCDETS_MAX " << s.ncdets_max
            << " -> " << sc(s.ncdets_max) << "\n";
        s.ntdets_max = sc(s.ntdets_max);
        s.ncdets_max = sc(s.ncdets_max);
        s.ntdets_min = std::min(s.ntdets_min, s.ntdets_max);
      }

      // Setup printing
      bool print_davidson = true, print_ci = true, print_mcscf = true,
           print_diis = true, print_asci_search = true;
      OPT_KEYWORD("PRINT.DAVIDSON", print_davidson, bool);
      OPT_KEYWORD("PRINT.CI", print_ci, bool);
      OPT_KEYWORD("PRINT.MCSCF", print_mcscf, bool);
      OPT_KEYWORD("PRINT.DIIS", print_diis, bool);
      OPT_KEYWORD("PRINT.ASCI_SEARCH", print_asci_search, bool);
      if(not print_davidson) spdlog::null_logger_mt("davidson");
      if(not print_ci) spdlog::null_logger_mt("ci_solver");
      if(not print_mcscf) spdlog::null_logger_mt("mcscf");
      if(not print_diis) spdlog::null_logger_mt("diis");
      if(not print_asci_search) spdlog::null_logger_mt("asci_search");

      // ---- final mu from locFCIDUMP.dat
      // --------------------------------------------
      auto print_mu = [&](const char* tag) {
        out << tag << std::fixed << std::setprecision(8);
        for(size_t i = 0; i < p.n_imp; ++i) out << " " << p.T[i * p.norb + i];
        out << "\n";
      };
      print_mu("impurity diag (= -mu [+ CFS]) from CI.FCIDUMP :");
      const std::string loc_fname = ini_dir + "/locFCIDUMP.dat";
      if(file_exists(loc_fname)) {
        if(p.n_inactive != 0)
          throw std::runtime_error(
              "locFCIDUMP.dat overlay needs NINACTIVE = 0: it assumes the "
              "impurity orbitals "
              "are the leading indices of T");
        const size_t nloc = macis::read_fcidump_norb(loc_fname);
        if(nloc != p.n_imp)
          throw std::runtime_error(
              loc_fname + " has " + std::to_string(nloc) +
              " orbitals, but CI.NIMP = " + std::to_string(p.n_imp));
        // read_fcidump_1body writes only the entries present in the file, i.e.
        // the n_imp x n_imp impurity block of the LDT = norb matrix.
        macis::read_fcidump_1body(loc_fname, p.T.data(), p.norb);
        // The mu search shifts the impurity diagonal of both spin channels by
        // the same amount (set_impurity_diagonal), so do the same to Td.
        if(p.spin_dep)
          for(size_t i = 0; i < p.n_imp; ++i)
            p.Td[i * p.norb + i] = p.T[i * p.norb + i];
        out << "mu overlay  : " << loc_fname << "\n";
        print_mu("impurity diag (= -mu [+ CFS]) final           :");
      } else {
        out << "mu overlay  : none (no locFCIDUMP.dat next to input.in); using "
               "CI.FCIDUMP's mu\n";
      }

      if(is_root) {
        console->info("[Wavefunction Data]:");
        console->info("  * FCIDUMP = {}", fcidump_fname);
        console->info("ECORE = {:.12f}", p.E_core);
        console->info("TMEM   = {:.2e} GiB", macis::to_gib(p.T));
        console->info("VMEM   = {:.2e} GiB", macis::to_gib(p.V));
      }

      p.occs.resize(p.n_active, 0);
      p.orb_rot.resize(p.n_active * p.n_active);
      for(size_t i = 0; i < p.n_active; ++i)
        p.orb_rot[i * p.n_active + i] = 1.0;
      p.E = 0.0;

      // Copy integrals into active subsets
      p.T_active.resize(p.n_active * p.n_active);
      p.Td_active.resize(p.n_active * p.n_active);
      p.V_active.resize(p.n_active * p.n_active * p.n_active * p.n_active);
      p.F_inactive.resize(norb2);
      p.Fd_inactive.resize(norb2);

      macis::active_hamiltonian(
          NumOrbital(p.norb), NumActive(p.n_active), NumInactive(p.n_inactive),
          p.T.data(), p.norb, p.V.data(), p.norb, p.F_inactive.data(), p.norb,
          p.T_active.data(), p.n_active, p.V_active.data(), p.n_active);
      if(p.spin_dep)
        macis::active_hamiltonian(
            NumOrbital(p.norb), NumActive(p.n_active),
            NumInactive(p.n_inactive), p.Td.data(), p.norb, p.V.data(), p.norb,
            p.Fd_inactive.data(), p.norb, p.Td_active.data(), p.n_active,
            p.V_active.data(), p.n_active);

      p.E_inactive =
          macis::inactive_energy(NumInactive(p.n_inactive), p.T.data(), p.norb,
                                 p.F_inactive.data(), p.norb);
      if(p.spin_dep) {
        for(size_t ii = 0; ii < p.n_inactive; ii++)
          p.E_inactive +=
              p.Td[ii * (1 + p.n_inactive)] - p.T[ii * (1 + p.n_inactive)];
      }
      console->info("E(inactive) = {:.12f}", p.E_inactive);

      // ---- reference sector
      // ----------------------------------------------------------
      size_t N0;
      if(opt.half)
        N0 = p.n_active;
      else if(opt.nalpha >= 0)
        N0 = opt.nalpha + opt.nbeta;
      else
        N0 = p.nalpha + p.nbeta;
      const size_t Nmax = 2 * p.n_active;
      if(N0 > Nmax)
        throw std::runtime_error("reference N = " + std::to_string(N0) +
                                 " exceeds 2*NACTIVE");
      const size_t a0 = split_alpha(N0), b0 = split_beta(N0);
      out << "reference   : N = " << N0 << ", solved as " << sector_str(a0, b0);
      if(!opt.half and opt.nalpha < 0 and (p.nalpha != a0 or p.nbeta != b0))
        out << " (input.in asks for " << sector_str(p.nalpha, p.nbeta)
            << "; only minimal |S_z| is scanned)";
      out << "\n";
      if(p.spin_dep)
        out << "note        : spin-dependent input (CI.FCIDUMP_DO): the +-S_z "
               "mirrors are not "
               "equivalent; only the minimal-|S_z| sector ((N+1)/2, N/2) is "
               "scanned\n";
      const size_t nrots_cold = opt.warm_nrots0 ? 0 : nrots_input;
      const bool inherit_basis = opt.warm and nrots_cold > 0;
      out << "mode        : "
          << (!opt.warm         ? "cold"
              : opt.warm_nrots0 ? "warm start (NROTS forced to 0)"
              : inherit_basis   ? "warm start (seeded sectors inherit the "
                                  "parent's natural orbitals, NROTS = 0 "
                                  "there; cold starts use NROTS(input))"
                                : "warm start (NROTS = 0 in the input)")
          << ", search = " << opt.search
          << (opt.search == "walk" ? ", margin = " + std::to_string(opt.margin)
                                   : ", window = " + std::to_string(opt.window))
          << ", NROTS(input) = " << nrots_input << "\n";

      Context ctx{
          &p,
          Pristine{p.T_active, p.V_active, p.Td_active, p.asci_settings,
                   p.just_singles},
          opt,
          size_t(opt.seed_parents > 0 ? opt.seed_parents
                                      : long(p.asci_settings.ncdets_max)),
          size_t(
              opt.seed_size > 0
                  ? std::min<size_t>(opt.seed_size, p.asci_settings.ntdets_max)
                  : p.asci_settings.ntdets_max),
          nrots_cold,
          &out};
      if(opt.warm)
        out << "seeds       : top " << ctx.seed_parents
            << " parent determinants, keep " << ctx.seed_size << "\n";
      if(inherit_basis)
        out << "note        : a seeded sector is solved in the natural-orbital "
               "basis of the cold sector it descends from. That basis is "
               "optimal for that sector only, which lowers its E slightly "
               "relative to the others (typically ~1e-5 Ha, the size of the "
               "NROTS gain). Near-degenerate sectors (--etol warning): confirm "
               "with a cold run.\n";

      // Reference wavefunction from the archive, warm mode only. With NROTS > 0
      // it is written in the natural-orbital basis stored next to it in
      // rot_matrix.dat (both come from the solver's last call).
      SeedSource ref_src;
      basis_t ref_U;
      if(opt.warm) {
        const std::string wfn_name =
            asci_wfn_out_fname.size()
                ? fs::path(asci_wfn_out_fname).filename().string()
                : "wfn.out";
        const std::string wfn_file = ini_dir + "/" + wfn_name;
        const std::string rot_file = ini_dir + "/rot_matrix.dat";
        std::string why;
        if(!file_exists(wfn_file))
          why = "no " + wfn_name + " next to input.in";
        else if(nrots_input > 0 and opt.warm_nrots0)
          why = "input.in has NROTS = " + std::to_string(nrots_input) +
                ", so " + wfn_name +
                " is in a natural-orbital basis, and --warm-nrots0 keeps "
                "every sector in the original one";
        else if(nrots_input > 0 and !file_exists(rot_file))
          why = "input.in has NROTS = " + std::to_string(nrots_input) +
                ", so " + wfn_name +
                " is in a natural-orbital basis, and there is no "
                "rot_matrix.dat next to input.in to say which";
        else if(nrots_input > 0 and
                (ref_U = read_rot_matrix(rot_file, p.n_active, p.n_imp, why))
                    .empty())
          ;  // why set by read_rot_matrix
        else {
          std::vector<wfn_type> d;
          std::vector<double> c;
          auto h = macis::read_wavefunction(wfn_file, d, c, true);
          if(h.norb != p.n_active)
            why = wfn_name + " has " + std::to_string(h.norb) +
                  " orbitals, NACTIVE = " + std::to_string(p.n_active);
          else if(h.nalpha != a0 or h.nbeta != b0)
            why = wfn_name + " is in " + sector_str(h.nalpha, h.nbeta) +
                  ", the reference is " + sector_str(a0, b0);
        }
        if(why.empty()) {
          ref_src.kind = SeedSource::File;
          ref_src.fname = wfn_file;
          ref_src.label = "wfn.out";
          ref_src.U = &ref_U;
          out << "reference wfn: reusing " << wfn_file;
          if(!ref_U.empty()) out << " in the basis of " << rot_file;
          out << "\n";
        } else {
          out << "reference wfn: solved from scratch (" << why << ")\n";
        }
      }

      // Solver side files (active_ordm.dat, rot_matrix*.dat) land in the
      // workdir.
      if(chdir(opt.workdir.c_str()) != 0)
        throw std::runtime_error("cannot chdir into " + opt.workdir);

      // ---- search
      // --------------------------------------------------------------------
      Scan scan(ctx, N0, Nmax);
      scan.solve_reference(ref_src);
      if(opt.search == "walk")
        scan.walk(opt.margin);
      else
        scan.window(opt.window);

      auto mopt = scan.argmin();
      if(!mopt) throw std::runtime_error("no sector converged");
      const size_t m = *mopt;
      const double Emin = scan.res.at(m).E;

      // --check-spin
      std::optional<SectorResult> spin_res;
      if(opt.check_spin) {
        const auto& g = scan.res.at(m);
        if(g.nb >= 1 and g.na + 1 <= p.n_active) {
          SeedSource src;
          if(opt.warm and !g.dets.empty()) {
            src.kind = SeedSource::Parent;
            src.dets = &g.dets;
            src.C = &g.C;
            src.pa = g.na;
            src.pb = g.nb;
            src.U = &g.U;
            src.label = sector_str(g.na, g.nb);
          }
          spin_res = solve_sector(ctx, g.na + 1, g.nb - 1, src);
        } else {
          out << "note        : --check-spin: no (a+1, b-1) sector at N = " << m
              << "\n";
        }
      }

      // ---- output
      // --------------------------------------------------------------------
      if(is_root) {
        std::ofstream dat(opt.workdir + "/sector_scan.dat");
        dat << "# explore_charge_sectors  input = " << input_file
            << "  mode = " << (opt.warm ? "warm" : "cold")
            << "  search = " << opt.search << "\n";
        write_header(dat, "#");
        out << "\n" << std::string(90, '=') << "\nCHARGE SECTOR SCAN\n";
        write_header(out, "");
        for(auto& [N, r] : scan.res) {
          write_row(out, N, r, Emin);
          write_row(dat, N, r, Emin);
        }
        if(spin_res) {
          dat << "# --check-spin at N = " << m << "\n#";
          write_row(dat, m, *spin_res, Emin);
          out << "check-spin:\n";
          write_row(out, m, *spin_res, Emin);
        }
        out << "\n";

        const auto& g = scan.res.at(m);
        auto conv = [&](long N) -> const SectorResult* {
          if(N < 0 or !scan.res.count(N)) return nullptr;
          const auto& r = scan.res.at(N);
          return r.converged ? &r : nullptr;
        };
        out << std::fixed << std::setprecision(10);
        out << "minimum     : N = " << m << " " << sector_str(g.na, g.nb)
            << "  E = " << g.E << "\n";

        const bool lo_ok = m == 0 or conv(long(m) - 1);
        const bool hi_ok = m == Nmax or conv(long(m) + 1);
        out << "bracketed   : " << (lo_ok and hi_ok ? "yes" : "NO")
            << (m == 0 ? " (N = 0 is the lower bound)" : "")
            << (m == Nmax ? " (N = 2*NACTIVE is the upper bound)" : "") << "\n";
        out << "gaps        :";
        if(auto r = conv(long(m) - 1))
          out << "  Omega(N-1)-Omega(N) = " << r->E - g.E;
        if(auto r = conv(long(m) + 1))
          out << "  Omega(N+1)-Omega(N) = " << r->E - g.E;
        out << "\n";

        std::vector<size_t> minima;
        for(auto& [N, r] : scan.res) {
          if(!r.converged) continue;
          auto l = conv(long(N) - 1), h = conv(long(N) + 1);
          if(!l and !h) continue;
          if((!l or r.E < l->E) and (!h or r.E < h->E)) minima.push_back(N);
        }
        out << "local minima: ";
        for(auto N : minima) out << N << " ";
        out << (minima.size() > 1
                    ? " -> Omega(N) is NOT convex over the scanned range"
                    : "")
            << "\n";

        bool near = false;
        for(auto& [N, r] : scan.res)
          if(N != m and r.converged and r.E - Emin < opt.etol) {
            out << "WARNING     : N = " << N
                << " lies within --etol = " << std::scientific
                << std::setprecision(2) << opt.etol << std::fixed
                << std::setprecision(10)
                << " of the minimum (E - E_min = " << r.E - Emin
                << "): the sector is not resolved at this accuracy\n";
            near = true;
          }
        if(!near) out << "degeneracy  : no other sector within --etol\n";

        for(auto& [N, r] : scan.res)
          if(!r.converged)
            out << "WARNING     : N = " << N << " " << r.status
                << "; it is left out of the minimum\n";

        if(auto r = conv(long(N0))) {
          if(m == N0)
            out << "reference   : N = " << N0
                << " IS the ground-state sector\n";
          else
            out << "reference   : N = " << N0
                << " is NOT the ground-state sector (E - E_min = "
                << r->E - Emin << ")\n";
        } else {
          out << "reference   : N = " << N0 << " did not converge\n";
        }

        if(spin_res) {
          if(spin_res->converged) {
            const double d = spin_res->E - g.E;
            out << "check-spin  : " << sector_str(spin_res->na, spin_res->nb)
                << " E - E_min = " << d
                << (d < opt.etol
                        ? "  -> high-spin: the ground state has S > |S_z|min"
                        : "  -> minimal-|S_z| ground state")
                << "\n";
          } else {
            out << "check-spin  : " << sector_str(spin_res->na, spin_res->nb)
                << " " << spin_res->status << "\n";
          }
        }

        out << "results     : " << opt.workdir << "/sector_scan.dat\n";
        out << std::setprecision(12);
        out << "GROUND_SECTOR NALPHA = " << g.na << " NBETA = " << g.nb
            << " N = " << m << " E = " << g.E << std::endl;
      }
    } catch(const std::exception& e) {
      if(is_root) std::cerr << "ERROR: " << e.what() << std::endl;
      rc = 1;
    }
  }
  MACIS_MPI_CODE(MPI_Finalize();)
  return rc;
}
