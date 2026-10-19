// Estimate the (NALPHA, NBETA) sector holding the ground state of an impurity FCIDUMP.
//
// The FCIDUMP carries -mu on the impurity diagonal and bath levels measured from mu, so it
// already defines H - mu*N: its E(N) are compared directly across sectors (subtracting mu*N
// again would double-count mu). Two exactly solvable limits bracket the answer:
//   1. U = 0 : one-body diagonalization; sector = negative single-particle levels per spin.
//   2. V = 0 : bath decoupled (its negative levels) + ED of the isolated interacting impurity
//              over every (n_up, n_dn).
// plus the heuristic N_est = N(U=0) - n_imp(U=0) + N_imp(atomic). None is exact --
// hybridization typically moves the answer by one -- so the suggested sectors are meant to be
// confirmed with ASCI, keeping the lowest E(CI). --exact runs ED of the full problem.
//
// Usage: charge_sector_estimate {FCIDUMP.dat | input.in} [--nimp N] [--fcidump-do FILE]
//                               [--nalpha A --nbeta B] [--window W] [--exact]
//                               [--exact-dim-max D] [--degen-tol T]
// An input.in (recognized by its [SECTION] headers) supplies CI.FCIDUMP, CI.NALPHA, CI.NBETA,
// CI.NIMP and CI.FCIDUMP_DO; command-line flags override it. Relative paths in it are taken
// from its own directory (the solver's cwd); a stale absolute path falls back to the same
// file name next to input.in.
// For an archived iteration pass locFCIDUMP's mu by hand: FCIDUMP.dat in ASCI.tar.gz holds
// the mu the solver STARTED from, not the one the mu search ended on.

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <lapack.hh>
#include <limits>
#include <map>
#include <macis/hamiltonian_generator/sd_build.hpp>
#include <macis/sd_operations.hpp>
#include <macis/solvers/selected_ci_diag.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/util/mpi.hpp>
#include <set>
#include <string>
#include <vector>

#include "../tests/ini_input.hpp"

constexpr size_t nwfn_bits = 64;
constexpr size_t DENSE_MAX = 2000;

using sector_t = std::pair<size_t, size_t>;
using energies_t = std::map<sector_t, double>;

namespace {

bool is_root = true;

struct Options {
  std::string source, fcidump, fcidump_do;
  long nimp = -1, nalpha = -1, nbeta = -1, window = 1;
  bool exact = false;
  double exact_dim_max = 2e6, degen_tol = 1e-8;
};

Options parse_args(int argc, char** argv) {
  Options o;
  auto need = [&](int& i) -> std::string {
    if(i + 1 >= argc) throw std::runtime_error(std::string(argv[i]) + " needs a value");
    return argv[++i];
  };
  for(int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if(a == "--nimp") o.nimp = std::stol(need(i));
    else if(a == "--fcidump-do") o.fcidump_do = need(i);
    else if(a == "--nalpha") o.nalpha = std::stol(need(i));
    else if(a == "--nbeta") o.nbeta = std::stol(need(i));
    else if(a == "--window") o.window = std::stol(need(i));
    else if(a == "--exact") o.exact = true;
    else if(a == "--exact-dim-max") o.exact_dim_max = std::stod(need(i));
    else if(a == "--degen-tol") o.degen_tol = std::stod(need(i));
    else if(a.rfind("--", 0) == 0) throw std::runtime_error("unknown option " + a);
    else if(o.source.empty()) o.source = a;
    else throw std::runtime_error("unexpected argument " + a);
  }
  if(o.source.empty())
    throw std::runtime_error(
        "usage: charge_sector_estimate {FCIDUMP.dat | input.in} [--nimp N] [--fcidump-do FILE] "
        "[--nalpha A --nbeta B] [--window W] [--exact] [--exact-dim-max D] "
        "[--degen-tol T]");
  return o;
}

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

// Resolve a path written in input.in the way the solver would see it (cwd = its directory).
std::string resolve(const std::string& path, const std::string& ini_dir, std::ostream& out) {
  if(path.empty() or path[0] != '/') return ini_dir + "/" + path;
  if(file_exists(path)) return path;
  std::string local = ini_dir + "/" + path.substr(path.rfind('/') + 1);
  if(!file_exists(local))
    throw std::runtime_error(path + " does not exist, and neither does " + local);
  out << "note        : " << path << " does not exist; using " << local << "\n";
  return local;
}

// Fill whatever the command line left unset from input.in.
void apply_input_in(Options& o, std::ostream& out) {
  INIFile input(o.source);
  const std::string dir = dir_of(o.source);
  if(!input.containsData("CI.FCIDUMP"))
    throw std::runtime_error(o.source + " has no CI.FCIDUMP");
  o.fcidump = resolve(input.getData<std::string>("CI.FCIDUMP"), dir, out);
  if(o.fcidump_do.empty() and input.containsData("CI.FCIDUMP_DO")) {
    auto f = input.getData<std::string>("CI.FCIDUMP_DO");
    if(f != "NONE") o.fcidump_do = resolve(f, dir, out);
  }
  if(o.nimp < 0 and input.containsData("CI.NIMP"))
    o.nimp = input.getData<size_t>("CI.NIMP");
  if(o.nalpha < 0 and o.nbeta < 0 and input.containsData("CI.NALPHA") and
     input.containsData("CI.NBETA")) {
    o.nalpha = input.getData<size_t>("CI.NALPHA");
    o.nbeta = input.getData<size_t>("CI.NBETA");
  }
}

inline size_t idx4(size_t p, size_t q, size_t r, size_t s, size_t n) {
  return p + n * (q + n * (r + n * s));
}

// Integrals restricted to orbitals [0, n) of a norb-orbital problem, column-major.
struct Integrals {
  size_t n;
  std::vector<double> Tu, Td, V;
};

Integrals leading_block(const std::vector<double>& Tu, const std::vector<double>& Td,
                        const std::vector<double>& V, size_t norb, size_t n) {
  Integrals b{n, std::vector<double>(n * n), std::vector<double>(n * n),
              std::vector<double>(n * n * n * n)};
  for(size_t q = 0; q < n; ++q)
    for(size_t p = 0; p < n; ++p) {
      b.Tu[p + n * q] = Tu[p + norb * q];
      b.Td[p + n * q] = Td[p + norb * q];
    }
  for(size_t s = 0; s < n; ++s)
    for(size_t r = 0; r < n; ++r)
      for(size_t q = 0; q < n; ++q)
        for(size_t p = 0; p < n; ++p)
          b.V[idx4(p, q, r, s, n)] = V[idx4(p, q, r, s, norb)];
  return b;
}

double binom(size_t n, size_t k) {
  if(k > n) return 0;
  double r = 1;
  for(size_t i = 1; i <= k; ++i) r = r * double(n - k + i) / double(i);
  return r;
}

// Lowest eigenvalue of the (nu, nd) sector, without E_core.
double sector_ground(Integrals& I, size_t nu, size_t nd) {
  using generator_t = macis::SDBuildHamiltonianGenerator<nwfn_bits>;
  const size_t n = I.n;
  generator_t ham_gen(macis::matrix_span<double>(I.Tu.data(), n, n),
                      macis::rank4_span<double>(I.V.data(), n, n, n, n));
  ham_gen.ReadTdo(macis::matrix_span<double>(I.Td.data(), n, n));

  auto dets = macis::generate_hilbert_space<nwfn_bits>(n, nu, nd);
  const size_t dim = dets.size();
  if(dim == 0) return std::numeric_limits<double>::infinity();

  if(dim <= DENSE_MAX) {
    std::vector<double> H(dim * dim), W(dim);
    for(size_t j = 0; j < dim; ++j)
      for(size_t i = 0; i <= j; ++i)
        H[i + dim * j] = H[j + dim * i] = ham_gen.matrix_element(dets[i], dets[j]);
    lapack::syev(lapack::Job::NoVec, lapack::Uplo::Lower, dim, H.data(), dim, W.data());
    return W[0];
  }

  std::vector<double> C;
  return macis::selected_ci_diag(dets.begin(), dets.end(), ham_gen, 1e-16, 100, 1e-8,
                                 C MACIS_MPI_CODE(, MPI_COMM_WORLD), true);
}

// Returns E per sector. With SU(2) symmetry (spin_sym) only nu >= nd is solved.
energies_t scan(Integrals& I, const std::vector<sector_t>& sectors, bool spin_sym) {
  energies_t E;
  for(auto [u, d] : sectors) {
    if(spin_sym and u < d) continue;
    E[{u, d}] = sector_ground(I, u, d);
    if(spin_sym) E[{d, u}] = E[{u, d}];
  }
  return E;
}

std::map<size_t, double> min_over_sz(const energies_t& E) {
  std::map<size_t, double> EN;
  for(auto& [s, e] : E) {
    auto N = s.first + s.second;
    EN[N] = EN.count(N) ? std::min(EN[N], e) : e;
  }
  return EN;
}

std::pair<double, std::vector<sector_t>> lowest(const energies_t& E, double tol) {
  double e0 = std::numeric_limits<double>::infinity();
  for(auto& [s, e] : E) e0 = std::min(e0, e);
  std::vector<sector_t> gs;
  for(auto& [s, e] : E)
    if(e - e0 < tol) gs.push_back(s);
  return {e0, gs};
}

std::string fmt_sectors(const std::vector<sector_t>& v) {
  std::string s;
  for(size_t i = 0; i < v.size(); ++i)
    s += (i ? ", (" : "(") + std::to_string(v[i].first) + "," +
         std::to_string(v[i].second) + ")";
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  MACIS_MPI_CODE(MPI_Init(&argc, &argv);)
  MACIS_MPI_CODE(is_root = macis::comm_rank(MPI_COMM_WORLD) == 0;)
  spdlog::set_level(spdlog::level::warn);
  int rc = 0;
  try {
    auto o = parse_args(argc, argv);
    std::ostream null_stream(nullptr);
    std::ostream& out = is_root ? std::cout : null_stream;
    out << std::fixed;
    if(looks_like_ini(o.source)) {
      out << "input.in    : " << o.source << "\n";
      apply_input_in(o, out);
    } else {
      o.fcidump = o.source;
    }
    if((o.nalpha < 0) != (o.nbeta < 0))
      throw std::runtime_error("--nalpha and --nbeta go together");

    // --- read ----------------------------------------------------------------------------
    const size_t norb = macis::read_fcidump_norb(o.fcidump);
    if(norb > nwfn_bits / 2)
      throw std::runtime_error("norb = " + std::to_string(norb) + " exceeds " +
                               std::to_string(nwfn_bits / 2) + " (nwfn_bits / 2)");
    std::vector<double> Tu(norb * norb, 0.), V(norb * norb * norb * norb, 0.);
    const double ecore = macis::read_fcidump_core(o.fcidump);
    macis::read_fcidump_1body(o.fcidump, Tu.data(), norb);
    macis::read_fcidump_2body(o.fcidump, V.data(), norb);
    std::vector<double> Td = Tu;
    const bool spin_sym = o.fcidump_do.empty();
    if(!spin_sym) {
      if(macis::read_fcidump_norb(o.fcidump_do) > norb)
        throw std::runtime_error(o.fcidump_do + " has more orbitals than " + o.fcidump);
      std::fill(Td.begin(), Td.end(), 0.);
      macis::read_fcidump_1body(o.fcidump_do, Td.data(), norb);
    }

    long max_2b = -1;
    for(size_t s = 0; s < norb; ++s)
      for(size_t r = 0; r < norb; ++r)
        for(size_t q = 0; q < norb; ++q)
          for(size_t p = 0; p < norb; ++p)
            if(std::abs(V[idx4(p, q, r, s, norb)]) > 1e-14)
              max_2b = std::max<long>(max_2b, std::max({p, q, r, s}));
    const size_t nimp = o.nimp > 0 ? o.nimp : (max_2b >= 0 ? max_2b + 1 : 0);
    if(nimp == 0 or nimp > norb)
      throw std::runtime_error("cannot determine the impurity orbitals: pass --nimp");
    if(max_2b >= long(nimp))
      throw std::runtime_error(
          "two-body integrals touch orbital " + std::to_string(max_2b + 1) + " > nimp = " +
          std::to_string(nimp) + ": the bath is not non-interacting, the V = 0 limit is wrong");
    const size_t nbath = norb - nimp;

    out << "source      : " << o.fcidump << "\n";
    out << "orbitals    : " << norb << " total = " << nimp << " impurity + " << nbath
        << " bath   (E_core = " << std::setprecision(6) << ecore << ")\n";
    out << "impurity diag (= -mu [+ CFS]):" << std::setprecision(8);
    for(size_t i = 0; i < nimp; ++i) out << " " << Tu[i * norb + i];
    out << "\n";
    if(!spin_sym) out << "spin-dependent one-body: down channel from " << o.fcidump_do << "\n";
    if(o.nalpha >= 0)
      out << "requested   : NALPHA = " << o.nalpha << ", NBETA = " << o.nbeta
          << "   (N = " << o.nalpha + o.nbeta << ")\n";
    out << "\n";

    // --- 1. U = 0 ------------------------------------------------------------------------
    out << "=== 1. U = 0 limit (one-body diagonalization) " << std::string(42, '=') << "\n";
    size_t n0[2] = {0, 0};
    double nimp0 = 0.;
    for(int sp = 0; sp < (spin_sym ? 1 : 2); ++sp) {
      std::vector<double> A = sp ? Td : Tu, W(norb);
      lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, norb, A.data(), norb, W.data());
      size_t n = 0;
      double w_imp = 0.;
      for(size_t k = 0; k < norb; ++k)
        if(W[k] < 0) {
          ++n;
          for(size_t i = 0; i < nimp; ++i) w_imp += A[i + norb * k] * A[i + norb * k];
        }
      n0[sp] = n;
      nimp0 += w_imp;
      out << "  " << (spin_sym ? "each spin" : (sp ? "dn" : "up")) << ": " << std::setw(3)
          << n << " levels below 0; nearest to 0: " << std::showpos << std::setprecision(5)
          << (n ? W[n - 1] : NAN) << " (last occ), " << (n < norb ? W[n] : NAN)
          << " (first empty)" << std::noshowpos << "\n";
      for(size_t k = 0; k < norb; ++k)
        if(std::abs(W[k]) < 1e-6)
          out << "  WARNING: level " << W[k] << " sits at the Fermi level: N(U=0) is ambiguous\n";
    }
    if(spin_sym) {
      n0[1] = n0[0];
      nimp0 *= 2;
    }
    const size_t N0 = n0[0] + n0[1];
    out << "  -> sector (NALPHA, NBETA) = (" << n0[0] << "," << n0[1] << "),  N = " << N0
        << ";  impurity holds " << std::setprecision(4) << nimp0 << " electrons at U = 0\n\n";

    // --- 2. V = 0 ------------------------------------------------------------------------
    out << "=== 2. V = 0 limit (atomic ED + decoupled bath) " << std::string(40, '=') << "\n";
    size_t nb[2] = {0, 0};
    for(int sp = 0; sp < 2 and nbath; ++sp) {
      const auto& T = sp ? Td : Tu;
      std::vector<double> A(nbath * nbath), W(nbath);
      for(size_t q = 0; q < nbath; ++q)
        for(size_t p = 0; p < nbath; ++p) A[p + nbath * q] = T[(nimp + p) + norb * (nimp + q)];
      lapack::syev(lapack::Job::NoVec, lapack::Uplo::Lower, nbath, A.data(), nbath, W.data());
      nb[sp] = std::count_if(W.begin(), W.end(), [](double w) { return w < 0; });
    }
    out << "  bath: (" << nb[0] << "," << nb[1] << ") levels below 0\n";

    auto imp = leading_block(Tu, Td, V, norb, nimp);
    std::vector<sector_t> at_sectors;
    for(size_t u = 0; u <= nimp; ++u)
      for(size_t d = 0; d <= nimp; ++d) at_sectors.push_back({u, d});
    auto Eat = scan(imp, at_sectors, spin_sym);
    auto EatN = min_over_sz(Eat);
    auto [e_at, gs_at] = lowest(Eat, o.degen_tol);
    std::set<size_t> Nat;
    for(auto [u, d] : gs_at) Nat.insert(u + d);

    out << "  atomic E(N_imp) (min over S_z):\n";
    for(auto& [n, e] : EatN)
      out << "    N_imp = " << std::setw(2) << n << ": " << std::showpos << std::setprecision(8)
          << e << "   (dE = " << std::setprecision(5) << e - e_at << ")" << std::noshowpos
          << (Nat.count(n) ? "  <- ground state" : "") << "\n";
    out << "  atomic ground sector(s): " << fmt_sectors(gs_at) << "\n";
    if(Nat.size() > 1)
      out << "  WARNING: atomic ground state is charge-degenerate (mixed-valence point): the "
             "sector is decided by the bath, not the atom\n";
    {
      out << "  atomic charge gaps:";
      size_t lo = *Nat.begin(), hi = *Nat.rbegin();
      if(lo > 0) out << " N_imp=" << lo - 1 << ": " << std::showpos << EatN[lo - 1] - e_at;
      if(hi < 2 * nimp) out << " N_imp=" << hi + 1 << ": " << std::showpos << EatN[hi + 1] - e_at;
      out << std::noshowpos << "\n";
    }
    std::set<size_t> NV0;
    std::vector<sector_t> sV0;
    for(auto [u, d] : gs_at) {
      NV0.insert(nb[0] + nb[1] + u + d);
      sV0.push_back({nb[0] + u, nb[1] + d});
    }
    out << "  -> N(V=0) =";
    for(auto n : NV0) out << " " << n;
    out << "  (sectors " << fmt_sectors(sV0) << ")\n\n";

    // --- combined ------------------------------------------------------------------------
    out << "=== Combined estimate " << std::string(67, '=') << "\n";
    std::set<size_t> Nest;
    out << "  N_est = N(U=0) - n_imp(U=0) + N_imp(atomic) = " << N0 << " - "
        << std::setprecision(3) << nimp0 << " + N_imp =";
    for(auto n : Nat) {
      double x = N0 - nimp0 + n;
      out << " " << std::setprecision(2) << x;
      Nest.insert(size_t(std::lround(x)));
    }
    out << "  ->";
    for(auto n : Nest) out << " " << n;
    out << "\n  (heuristic: it ignores how the bath refills when the impurity empties; a value "
           "near a half-integer is a coin toss)\n";

    size_t lo = std::min(*Nest.begin(), *NV0.begin());
    size_t hi = std::max(*Nest.rbegin(), *NV0.rbegin());
    long c_lo = std::max<long>(0, long(lo) - o.window);
    long c_hi = std::min<long>(2 * norb, long(hi) + o.window);
    out << "  U = 0 limit N = " << N0
        << ((long(N0) < c_lo or long(N0) > c_hi)
                ? " (outside the window: at large U the atomic limit is the relevant one)\n"
                : " (inside the window)\n");

    auto split = [&](long N) -> std::pair<long, long> {
      if(spin_sym) return {(N + 1) / 2, N / 2};
      long u0 = sV0[0].first, d0 = sV0[0].second, diff = N - (u0 + d0);
      long half = diff >= 0 ? diff / 2 : -((-diff + 1) / 2);
      return {u0 + diff - half, d0 + half};
    };
    out << "  run ASCI in these sectors and keep the lowest E(CI) (it is the grand potential):\n";
    bool requested_in = false;
    for(long N = c_lo; N <= c_hi; ++N) {
      auto [u, d] = split(N);
      if(u < 0 or d < 0 or u > long(norb) or d > long(norb)) continue;
      std::vector<std::string> tag;
      if(Nest.count(N)) tag.push_back("combined heuristic");
      if(NV0.count(N)) tag.push_back("V=0 limit");
      if(o.nalpha >= 0 and N == o.nalpha + o.nbeta) {
        tag.push_back("currently requested N");
        requested_in = true;
      }
      out << "    NALPHA = " << std::setw(3) << u << ", NBETA = " << std::setw(3) << d
          << "   (N = " << std::setw(3) << N << ")  ";
      for(size_t i = 0; i < tag.size(); ++i) out << (i ? ", " : " ") << tag[i];
      out << "\n";
    }
    if(o.nalpha >= 0 and !requested_in)
      out << "  NOTE: the requested N = " << o.nalpha + o.nbeta
          << " lies outside every estimate above.\n";
    if(spin_sym)
      out << "  (with SU(2) symmetry the minimal-|S_z| sector contains every spin multiplet, "
             "so only N needs scanning)\n";
    out << "\n";

    // --- optional exact check -----------------------------------------------------------
    if(o.exact) {
      out << "=== Exact ED of the full problem, every sector " << std::string(42, '=') << "\n";
      double dim = binom(norb, norb / 2) * binom(norb, norb / 2);
      if(dim > o.exact_dim_max) {
        out << "  skipped: largest sector has dimension " << std::setprecision(0) << dim
            << " > --exact-dim-max = " << o.exact_dim_max << "\n";
      } else {
        Integrals full{norb, Tu, Td, V};
        std::vector<sector_t> all;
        for(size_t u = 0; u <= norb; ++u)
          for(size_t d = 0; d <= norb; ++d) all.push_back({u, d});
        auto Eex = scan(full, all, spin_sym);
        auto EexN = min_over_sz(Eex);
        auto [e0, gs] = lowest(Eex, o.degen_tol);
        long Ngs = gs[0].first + gs[0].second;
        for(auto& [n, e] : EexN)
          if(std::abs(long(n) - Ngs) <= 3)
            out << "    N = " << std::setw(3) << n << ": E = " << std::showpos
                << std::setprecision(9) << e + ecore << "   (dE = " << std::setprecision(6)
                << e - e0 << ")" << std::noshowpos << "\n";
        out << "  exact ground sector(s): " << fmt_sectors(gs) << "   E0 = " << std::showpos
            << std::setprecision(9) << e0 + ecore << std::noshowpos << "\n";
        if(o.nalpha >= 0) {
          double er = Eex.at({size_t(o.nalpha), size_t(o.nbeta)});
          out << "  requested sector (" << o.nalpha << "," << o.nbeta << "): E = "
              << std::showpos << er + ecore << "  (dE = " << std::scientific
              << std::setprecision(3) << er - e0 << ")" << std::noshowpos << std::fixed << "\n";
        }
      }
    }
  } catch(const std::exception& e) {
    if(is_root) std::cerr << "ERROR: " << e.what() << std::endl;
    rc = 1;
  }
  MACIS_MPI_CODE(MPI_Finalize();)
  return rc;
}
