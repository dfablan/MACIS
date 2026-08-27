/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <iostream>
#include <macis/asci/determinant_symmetry.hpp>
#include <macis/asci/grow.hpp>
#include <macis/asci/refine.hpp>
#include <macis/bitset_operations.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/model/hubbard.hpp>
#include <macis/sd_operations.hpp>
#include <macis/solvers/selected_ci_diag.hpp>
#include <macis/types.hpp>
#include <macis/util/mcscf.hpp>

#include "ut_common.hpp"

using wfn64 = macis::wfn_t<64>;

namespace {

wfn64 make_det(std::initializer_list<unsigned> alpha,
               std::initializer_list<unsigned> beta) {
  wfn64 w(0);
  for(auto p : alpha) w.set(p);
  for(auto p : beta) w.set(p + 32);
  return w;
}

bool contains_det(const std::vector<wfn64>& sorted_dets, const wfn64& w) {
  return std::binary_search(sorted_dets.begin(), sorted_dets.end(), w,
                            macis::bitset_less_comparator<64>{});
}

}  // namespace

TEST_CASE("Permute Orbitals") {
  const std::vector<uint32_t> id = {0, 1, 2};
  const std::vector<uint32_t> cycle = {1, 2, 0};
  const std::vector<uint32_t> swap01 = {1, 0, 2};

  // Identity: bits inside and beyond the permutation range are untouched
  auto w = make_det({0, 2, 5}, {1});
  REQUIRE(macis::permute_orbitals(w, id) == w);

  // 3-cycle moves both spin sectors together; orbital 4 >= perm.size() fixed
  w = make_det({0, 4}, {2});
  REQUIRE(macis::permute_orbitals(w, cycle) == make_det({1, 4}, {0}));

  // Multi-bit image
  w = make_det({0, 1}, {0});
  REQUIRE(macis::permute_orbitals(w, cycle) == make_det({1, 2}, {1}));

  // Involution applied twice is the identity
  w = make_det({0, 2}, {1, 2});
  REQUIRE(macis::permute_orbitals(macis::permute_orbitals(w, swap01), swap01) ==
          w);
}

TEST_CASE("Valid Permutation") {
  REQUIRE(macis::is_valid_permutation({0, 1, 2}, 3));
  REQUIRE(macis::is_valid_permutation({2, 0, 1}, 3));
  // Repeats
  REQUIRE(!macis::is_valid_permutation({0, 0, 2}, 3));
  // Out of range
  REQUIRE(!macis::is_valid_permutation({0, 1, 3}, 3));
  // Wrong length
  REQUIRE(!macis::is_valid_permutation({0, 1}, 3));
  REQUIRE(!macis::is_valid_permutation({0, 1, 2, 3}, 3));
}

TEST_CASE("Expand Permutation Group") {
  const std::vector<uint32_t> id = {0, 1, 2};
  const std::vector<uint32_t> cycle = {1, 2, 0};
  const std::vector<uint32_t> swap01 = {1, 0, 2};

  // Cyclic group C3
  auto c3 = macis::expand_permutation_group({cycle}, 3);
  REQUIRE(c3.size() == 3);
  REQUIRE(std::count(c3.begin(), c3.end(), id) == 1);

  // {3-cycle, transposition} generate S3
  auto s3 = macis::expand_permutation_group({cycle, swap01}, 3);
  REQUIRE(s3.size() == 6);
  REQUIRE(std::count(s3.begin(), s3.end(), id) == 1);

  // Deterministic
  REQUIRE(macis::expand_permutation_group({cycle, swap01}, 3) == s3);

  // Size cap throws
  REQUIRE_THROWS(macis::expand_permutation_group({cycle, swap01}, 3, 5));

  // Invalid generator throws
  REQUIRE_THROWS(macis::expand_permutation_group({{0, 0, 2}}, 3));
}

TEST_CASE("Orbit Representative") {
  const std::vector<uint32_t> cycle = {1, 2, 0};
  auto group = macis::expand_permutation_group({cycle}, 3);

  // Orbit {a0, a1, a2}: representative is the bitset_less minimum (a0)
  const auto rep = make_det({0}, {});
  REQUIRE(macis::orbit_representative(make_det({2}, {}), group) == rep);
  REQUIRE(macis::orbit_representative(make_det({1}, {}), group) == rep);
  REQUIRE(macis::orbit_representative(rep, group) == rep);

  // Fixed point maps to itself
  const auto fixed = make_det({0, 1, 2}, {});
  REQUIRE(macis::orbit_representative(fixed, group) == fixed);
}

TEST_CASE("Symmetric Orbit Select") {
  const std::vector<uint32_t> cycle = {1, 2, 0};
  auto group = macis::expand_permutation_group({cycle}, 3);

  // Orbit U (fixed point, size 1): score 0.95
  // Orbit A = {a0, a1, a2}   (size 3): only a0 present, score 0.9
  // Orbit B = {(a0,b1), (a1,b2), (a2,b0)} (size 3): two members present,
  //           a candidate (rv < 0) and a seed (rv > 0), score 0.5
  const auto detU = make_det({0, 1, 2}, {});
  const auto detA0 = make_det({0}, {});
  const auto detB0 = make_det({0}, {1});
  const auto detB1 = make_det({1}, {2});

  macis::asci_contrib_container<wfn64> pairs = {
      {detU, -0.95}, {detA0, -0.9}, {detB0, -0.5}, {detB1, 0.4}};

  // Budget 7: all three orbits fit (1 + 3 + 3); absent orbit members are
  // materialized
  {
    auto sel = macis::symmetric_orbit_select(pairs, group, 7);
    REQUIRE(sel.size() == 7);
    REQUIRE(contains_det(sel, detU));
    REQUIRE(contains_det(sel, make_det({1}, {})));
    REQUIRE(contains_det(sel, make_det({2}, {})));
    REQUIRE(contains_det(sel, make_det({2}, {0})));
  }

  // Budget 6: U (1) + A (3) fit; B (3) does not -> strict stop at 4
  {
    auto sel = macis::symmetric_orbit_select(pairs, group, 6);
    REQUIRE(sel.size() == 4);
    REQUIRE(contains_det(sel, detU));
    REQUIRE(contains_det(sel, detA0));
    REQUIRE(!contains_det(sel, detB0));
  }

  // Budget 3: U (1) fits; A (3) would overflow -> stop at 1 (no back-fill)
  {
    auto sel = macis::symmetric_orbit_select(pairs, group, 3);
    REQUIRE(sel.size() == 1);
    REQUIRE(sel[0] == detU);
  }

  // Budget 0: nothing
  REQUIRE(macis::symmetric_orbit_select(pairs, group, 0).empty());

  // Seeds (positive rv) rank via |rv|: a strong seed promotes its orbit
  {
    macis::asci_contrib_container<wfn64> pairs2 = {{detB0, 0.99},
                                                   {detA0, -0.9}};
    auto sel = macis::symmetric_orbit_select(pairs2, group, 3);
    REQUIRE(sel.size() == 3);
    REQUIRE(contains_det(sel, detB1));
    REQUIRE(!contains_det(sel, detA0));
  }

  // Duplicate input entries do not duplicate output determinants
  {
    auto pairs3 = pairs;
    pairs3.push_back({detA0, -0.9});
    REQUIRE(macis::symmetric_orbit_select(pairs3, group, 7) ==
            macis::symmetric_orbit_select(pairs, group, 7));
  }
}

TEST_CASE("ASCI Symmetric Search") {
  MACIS_MPI_CODE(MPI_Barrier(MPI_COMM_WORLD);)

  for(auto name :
      {"davidson", "ci_solver", "asci_search", "asci_grow", "asci_refine"})
    if(!spdlog::get(name)) spdlog::null_logger_mt(name);

#ifdef MACIS_ENABLE_MPI
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#else
  int world_size = 1;
#endif

  // 6-site Hubbard ring at half filling: exactly invariant under the cyclic
  // translation i -> i+1 (mod 6)
  const size_t nsites = 6;
  std::vector<double> T, V;
  macis::hubbard_1d(nsites, 1.0, 4.0, T, V, /*pbc=*/true);

  using generator_t = macis::DoubleLoopHamiltonianGenerator<64>;
  generator_t ham_gen(
      macis::matrix_span<double>(T.data(), nsites, nsites),
      macis::rank4_span<double>(V.data(), nsites, nsites, nsites, nsites));

  const uint32_t nalpha = 3, nbeta = 3;

  std::vector<uint32_t> cycle(nsites);
  for(size_t i = 0; i < nsites; ++i) cycle[i] = (i + 1) % nsites;
  auto group = std::make_shared<const std::vector<std::vector<uint32_t>>>(
      macis::expand_permutation_group({cycle}, nsites));
  REQUIRE(group->size() == 6);

  macis::MCSCFSettings mcscf_settings;
  mcscf_settings.ci_res_tol = 1e-10;
  // Davidson caps its iteration count at the subspace size (davidson.hpp:205),
  // so a 1e-10 residual is unreachable with the default 20 and the solver
  // throws instead of converging.
  mcscf_settings.ci_max_subspace = 100;

  macis::ASCISettings asci_settings;
  // Deliberately awkward budget: an unsymmetrized top-K must split orbits
  asci_settings.ntdets_max = 101;
  asci_settings.ntdets_min = 10;
  // Only 3 occupied alpha orbitals: higher constraint levels are not
  // meaningful for this system
  asci_settings.constraint_level = 0;
  // The whole-orbit budget can change the space by one orbit between refine
  // iterations; keep the energy convergence check commensurate
  asci_settings.refine_energy_tol = 1e-4;
  // The default 6 is not enough to settle once the space can move by a whole
  // orbit between iterations
  asci_settings.max_refine_iter = 30;

  auto run_asci = [&](bool symmetrize) {
    macis::ASCISettings settings = asci_settings;
    settings.symmetrize_dets = symmetrize;
    if(symmetrize) settings.sym_group = group;

    std::vector<wfn64> dets = {
        macis::canonical_hf_determinant<64>(nalpha, nbeta)};
    std::vector<double> C = {1.0};
    double E0 = ham_gen.matrix_element(dets[0], dets[0]);

    std::tie(E0, dets, C) = macis::asci_grow(
        settings, mcscf_settings, E0, std::move(dets), std::move(C), ham_gen,
        nsites MACIS_MPI_CODE(, MPI_COMM_WORLD));
    std::tie(E0, dets, C) = macis::asci_refine(
        settings, mcscf_settings, E0, std::move(dets), std::move(C), ham_gen,
        nsites MACIS_MPI_CODE(, MPI_COMM_WORLD));
    return std::make_tuple(E0, dets, C);
  };

  // FCI reference on the full 400-determinant space: every truncated energy
  // must respect the variational bound
  double E_fci;
  {
    auto fci_dets = macis::generate_hilbert_space<64>(nsites, nalpha, nbeta);
    std::vector<double> C_fci;
    E_fci = macis::selected_ci_diag<64>(
        fci_dets.begin(), fci_dets.end(), ham_gen, mcscf_settings.ci_matel_tol,
        mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol,
        C_fci MACIS_MPI_CODE(, MPI_COMM_WORLD));
  }

  auto [E_sym, dets_sym, C_sym] = run_asci(true);

  // Strict budget, but the whole-orbit deficit stays below one orbit
  REQUIRE(dets_sym.size() <= asci_settings.ntdets_max);
  REQUIRE(dets_sym.size() >= 90);
  REQUIRE(C_sym.size() == dets_sym.size());
  REQUIRE(E_sym >= E_fci - 1e-8);

  // Closure: the image of every determinant under every group element is in
  // the final set
  {
    auto sorted = dets_sym;
    std::sort(sorted.begin(), sorted.end(),
              macis::bitset_less_comparator<64>{});
    REQUIRE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
    for(const auto& d : dets_sym)
      for(const auto& g : *group)
        REQUIRE(contains_det(sorted, macis::permute_orbitals(d, g)));
  }

  // 1-RDM inherits the symmetry: in a G-closed space the projected
  // Hamiltonian commutes with the group representation
  {
    std::vector<double> ordm(nsites * nsites, 0.0),
        trdm(nsites * nsites * nsites * nsites, 0.0);
    ham_gen.form_rdms(
        dets_sym.begin(), dets_sym.end(), dets_sym.begin(), dets_sym.end(),
        C_sym.data(), macis::matrix_span<double>(ordm.data(), nsites, nsites),
        macis::rank4_span<double>(trdm.data(), nsites, nsites, nsites,
                                  nsites));
    double max_dev = 0.0;
    for(const auto& g : *group)
      for(size_t q = 0; q < nsites; ++q)
        for(size_t r = 0; r < nsites; ++r)
          max_dev = std::max(
              max_dev, std::abs(ordm[g[r] + g[q] * nsites] -
                                ordm[r + q * nsites]));
    REQUIRE(max_dev < 1e-6);
  }

  // Negative control: the plain top-K fills the budget exactly and its energy
  // sits within a bounded difference of the symmetrized one. Serial only: the
  // MPI top-K path re-injects the seed determinants on every rank and the
  // gathered list then carries duplicates -- a pre-existing defect of the
  // unsymmetrized MPI path (the whole-orbit closure is immune to it).
  if(world_size == 1) {
    auto [E_unsym, dets_unsym, C_unsym] = run_asci(false);
    REQUIRE(dets_unsym.size() == asci_settings.ntdets_max);
    REQUIRE(E_unsym >= E_fci - 1e-8);
    REQUIRE(std::abs(E_sym - E_unsym) < 0.1);
  }

  MACIS_MPI_CODE(MPI_Barrier(MPI_COMM_WORLD);)
  spdlog::drop_all();
}
