/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <lapack.hh>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/util/orbital_energies.hpp>

#include "ut_common.hpp"

TEST_CASE("FCIDUMP") {
  ROOT_ONLY(MPI_COMM_WORLD);

  SECTION("READ") {
    size_t norb_ref = 24;
    SECTION("NORB") {
      auto norb = macis::read_fcidump_norb(water_ccpvdz_fcidump);
      REQUIRE(norb == norb_ref);
    }

    SECTION("Core") {
      auto coreE = macis::read_fcidump_core(water_ccpvdz_fcidump);
      REQUIRE(coreE == Approx(9.191200742618042));
    }

    SECTION("OneBody") {
      std::vector<double> T(norb_ref * norb_ref);
      macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb_ref);
      double sum = std::accumulate(T.begin(), T.end(), 0.0);
      REQUIRE(sum == Approx(-1.095432762653e+02));
    }

    SECTION("TwoBody") {
      std::vector<double> V(norb_ref * norb_ref * norb_ref * norb_ref);
      macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb_ref);
      double sum = std::accumulate(V.begin(), V.end(), 0.0);
      REQUIRE(sum == Approx(2.701609068389e+02));
    }

    SECTION("Validity Checks") {
      auto norb = norb_ref;
      size_t nocc = 5;
      const auto norb2 = norb * norb;
      const auto norb3 = norb2 * norb;
      std::vector<double> T(norb * norb);
      std::vector<double> V(norb * norb * norb * norb);
      macis::read_fcidump_1body(water_ccpvdz_fcidump, T.data(), norb);
      macis::read_fcidump_2body(water_ccpvdz_fcidump, V.data(), norb);

      std::vector<double> eps(norb);
      macis::canonical_orbital_energies(macis::NumOrbital(norb),
                                        macis::NumInactive(nocc), T.data(),
                                        norb, V.data(), norb, eps.data());

      // Check orbital energies
      std::vector<double> ref_eps = {
          -20.5504959651472,  -1.33652308180416,  -0.699084807015877,
          -0.566535827115135, -0.493126779151562, 0.185508268977809,
          0.25620321890592,   0.78902083505165,   0.854064342891826,
          1.16354210561965,   1.20037835222836,   1.25333579878846,
          1.44452900701373,   1.4762147730592,    1.67453827449385,
          1.86734472965445,   1.93460967763627,   2.4520550848955,
          2.48955724627828,   3.28543361094139,   3.33853354817945,
          3.5101570353257,    3.86543012937399,   4.14719831587618,
      };

      for(auto i = 0ul; i < norb; ++i) REQUIRE(eps[i] == Approx(ref_eps[i]));

      // MP2
      double EMP2 = 0.;
      for(size_t i = 0; i < nocc; ++i)
        for(size_t a = nocc; a < norb; ++a)
          for(size_t j = 0; j < nocc; ++j)
            for(size_t b = nocc; b < norb; ++b) {
              double den = eps[a] + eps[b] - eps[i] - eps[j];
              double dir = V[a + i * norb + b * norb2 + j * norb3];
              double exh = V[b + i * norb + a * norb2 + j * norb3];

              EMP2 -= (dir * (2 * dir - exh)) / den;
            }
      REQUIRE(EMP2 == Approx(-0.203989305096243));
    }
  }

  // Both FCIDUMP column orderings must parse identically: "<p> <q> <r> <s>
  // <integral>" (what write_fcidump emits) and "<integral> <p> <q> <r> <s>"
  // (the Molpro/PySCF convention). The two fixtures hold the same integrals.
  SECTION("Column Ordering") {
    const size_t norb = 2;
    const size_t norb4 = norb * norb * norb * norb;

    REQUIRE(macis::read_fcidump_norb(layout_index_first_fcidump) == norb);
    REQUIRE(macis::read_fcidump_norb(layout_integral_first_fcidump) == norb);

    std::vector<double> T_idx(norb * norb), T_int(norb * norb);
    std::vector<double> V_idx(norb4), V_int(norb4);

    macis::read_fcidump_1body(layout_index_first_fcidump, T_idx.data(), norb);
    macis::read_fcidump_2body(layout_index_first_fcidump, V_idx.data(), norb);
    macis::read_fcidump_1body(layout_integral_first_fcidump, T_int.data(),
                              norb);
    macis::read_fcidump_2body(layout_integral_first_fcidump, V_int.data(),
                              norb);

    // Anchor the known-good (index-first) read so the comparison below cannot
    // be satisfied by both layouts being misparsed the same way.
    REQUIRE(macis::read_fcidump_core(layout_index_first_fcidump) ==
            Approx(0.125));
    REQUIRE(T_idx[0 + 0 * norb] == Approx(-1.25));
    REQUIRE(T_idx[0 + 1 * norb] == Approx(-0.5));
    REQUIRE(T_idx[1 + 0 * norb] == Approx(-0.5));
    REQUIRE(V_idx[0] == Approx(0.5));

    REQUIRE(macis::read_fcidump_core(layout_integral_first_fcidump) ==
            Approx(0.125));
    for(size_t i = 0; i < norb * norb; ++i)
      REQUIRE(T_int[i] == Approx(T_idx[i]));
    for(size_t i = 0; i < norb4; ++i) REQUIRE(V_int[i] == Approx(V_idx[i]));
  }

  // The (pointer, LD) overloads must honour an LD larger than the file's norb:
  // the integrals go to the leading norb block, and nothing else is touched.
  SECTION("Leading Dimension") {
    const size_t norb = 2, LD = 5;
    const double pad = 99.0;
    // The readers write only the entries listed in the file, so the reference
    // starts from the same fill value as the padded arrays.
    std::vector<double> T_ref(norb * norb, pad),
        V_ref(norb * norb * norb * norb, pad);
    macis::read_fcidump_1body(layout_index_first_fcidump, T_ref.data(), norb);
    macis::read_fcidump_2body(layout_index_first_fcidump, V_ref.data(), norb);

    std::vector<double> T(LD * LD, pad), V(LD * LD * LD * LD, pad);
    macis::read_fcidump_1body(layout_index_first_fcidump, T.data(), LD);
    macis::read_fcidump_2body(layout_index_first_fcidump, V.data(), LD);

    for(size_t q = 0; q < LD; ++q)
      for(size_t p = 0; p < LD; ++p) {
        const double ref = (p < norb and q < norb) ? T_ref[p + q * norb] : pad;
        REQUIRE(T[p + q * LD] == Approx(ref));
      }
    for(size_t s = 0; s < LD; ++s)
      for(size_t r = 0; r < LD; ++r)
        for(size_t q = 0; q < LD; ++q)
          for(size_t p = 0; p < LD; ++p) {
            const bool in = p < norb and q < norb and r < norb and s < norb;
            const double ref =
                in ? V_ref[p + norb * (q + norb * (r + norb * s))] : pad;
            REQUIRE(V[p + LD * (q + LD * (r + LD * s))] == Approx(ref));
          }

    // An LD smaller than norb cannot hold the integrals
    REQUIRE_THROWS(
        macis::read_fcidump_1body(layout_index_first_fcidump, T.data(), 1));
    REQUIRE_THROWS(
        macis::read_fcidump_2body(layout_index_first_fcidump, V.data(), 1));
  }
}

// Full CI of a three-orbital atom (64 determinants) built from the FCIDUMP
// with the MACIS Hamiltonian generator; eigenvalues sorted ascending.
static std::vector<double> atom3_spectrum(const std::string& fname) {
  const size_t norb = 3;
  std::vector<double> T(norb * norb, 0.0), V(norb * norb * norb * norb, 0.0);
  macis::read_fcidump_2body(fname, V.data(), norb);
  macis::DoubleLoopHamiltonianGenerator<64> ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));

  std::vector<macis::wfn_t<64>> dets;
  for(unsigned a = 0; a < 8; ++a)
    for(unsigned b = 0; b < 8; ++b)
      dets.push_back(macis::wfn_t<64>(a) | (macis::wfn_t<64>(b) << 32));
  const size_t ndets = dets.size();
  std::vector<double> H(ndets * ndets);
  // matrix_element assumes equal alpha and beta electron counts in bra and
  // ket: other blocks of H vanish and are left at zero.
  auto counts = [](macis::wfn_t<64> d) {
    return std::pair{(d & macis::wfn_t<64>(0xFFFFFFFFul)).count(),
                     (d >> 32).count()};
  };
  for(size_t i = 0; i < ndets; ++i)
    for(size_t j = 0; j < ndets; ++j)
      if(counts(dets[i]) == counts(dets[j]))
        H[i + j * ndets] = ham_gen.matrix_element(dets[i], dets[j]);
  for(size_t i = 0; i < ndets; ++i)
    for(size_t j = 0; j < ndets; ++j)
      REQUIRE(H[i + j * ndets] == Approx(H[j + i * ndets]).margin(1e-12));

  std::vector<double> W(ndets);
  lapack::syev(lapack::Job::NoVec, lapack::Uplo::Lower, ndets, H.data(), ndets,
               W.data());
  return W;
}

// Closed-form spectrum: {n, 2S, L or -1, degeneracy} for the three-orbital
// shell (orbital multiplets of t2g-like p orbitals).
static std::vector<double> atom3_reference(
    double U, double J, bool pair_hopping) {
  struct Multiplet { int n, twoS, L, deg; };
  const std::vector<Multiplet> ms = {
      {0, 0, 0, 1},  {1, 1, 1, 6},  {2, 2, 1, 9},  {2, 0, 2, 5}, {2, 0, 0, 1},
      {3, 3, 0, 4},  {3, 1, 2, 10}, {3, 1, 1, 6},  {4, 2, 1, 9}, {4, 0, 2, 5},
      {4, 0, 0, 1},  {5, 1, 1, 6},  {6, 0, 0, 1}};
  std::vector<double> E;
  for(auto m : ms) {
    const double n = m.n, S = 0.5 * m.twoS, L = m.L;
    // Kanamori: (U-3J) n(n-1)/2 - 2J S^2 - J/2 L^2 + 5J/2 n
    // Dworin-Narath: (U-3J/2) n(n-1)/2 - J S^2 + 3J/4 n (no L dependence)
    const double e =
        pair_hopping ? (U - 3 * J) * n * (n - 1) / 2 - 2 * J * S * (S + 1) -
                           0.5 * J * L * (L + 1) + 2.5 * J * n
                     : (U - 1.5 * J) * n * (n - 1) / 2 - J * S * (S + 1) +
                           0.75 * J * n;
    E.insert(E.end(), m.deg, e);
  }
  std::sort(E.begin(), E.end());
  return E;
}

TEST_CASE("FCIDUMP two-body symmetry") {
  ROOT_ONLY(MPI_COMM_WORLD);
  const double U = 5.0, J = 1.0;
  const size_t norb = 3, norb4 = norb * norb * norb * norb;
  auto at = [norb](size_t p, size_t q, size_t r, size_t s) {
    return p + norb * (q + norb * (r + norb * s));
  };

  SECTION("Exchange and pair hopping kept distinct") {
    std::vector<double> V(norb4, 0.0);
    macis::read_fcidump_2body(dworin_narath3_fcidump, V.data(), norb);
    REQUIRE(V[at(0, 1, 1, 0)] == Approx(J));
    REQUIRE(V[at(1, 0, 0, 1)] == Approx(J));
    REQUIRE(V[at(0, 1, 0, 1)] == Approx(0.0).margin(1e-14));
    REQUIRE(V[at(1, 0, 1, 0)] == Approx(0.0).margin(1e-14));
    REQUIRE(V[at(1, 1, 0, 0)] == Approx(U - J));
  }

  SECTION("One line per 8-fold class reads as before") {
    // layout.index_first lists (11|22) only; (22|11) must be filled.
    std::vector<double> V(16, 0.0);
    macis::read_fcidump_2body(layout_index_first_fcidump, V.data(), 2);
    REQUIRE(V[0 + 2 * (0 + 2 * (1 + 2 * 1))] == Approx(0.25));  // (11|22)
    REQUIRE(V[1 + 2 * (1 + 2 * (0 + 2 * 0))] == Approx(0.25));  // (22|11)
  }

  SECTION("Kanamori spectrum") {
    auto W = atom3_spectrum(kanamori3_fcidump);
    auto E = atom3_reference(U, J, true);
    REQUIRE(W.size() == E.size());
    for(size_t i = 0; i < W.size(); ++i) REQUIRE(W[i] == Approx(E[i]).margin(1e-10));
  }

  SECTION("Dworin-Narath spectrum") {
    auto W = atom3_spectrum(dworin_narath3_fcidump);
    auto E = atom3_reference(U, J, false);
    REQUIRE(W.size() == E.size());
    for(size_t i = 0; i < W.size(); ++i) REQUIRE(W[i] == Approx(E[i]).margin(1e-10));
  }

  SECTION("Writer round trip keeps the explicit zeros") {
    std::vector<double> T(norb * norb, 0.0), V(norb4, 0.0), V2(norb4, 0.0);
    macis::read_fcidump_2body(dworin_narath3_fcidump, V.data(), norb);
    const std::string out = "dworin_narath3.roundtrip.fci.dat";
    std::remove(out.c_str());  // the spdlog sink appends
    macis::write_fcidump(out, norb, T.data(), norb, V.data(), norb, 0.0);
    spdlog::drop("fcidump");
    macis::read_fcidump_2body(out, V2.data(), norb);
    for(size_t i = 0; i < norb4; ++i) REQUIRE(V2[i] == Approx(V[i]).margin(1e-14));
  }
}
