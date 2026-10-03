/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/hamiltonian_generator.hpp>
#include <map>
#include <string>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <blas.hh>
#include <lapack.hh>

#pragma GCC diagnostic pop

namespace macis {

template <size_t N>
void HamiltonianGenerator<N>::rotate_hamiltonian_ordm(const double* ordm,
                                                      double* rot_mat) {
  // SVD on ordm to get natural orbitals
  std::vector<double> natural_orbitals(ordm, ordm + norb2_);
  std::vector<double> S(norb_);
  lapack::gesvd(lapack::Job::OverwriteVec, lapack::Job::NoVec, norb_, norb_,
                natural_orbitals.data(), norb_, S.data(), NULL, 1, NULL, 1);

#if 0
  {
    std::vector<double> tmp(norb2_);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
      norb_, norb_, norb_, 1., natural_orbitals.data(), norb_,
      natural_orbitals.data(), norb_, 0., tmp.data(), norb_ );
    for( auto i = 0; i < norb_; ++i ) tmp[i*(norb_+1)] -= 1.;
    std::cout << "MAX = " << *std::max_element(tmp.begin(),tmp.end(),
      []( auto x, auto y ){ return std::abs(x) < std::abs(y); } ) << std::endl;

    double max_diff = 0.;
    for( auto i = 0; i < norb_; ++i )
    for( auto j = i+1; j < norb_; ++j ) {
      max_diff = std::max( max_diff, 
        std::abs( ordm[i+j*norb_] - ordm[j+i*norb_]));
    }
    std::cout << "MAX = " << max_diff << std::endl;
  }
#endif

  std::vector<double> tmp(norb3_ * norb_), tmp2(norb3_ * norb_);

  // Save rotation matrix
  if(rot_mat != nullptr)
    std::copy(natural_orbitals.data(), natural_orbitals.data() + norb2_,
              rot_mat);

  // Transform Tu
  // Tu <- N**H * Tu * N
  auto* Tu_pq_ptr = Tu_pq_.data_handle();
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             norb_, norb_, norb_, 1., Tu_pq_ptr, norb_, natural_orbitals.data(),
             norb_, 0., tmp.data(), norb_);
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, norb_,
             norb_, norb_, 1., natural_orbitals.data(), norb_, tmp.data(),
             norb_, 0., Tu_pq_ptr, norb_);

  // Transform Td
  // Td <- N**H * Td * N
  auto* Td_pq_ptr = Td_pq_.data_handle();
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             norb_, norb_, norb_, 1., Td_pq_ptr, norb_, natural_orbitals.data(),
             norb_, 0., tmp.data(), norb_);
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, norb_,
             norb_, norb_, 1., natural_orbitals.data(), norb_, tmp.data(),
             norb_, 0., Td_pq_ptr, norb_);

  // Transorm V

  // 1st Quarter
  // (pj|kl) = N(i,p) (ij|kl)
  // W(p,jkl) = N(i,p) * V(i,jkl)
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, norb_,
             norb3_, norb_, 1., natural_orbitals.data(), norb_,
             V_pqrs_.data_handle(), norb_, 0., tmp.data(), norb_);

  // 2nd Quarter
  // (pq|kl) = N(j,q) (pj|kl)
  // W_kl(p,q) = V_kl(p,j) N(j,q)
  for(auto kl = 0; kl < norb2_; ++kl) {
    auto* V_kl = tmp.data() + kl * norb2_;
    auto* W_kl = tmp2.data() + kl * norb2_;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               norb_, norb_, norb_, 1., V_kl, norb_, natural_orbitals.data(),
               norb_, 0., W_kl, norb_);
  }

  // 3rd Quarter
  // (pq|rl) = N(k,r) (pq|kl)
  // W_l(pq,r) = V_l(pq,k) N(k,r)
  for(auto l = 0; l < norb_; ++l) {
    auto* V_l = tmp2.data() + l * norb3_;
    auto* W_l = tmp.data() + l * norb3_;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               norb2_, norb_, norb_, 1., V_l, norb2_, natural_orbitals.data(),
               norb_, 0., W_l, norb2_);
  }

  // 4th Quarter
  // (pq|rs) = N(l,s) (pq|rl)
  // W(pqr,s) = V(pqr,l) N(l,s)
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             norb3_, norb_, norb_, 1., tmp.data(), norb3_,
             natural_orbitals.data(), norb_, 0., V_pqrs_.data_handle(), norb3_);

  // Regenerate intermediates
  generate_integral_intermediates(V_pqrs_);
  SetJustSingles(false);
}

template <size_t N>
void HamiltonianGenerator<N>::rotate_hamiltonian_ordm_imp_bath(
    const double* ordm, const size_t nimps, double* rot_mat, bool spin_dep,
    double* occs_out, const std::vector<int>* group_of) {
  // assert nimp>0
  if(nimps == 0)
    throw std::runtime_error(
        "Invalid number of impurities for rotate_hamiltonian_ordm_imp_bath");
  if(group_of != nullptr and group_of->size() != size_t(norb_))
    throw std::runtime_error(
        "rotate_hamiltonian_ordm_imp_bath: group_of has " +
        std::to_string(group_of->size()) + " entries, expected " +
        std::to_string(norb_));

  // Blocks of orbitals diagonalized separately: the impurity and the bath,
  // and with group_of also split by group (a group of -1 -- an orbital in no
  // group -- is a block of its own). Each block's natural orbitals are
  // written back into the block's own index set, so with group_of the group
  // of every index is the same before and after the rotation.
  std::vector<std::vector<size_t>> blocks;
  for(int side = 0; side < 2; ++side) {
    const size_t lo = side ? nimps : 0, hi = side ? norb_ : nimps;
    if(group_of == nullptr) {
      blocks.emplace_back();
      for(size_t i = lo; i < hi; ++i) blocks.back().push_back(i);
      continue;
    }
    std::map<int, std::vector<size_t>> by_group;
    for(size_t i = lo; i < hi; ++i) {
      const int g = (*group_of)[i];
      if(g < 0)
        blocks.push_back({i});
      else
        by_group[g].push_back(i);
    }
    for(auto& [g, idx] : by_group) blocks.push_back(std::move(idx));
  }

  // SVD on each ordm block to get natural orbitals. gesvd returns singular
  // values in descending order within each block; since ordm's diagonal
  // blocks are PSD, these singular values are exactly the natural-orbital
  // occupations of the basis natural_orbitals below. Hand them back so
  // callers don't have to (and risk disagreeing with) re-diagonalize the
  // same blocks themselves.
  std::vector<double> natural_orbitals(norb2_, 0.);
  for(const auto& idx : blocks) {
    const size_t nb = idx.size();
    if(nb == 0) continue;
    std::vector<double> nat(nb * nb, 0.), S(nb);
    for(size_t j = 0; j < nb; ++j)
      for(size_t i = 0; i < nb; ++i)
        nat[i + j * nb] = ordm[idx[i] + idx[j] * norb_];
    lapack::gesvd(lapack::Job::OverwriteVec, lapack::Job::NoVec, nb, nb,
                  nat.data(), nb, S.data(), NULL, 1, NULL, 1);
    for(size_t j = 0; j < nb; ++j) {
      if(occs_out != nullptr) occs_out[idx[j]] = S[j];
      for(size_t i = 0; i < nb; ++i)
        natural_orbitals[idx[i] + idx[j] * norb_] = nat[i + j * nb];
    }
  }

  std::vector<double> tmp(norb_ * norb_, 0.0), tmp1(norb3_ * norb_, 0.0),
      tmp2(norb3_ * norb_, 0.0);

  // Save rotation matrix
  if(rot_mat != nullptr)
    std::copy(natural_orbitals.data(), natural_orbitals.data() + norb2_,
              rot_mat);

  // Transform Tu
  // Tu <- N**H * Tu * N
  auto* Tu_pq_ptr = Tu_pq_.data_handle();

  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             norb_, norb_, norb_, 1., Tu_pq_ptr, norb_, natural_orbitals.data(),
             norb_, 0., tmp.data(), norb_);

  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, norb_,
             norb_, norb_, 1., natural_orbitals.data(), norb_, tmp.data(),
             norb_, 0., Tu_pq_ptr, norb_);

  if(spin_dep) {
    std::cout << " Spin-dependent is set to TRUE. Performing rotation on Td"
              << std::endl;
    // Transform Td
    // Td <- N**H * Td * N
    auto* Td_pq_ptr = Td_pq_.data_handle();

    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               norb_, norb_, norb_, 1., Td_pq_ptr, norb_,
               natural_orbitals.data(), norb_, 0., tmp.data(), norb_);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               norb_, norb_, norb_, 1., natural_orbitals.data(), norb_,
               tmp.data(), norb_, 0., Td_pq_ptr, norb_);
  }

  // Transorm V

  // 1st Quarter
  // (pj|kl) = N(i,p) (ij|kl)
  // W(p,jkl) = N(i,p) * V(i,jkl)
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, norb_,
             norb3_, norb_, 1., natural_orbitals.data(), norb_,
             V_pqrs_.data_handle(), norb_, 0., tmp1.data(), norb_);

  // 2nd Quarter
  // (pq|kl) = N(j,q) (pj|kl)
  // W_kl(p,q) = V_kl(p,j) N(j,q)
  for(auto kl = 0; kl < norb2_; ++kl) {
    auto* V_kl = tmp1.data() + kl * norb2_;
    auto* W_kl = tmp2.data() + kl * norb2_;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               norb_, norb_, norb_, 1., V_kl, norb_, natural_orbitals.data(),
               norb_, 0., W_kl, norb_);
  }

  // 3rd Quarter
  // (pq|rl) = N(k,r) (pq|kl)
  // W_l(pq,r) = V_l(pq,k) N(k,r)
  for(auto l = 0; l < norb_; ++l) {
    auto* V_l = tmp2.data() + l * norb3_;
    auto* W_l = tmp1.data() + l * norb3_;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               norb2_, norb_, norb_, 1., V_l, norb2_, natural_orbitals.data(),
               norb_, 0., W_l, norb2_);
  }

  // 4th Quarter
  // (pq|rs) = N(l,s) (pq|rl)
  // W(pqr,s) = V(pqr,l) N(l,s)
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             norb3_, norb_, norb_, 1., tmp1.data(), norb3_,
             natural_orbitals.data(), norb_, 0., V_pqrs_.data_handle(), norb3_);

  // Regenerate intermediates
  generate_integral_intermediates(V_pqrs_);
  SetJustSingles(false);
}

template <size_t N>
void HamiltonianGenerator<N>::rotate_hamiltonian_rotmat_imp_bath(
    double* rot_mat, bool spin_dep) {
  // assert nimp>0

  assert(rot_mat != nullptr);

  std::vector<double> natural_orbitals(norb2_, 0.);

  // save rotation matrix
  std::copy(rot_mat, rot_mat + norb2_, natural_orbitals.data());

  std::vector<double> tmp(norb_ * norb_, 0.0), tmp1(norb3_ * norb_, 0.0),
      tmp2(norb3_ * norb_, 0.0);

  // Transform Tu
  // Tu <- N**H * Tu * N
  auto* Tu_pq_ptr = Tu_pq_.data_handle();
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             norb_, norb_, norb_, 1., Tu_pq_ptr, norb_, natural_orbitals.data(),
             norb_, 0., tmp.data(), norb_);
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, norb_,
             norb_, norb_, 1., natural_orbitals.data(), norb_, tmp.data(),
             norb_, 0., Tu_pq_ptr, norb_);

  if(spin_dep) {
    // Transform Td
    // Td <- N**H * Td * N
    auto* Td_pq_ptr = Td_pq_.data_handle();
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               norb_, norb_, norb_, 1., Td_pq_ptr, norb_,
               natural_orbitals.data(), norb_, 0., tmp.data(), norb_);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               norb_, norb_, norb_, 1., natural_orbitals.data(), norb_,
               tmp.data(), norb_, 0., Td_pq_ptr, norb_);
  }

  // Transorm V

  // 1st Quarter
  // (pj|kl) = N(i,p) (ij|kl)
  // W(p,jkl) = N(i,p) * V(i,jkl)
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, norb_,
             norb3_, norb_, 1., natural_orbitals.data(), norb_,
             V_pqrs_.data_handle(), norb_, 0., tmp1.data(), norb_);

  // 2nd Quarter
  // (pq|kl) = N(j,q) (pj|kl)
  // W_kl(p,q) = V_kl(p,j) N(j,q)
  for(auto kl = 0; kl < norb2_; ++kl) {
    auto* V_kl = tmp1.data() + kl * norb2_;
    auto* W_kl = tmp2.data() + kl * norb2_;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               norb_, norb_, norb_, 1., V_kl, norb_, natural_orbitals.data(),
               norb_, 0., W_kl, norb_);
  }

  // 3rd Quarter
  // (pq|rl) = N(k,r) (pq|kl)
  // W_l(pq,r) = V_l(pq,k) N(k,r)
  for(auto l = 0; l < norb_; ++l) {
    auto* V_l = tmp2.data() + l * norb3_;
    auto* W_l = tmp1.data() + l * norb3_;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               norb2_, norb_, norb_, 1., V_l, norb2_, natural_orbitals.data(),
               norb_, 0., W_l, norb2_);
  }

  // 4th Quarter
  // (pq|rs) = N(l,s) (pq|rl)
  // W(pqr,s) = V(pqr,l) N(l,s)
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             norb3_, norb_, norb_, 1., tmp1.data(), norb3_,
             natural_orbitals.data(), norb_, 0., V_pqrs_.data_handle(), norb3_);

  // Regenerate intermediates
  generate_integral_intermediates(V_pqrs_);
  SetJustSingles(false);
}

}  // namespace macis
