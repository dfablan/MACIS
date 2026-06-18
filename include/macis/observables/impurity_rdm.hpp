#pragma once
#include <array>
#include <atomic>
#include <bitset>
#include <blas.hh>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "macis/bitset_operations.hpp"
#include "macis/sd_operations.hpp"
#include "macis/types.hpp"

namespace macis {

inline int popcnt_u64_local_implementation(uint64_t x) {
  return static_cast<int>(std::bitset<64>(x).count());
}

template <size_t N>
struct ImpBathDecomp {
  uint64_t imp_up = 0, imp_dn = 0;
  uint64_t bath_up = 0, bath_dn = 0;
  uint64_t imp_key = 0, bath_key = 0;
  int n_imp_up = 0, n_imp_dn = 0, n_bath_up = 0, n_bath_dn = 0;
};

template <size_t N>
ImpBathDecomp<N> decompose_det(const macis::wfn_t<N>& det, size_t n_imp,
                               size_t n_active) {
  ImpBathDecomp<N> out;

  auto alpha = macis::bitset_lo_word(det);
  auto beta = macis::bitset_hi_word(det);

  for(size_t p = 0; p < n_imp; ++p) {
    if(alpha[n_imp - 1 - p]) out.imp_up |= (1ULL << p);
    if(beta[n_imp - 1 - p]) out.imp_dn |= (1ULL << p);
  }

  for(size_t p = 0; p < n_active - n_imp; ++p) {
    if(alpha[n_active - 1 - p]) out.bath_up |= (1ULL << p);
    if(beta[n_active - 1 - p]) out.bath_dn |= (1ULL << p);
  }

  out.imp_key = out.imp_dn | (out.imp_up << n_imp);
  out.bath_key = out.bath_dn | (out.bath_up << (n_active - n_imp));

  out.n_imp_up = popcnt_u64_local_implementation(out.imp_up);
  out.n_imp_dn = popcnt_u64_local_implementation(out.imp_dn);
  out.n_bath_up = popcnt_u64_local_implementation(out.bath_up);
  out.n_bath_dn = popcnt_u64_local_implementation(out.bath_dn);

  return out;
}

inline double determinant_local_implementation(std::vector<double> mat, int n) {
  if(n == 0) return 1.0;
  double det = 1.0;
  for(int i = 0; i < n; ++i) {
    int pivot = i;
    double best = std::abs(mat[(size_t)i * n + i]);
    for(int j = i + 1; j < n; ++j) {
      double v = std::abs(mat[(size_t)j * n + i]);
      if(v > best) {
        best = v;
        pivot = j;
      }
    }
    if(pivot != i) {
      for(int k = 0; k < n; ++k)
        std::swap(mat[(size_t)i * n + k], mat[(size_t)pivot * n + k]);
      det = -det;
    }
    if(std::abs(mat[(size_t)i * n + i]) < 1e-300)
      return 0.0;  // Matrix is singular
    det *= mat[(size_t)i * n + i];
    double inv = 1.0 / mat[(size_t)i * n + i];
    for(int j = i + 1; j < n; ++j) {
      double f = mat[(size_t)j * n + i] * inv;
      for(int k = i; k < n; ++k)
        mat[(size_t)j * n + k] -= f * mat[(size_t)i * n + k];
    }
  }
  return det;
}

inline int compute_fermionic_sign(int n_imp_up_bra, int n_imp_down_bra,
                                  int n_imp_up_ket, int n_imp_down_ket,
                                  int n_bath_up, int n_bath_down) {
  long long phase = 0;
  phase += 1LL * n_bath_up * (n_imp_up_bra + 2 * n_imp_down_ket + n_imp_up_ket);
  phase += 1LL * n_bath_down *
           (n_imp_up_bra + n_imp_down_bra + n_imp_up_ket + n_imp_down_ket);
  return (phase & 1LL) ? -1 : +1;
}

inline std::vector<double> compute_overlap_matrix(
    const std::vector<double>& rot_mat, int n_imp_orbitals,
    int n_active_orbitals, const std::vector<uint64_t>& basis_states) {
  const int n_bits = 2 * n_imp_orbitals;

  if(n_bits > 64) {
    throw std::runtime_error(
        "Error: 2*n_imp_orbitals > 64, cannot compute overlap matrix with "
        "uint64_t representation");
  }

  const size_t basis_size = basis_states.size();
  std::cout << "Basis size: " << basis_size << "\n";  // Remove later

  std::vector<double> overlap_mat(basis_size * basis_size, 0.0);
  std::vector<double> rot_mat_restricted_up;
  std::vector<double> rot_mat_restricted_down;
  rot_mat_restricted_up.reserve((size_t)n_imp_orbitals *
                                (size_t)n_imp_orbitals);
  rot_mat_restricted_down.reserve((size_t)n_imp_orbitals *
                                  (size_t)n_imp_orbitals);

  for(size_t i = 0; i < basis_size; ++i) {
    const uint64_t bra_state = basis_states[i];
    const uint64_t mask = (1ULL << (size_t)n_imp_orbitals) - 1;
    // extract the first n_imp_orbitals bits for up and the next n_imp_orbitals
    // bits for down (from left to right)
    const uint64_t bra_down = bra_state & mask;
    const uint64_t bra_up = (bra_state >> (size_t)n_imp_orbitals) & mask;
    const int n_occ_bra_up = (int)std::bitset<64>(bra_up).count();
    const int n_occ_bra_down = (int)std::bitset<64>(bra_down).count();

    for(size_t j = 0; j < basis_size; ++j) {
      const uint64_t ket_state = basis_states[j];
      const uint64_t ket_down = ket_state & mask;
      const uint64_t ket_up = (ket_state >> (size_t)n_imp_orbitals) & mask;

      // count the number of occupied orbitals in the bra and ket states
      const int n_occ_ket_up = (int)std::bitset<64>(ket_up).count();
      const int n_occ_ket_down = (int)std::bitset<64>(ket_down).count();

      // select only the states with the same number of occupied orbitals in the
      // bra and ket states
      if(n_occ_bra_up != n_occ_ket_up || n_occ_bra_down != n_occ_ket_down) {
        overlap_mat[i * basis_size + j] = 0.0;
        continue;
      }

      // Build restricted Rotation matrices for the current bra and ket states
      rot_mat_restricted_up.clear();
      rot_mat_restricted_down.clear();

      for(int m = 0; m < n_imp_orbitals; ++m) {
        if((bra_up >> (size_t)(n_imp_orbitals - 1 - m)) & 1ULL) {
          for(int n = 0; n < n_imp_orbitals; ++n) {
            if((ket_up >> (size_t)(n_imp_orbitals - 1 - n)) & 1ULL) {
              rot_mat_restricted_up.push_back(
                  rot_mat[(size_t)m * (size_t)n_active_orbitals + (size_t)n]);
            }
          }
        }
        if((bra_down >> (size_t)(n_imp_orbitals - 1 - m)) & 1ULL) {
          for(int n = 0; n < n_imp_orbitals; ++n) {
            if((ket_down >> (size_t)(n_imp_orbitals - 1 - n)) & 1ULL) {
              rot_mat_restricted_down.push_back(
                  rot_mat[(size_t)m * (size_t)n_active_orbitals + (size_t)n]);
            }
          }
        }
      }

      // O[i][j] = det(R_up) * det(R_down)
      overlap_mat[i * basis_size + j] =
          determinant_local_implementation(rot_mat_restricted_up,
                                           n_occ_bra_up) *
          determinant_local_implementation(rot_mat_restricted_down,
                                           n_occ_bra_down);
    }
  }

  return overlap_mat;
}

template <size_t N>
std::vector<double> build_reduced_density_matrix(
    const std::vector<ImpBathDecomp<N>>& determinants,
    const std::vector<double>& C, const std::vector<uint64_t>& basis_states,
    const std::vector<double>& overlap_matrix) {
  const size_t basis_size = basis_states.size();

  std::unordered_map<uint64_t, size_t> state_to_index;
  state_to_index.reserve(basis_states.size() * 2);
  for(size_t i = 0; i < basis_size; ++i) {
    state_to_index[basis_states[i]] = i;
  }

  // Map each determinant to its index in the canonical impurity basis.
  // Stored in a local vector to keep `determinants` const.
  std::vector<size_t> canonical_indices(determinants.size(), SIZE_MAX);
  for(size_t i = 0; i < determinants.size(); ++i) {
    auto it = state_to_index.find(determinants[i].imp_key);
    if(it != state_to_index.end()) {
      canonical_indices[i] = it->second;
    } else {
      std::cerr << "Warning: impurity int_rep " << determinants[i].imp_key
                << " not in basis\n";
    }
  }

  std::unordered_map<uint64_t, std::vector<size_t>> bath_state_groups;
  for(size_t i = 0; i < determinants.size(); ++i) {
    bath_state_groups[determinants[i].bath_key].push_back(i);
  }
  std::cout << "Unique bath states: " << bath_state_groups.size() << "\n";

  std::vector<uint64_t> bath_keys;
  bath_keys.reserve(bath_state_groups.size());
  for(const auto& kv : bath_state_groups) {
    bath_keys.push_back(kv.first);
  }

  std::vector<double> rho(basis_size * basis_size, 0.0);

  auto t0 = std::chrono::high_resolution_clock::now();

  std::exception_ptr thread_exception = nullptr;
  std::atomic<bool> abort_flag{false};

#pragma omp parallel
  {
    std::vector<double> rho_local(basis_size * basis_size, 0.0);

#pragma omp for schedule(dynamic, 10)
    for(size_t kk = 0; kk < bath_keys.size(); ++kk) {
      if(abort_flag.load(std::memory_order_relaxed)) continue;

      try {
        const uint64_t bath_key = bath_keys[kk];

        auto it = bath_state_groups.find(bath_key);
        if(it == bath_state_groups.end()) {
          throw std::runtime_error("Internal error: bath key not found");
        }
        const std::vector<size_t>& group_indices = it->second;

        std::vector<double> coeffs_vector(basis_size, 0.0);
        std::vector<std::array<int, 4>> occs_vector(basis_size, {0, 0, 0, 0});

        for(size_t idx : group_indices) {
          const auto& det = determinants[idx];
          const auto& coeff = C[idx];
          const size_t bidx = canonical_indices[idx];

          if(bidx < basis_size) {
            coeffs_vector[bidx] += coeff;
            occs_vector[bidx] = {det.n_imp_up, det.n_imp_dn, det.n_bath_up,
                                 det.n_bath_dn};
          }
        }

        for(size_t i = 0; i < basis_size; ++i) {
          if(std::abs(coeffs_vector[i]) < 1e-14) continue;

          for(size_t j = 0; j < basis_size; ++j) {
            if(std::abs(coeffs_vector[j]) < 1e-14) continue;

            const int sgn = compute_fermionic_sign(
                occs_vector[i][0], occs_vector[i][1], occs_vector[j][0],
                occs_vector[j][1], occs_vector[i][2], occs_vector[i][3]);

            rho_local[i * basis_size + j] +=
                coeffs_vector[i] * coeffs_vector[j] * double(sgn);
          }
        }
      } catch(...) {
#pragma omp critical
        {
          if(!thread_exception) thread_exception = std::current_exception();
        }
        abort_flag.store(true, std::memory_order_relaxed);
      }
    }

#pragma omp critical
    {
      for(size_t i = 0; i < rho.size(); ++i) {
        rho[i] += rho_local[i];
      }
    }
  }

  if(thread_exception) {
    std::rethrow_exception(thread_exception);
  }

  // Compute rho' = O * rho * O^T via two GEMMs.
  // Vectors are row-major; ColMajor BLAS interprets them as their transpose,
  // so the two calls mirror the pattern in impurity_solver.hpp.
  // Step 1: tmp = rho * O^T  (ColMajor: O * rho)
  // Step 2: rho' = O * tmp   (ColMajor: tmp * O^T)
  const int n = static_cast<int>(basis_size);
  std::vector<double> tmp(basis_size * basis_size, 0.0);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, n, n,
             n, 1.0, rho.data(), n, overlap_matrix.data(), n, 0.0, tmp.data(),
             n);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, n, n,
             n, 1.0, overlap_matrix.data(), n, tmp.data(), n, 0.0, rho.data(),
             n);

  auto t1 = std::chrono::high_resolution_clock::now();
  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
          .count();
  std::cout << "Time to build reduced density matrix: " << seconds << " s\n";

  double tr = 0.0;
  for(size_t i = 0; i < basis_size; ++i) {
    tr += rho[i * basis_size + i];
  }
  std::cout << "Trace: " << tr << "\n";

  return rho;
}

template <size_t N>
std::vector<double> compute_impurity_rdm_from_state(
    size_t n_imp, size_t n_active, const std::vector<macis::wfn_t<N>>& dets,
    const std::vector<double>& C, const std::vector<double>& orb_rot) {
  if(dets.size() != C.size()) {
    throw std::runtime_error(
        "compute_impurity_rdm_from_state: dets/C size mismatch");
  }
  if(n_imp > 32) {
    throw std::runtime_error(
        "compute_impurity_rdm_from_state: current uint64_t packing assumes "
        "n_imp <= 32 per spin block");
  }

  const size_t imp_hilb_dim = size_t{1} << (2 * n_imp);
  std::vector<double> rho;
  std::vector<double> overlap_matrix;

  std::vector<uint64_t> basis_states(imp_hilb_dim);
  // Gives the basis states in descending order
  for(size_t i = 0; i < imp_hilb_dim; ++i) {
    basis_states[i] = static_cast<uint64_t>(imp_hilb_dim - 1 - i);
  }

  overlap_matrix =
      compute_overlap_matrix(orb_rot, n_imp, n_active, basis_states);

  std::vector<ImpBathDecomp<N>> decomposed_dets(dets.size());
  for(size_t i = 0; i < dets.size(); ++i) {
    decomposed_dets[i] = decompose_det(dets[i], n_imp, n_active);
  }

  rho = build_reduced_density_matrix(decomposed_dets, C, basis_states,
                                     overlap_matrix);

  return rho;
}

}  // namespace macis