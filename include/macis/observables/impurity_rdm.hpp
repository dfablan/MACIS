#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <blas.hh>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
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

struct ImpRdmBlock {
  int n_up = 0;
  int n_dn = 0;
  size_t dim = 0;
  std::vector<uint32_t> basis_indices;
  std::vector<double> matrix;
};

struct ImpRdmBlocks {
  size_t basis_size = 0;
  int n_imp = 0;
  std::vector<ImpRdmBlock> blocks;
  std::vector<uint32_t> block_of;
  std::vector<uint32_t> local_of;

  double trace() const {
    double tr = 0.0;
    for(const auto& blk : blocks) {
      for(size_t r = 0; r < blk.dim; ++r) tr += blk.matrix[r * blk.dim + r];
    }
    return tr;
  }

  std::vector<double> to_dense() const {
    std::vector<double> rho(basis_size * basis_size, 0.0);
    for(const auto& blk : blocks) {
      for(size_t r = 0; r < blk.dim; ++r) {
        const size_t gi = blk.basis_indices[r];
        for(size_t c = 0; c < blk.dim; ++c) {
          rho[gi * basis_size + blk.basis_indices[c]] =
              blk.matrix[r * blk.dim + c];
        }
      }
    }
    return rho;
  }

  void write_dense_bin(const std::string& fname) const {
    std::ofstream out(fname, std::ios::binary);
    if(!out.is_open()) {
      throw std::runtime_error("Cannot write to file: " + fname);
    }
    const int basis_size_int = static_cast<int>(basis_size);
    out.write(reinterpret_cast<const char*>(&basis_size_int), sizeof(int));

    std::vector<double> row(basis_size, 0.0);
    for(size_t i = 0; i < basis_size; ++i) {
      std::fill(row.begin(), row.end(), 0.0);
      const auto& blk = blocks[block_of[i]];
      const size_t r = local_of[i];
      for(size_t c = 0; c < blk.dim; ++c) {
        row[blk.basis_indices[c]] = blk.matrix[r * blk.dim + c];
      }
      out.write(reinterpret_cast<const char*>(row.data()),
                (std::streamsize)(basis_size * sizeof(double)));
    }
    out.close();
  }

  void write_dense_txt(const std::string& fname) const {
    std::ofstream out_txt(fname);
    if(!out_txt.is_open()) {
      throw std::runtime_error("Cannot write text output: " + fname);
    }
    out_txt << std::scientific << std::setprecision(12);
    out_txt << "# Reduced Density Matrix (basis_size=" << basis_size << ")\n";
    out_txt << "# row col real imag\n";
    for(size_t i = 0; i < basis_size; ++i) {
      const auto& blk = blocks[block_of[i]];
      const size_t r = local_of[i];
      for(size_t c = 0; c < blk.dim; ++c) {
        const double v = blk.matrix[r * blk.dim + c];
        if(std::abs(v) > 1e-12) {
          out_txt << i << " " << blk.basis_indices[c] << " " << v << "\n";
        }
      }
    }
    out_txt.close();
  }

  void write_blocks_bin(const std::string& fname,
                        double drop_thresh = 0.0) const {
    std::ofstream out(fname, std::ios::binary);
    if(!out.is_open()) {
      throw std::runtime_error("Cannot write to file: " + fname);
    }
    const char magic[8] = {'R', 'D', 'M', 'B', 'L', 'O', 'C', 'K'};
    const uint32_t version = 1;
    const uint64_t bs = static_cast<uint64_t>(basis_size);
    const int32_t ni = static_cast<int32_t>(n_imp);
    const uint32_t nb = static_cast<uint32_t>(blocks.size());
    const double thresh = drop_thresh;
    out.write(magic, 8);
    out.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&bs), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(&ni), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&nb), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&thresh), sizeof(double));

    for(const auto& blk : blocks) {
      double tr = 0.0;
      double max_abs = 0.0;
      for(size_t r = 0; r < blk.dim; ++r) {
        tr += blk.matrix[r * blk.dim + r];
        for(size_t c = 0; c < blk.dim; ++c) {
          max_abs = std::max(max_abs, std::abs(blk.matrix[r * blk.dim + c]));
        }
      }
      const int32_t nu = blk.n_up;
      const int32_t nd = blk.n_dn;
      const uint64_t dim = static_cast<uint64_t>(blk.dim);
      const uint8_t stored = (max_abs > drop_thresh) ? 1 : 0;
      out.write(reinterpret_cast<const char*>(&nu), sizeof(int32_t));
      out.write(reinterpret_cast<const char*>(&nd), sizeof(int32_t));
      out.write(reinterpret_cast<const char*>(&dim), sizeof(uint64_t));
      out.write(reinterpret_cast<const char*>(&tr), sizeof(double));
      out.write(reinterpret_cast<const char*>(&stored), sizeof(uint8_t));
      out.write(reinterpret_cast<const char*>(blk.basis_indices.data()),
                (std::streamsize)(blk.dim * sizeof(uint32_t)));
      if(stored) {
        out.write(reinterpret_cast<const char*>(blk.matrix.data()),
                  (std::streamsize)(blk.dim * blk.dim * sizeof(double)));
      }
    }
    out.close();
  }

  static ImpRdmBlocks from_dense(const std::vector<double>& rho, int n_imp);
};

inline ImpRdmBlocks make_imp_rdm_blocks(
    const std::vector<uint64_t>& basis_states, int n_imp) {
  ImpRdmBlocks out;
  out.basis_size = basis_states.size();
  out.n_imp = n_imp;

  const uint64_t mask = (1ULL << (size_t)n_imp) - 1;
  const int nl = n_imp + 1;
  out.blocks.resize((size_t)nl * (size_t)nl);
  for(int u = 0; u < nl; ++u) {
    for(int d = 0; d < nl; ++d) {
      auto& blk = out.blocks[(size_t)u * (size_t)nl + (size_t)d];
      blk.n_up = u;
      blk.n_dn = d;
    }
  }

  out.block_of.assign(out.basis_size, 0);
  out.local_of.assign(out.basis_size, 0);
  for(size_t i = 0; i < out.basis_size; ++i) {
    const uint64_t s = basis_states[i];
    const int nu = popcnt_u64_local_implementation((s >> n_imp) & mask);
    const int nd = popcnt_u64_local_implementation(s & mask);
    const size_t b = (size_t)nu * (size_t)nl + (size_t)nd;
    out.block_of[i] = static_cast<uint32_t>(b);
    out.local_of[i] = static_cast<uint32_t>(out.blocks[b].basis_indices.size());
    out.blocks[b].basis_indices.push_back(static_cast<uint32_t>(i));
  }

  for(auto& blk : out.blocks) {
    blk.dim = blk.basis_indices.size();
    blk.matrix.assign(blk.dim * blk.dim, 0.0);
  }
  return out;
}

inline ImpRdmBlocks ImpRdmBlocks::from_dense(const std::vector<double>& rho,
                                             int n_imp) {
  const size_t basis_size =
      static_cast<size_t>(std::llround(std::sqrt((double)rho.size())));
  if(basis_size * basis_size != rho.size() ||
     basis_size != (size_t{1} << (2 * n_imp))) {
    throw std::runtime_error(
        "ImpRdmBlocks::from_dense: rho size inconsistent with n_imp");
  }
  std::vector<uint64_t> basis_states(basis_size);
  for(size_t i = 0; i < basis_size; ++i) {
    basis_states[i] = static_cast<uint64_t>(basis_size - 1 - i);
  }
  ImpRdmBlocks out = make_imp_rdm_blocks(basis_states, n_imp);
  for(auto& blk : out.blocks) {
    for(size_t r = 0; r < blk.dim; ++r) {
      const size_t gi = blk.basis_indices[r];
      for(size_t c = 0; c < blk.dim; ++c) {
        blk.matrix[r * blk.dim + c] =
            rho[gi * basis_size + blk.basis_indices[c]];
      }
    }
  }
  return out;
}

inline std::vector<double> compute_overlap_matrix_block(
    const std::vector<double>& rot_mat, int n_imp_orbitals,
    int n_active_orbitals, const std::vector<uint64_t>& basis_states,
    const std::vector<uint32_t>& indices) {
  const size_t dim = indices.size();
  std::vector<double> overlap_mat(dim * dim, 0.0);
  std::vector<double> rot_mat_restricted_up;
  std::vector<double> rot_mat_restricted_down;
  rot_mat_restricted_up.reserve((size_t)n_imp_orbitals *
                                (size_t)n_imp_orbitals);
  rot_mat_restricted_down.reserve((size_t)n_imp_orbitals *
                                  (size_t)n_imp_orbitals);
  const uint64_t mask = (1ULL << (size_t)n_imp_orbitals) - 1;

  for(size_t bi = 0; bi < dim; ++bi) {
    const uint64_t bra_state = basis_states[indices[bi]];
    const uint64_t bra_down = bra_state & mask;
    const uint64_t bra_up = (bra_state >> (size_t)n_imp_orbitals) & mask;
    const int n_occ_bra_up = (int)std::bitset<64>(bra_up).count();
    const int n_occ_bra_down = (int)std::bitset<64>(bra_down).count();

    for(size_t bj = 0; bj < dim; ++bj) {
      const uint64_t ket_state = basis_states[indices[bj]];
      const uint64_t ket_down = ket_state & mask;
      const uint64_t ket_up = (ket_state >> (size_t)n_imp_orbitals) & mask;
      const int n_occ_ket_up = (int)std::bitset<64>(ket_up).count();
      const int n_occ_ket_down = (int)std::bitset<64>(ket_down).count();
      if(n_occ_bra_up != n_occ_ket_up || n_occ_bra_down != n_occ_ket_down) {
        overlap_mat[bi * dim + bj] = 0.0;
        continue;
      }

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

      overlap_mat[bi * dim + bj] =
          determinant_local_implementation(rot_mat_restricted_up,
                                           n_occ_bra_up) *
          determinant_local_implementation(rot_mat_restricted_down,
                                           n_occ_bra_down);
    }
  }

  return overlap_mat;
}

template <size_t N>
void build_reduced_density_matrix_blocks(
    const std::vector<ImpBathDecomp<N>>& determinants,
    const std::vector<double>& C, const std::vector<uint64_t>& basis_states,
    ImpRdmBlocks& rdm) {
  const size_t basis_size = basis_states.size();

  std::unordered_map<uint64_t, size_t> state_to_index;
  state_to_index.reserve(basis_states.size() * 2);
  for(size_t i = 0; i < basis_size; ++i) {
    state_to_index[basis_states[i]] = i;
  }

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

  const size_t n_blocks = rdm.blocks.size();
  for(size_t b = 0; b < n_blocks; ++b) {
    std::fill(rdm.blocks[b].matrix.begin(), rdm.blocks[b].matrix.end(), 0.0);
  }

  std::exception_ptr thread_exception = nullptr;
  std::atomic<bool> abort_flag{false};

#pragma omp parallel
  {
    std::vector<std::vector<double>> rho_local(n_blocks);
    for(size_t b = 0; b < n_blocks; ++b) {
      rho_local[b].assign(rdm.blocks[b].dim * rdm.blocks[b].dim, 0.0);
    }

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

        struct Member {
          size_t bidx;
          double coeff;
          std::array<int, 4> occs;
        };
        std::vector<Member> members;
        members.reserve(group_indices.size());
        for(size_t idx : group_indices) {
          const auto& det = determinants[idx];
          const size_t bidx = canonical_indices[idx];
          if(bidx < basis_size) {
            members.push_back(
                {bidx, C[idx],
                 {det.n_imp_up, det.n_imp_dn, det.n_bath_up, det.n_bath_dn}});
          }
        }
        std::sort(members.begin(), members.end(),
                  [](const Member& a, const Member& b) {
                    return a.bidx < b.bidx;
                  });

        std::vector<size_t> loc;
        std::vector<double> coeff;
        std::vector<std::array<int, 4>> occs;
        for(const auto& m : members) {
          if(!loc.empty() && loc.back() == m.bidx) {
            coeff.back() += m.coeff;
            continue;
          }
          loc.push_back(m.bidx);
          coeff.push_back(m.coeff);
          occs.push_back(m.occs);
        }

        std::vector<size_t> keep;
        for(size_t m = 0; m < loc.size(); ++m) {
          if(std::abs(coeff[m]) >= 1e-14) keep.push_back(m);
        }
        if(keep.empty()) continue;

        const uint32_t b = rdm.block_of[loc[keep[0]]];
        for(size_t m : keep) {
          if(rdm.block_of[loc[m]] != b) {
            throw std::runtime_error(
                "build_reduced_density_matrix_blocks: wavefunction mixes "
                "impurity particle-number sectors, impurity RDM is not "
                "block-diagonal in (N_up,N_dn)");
          }
        }

        const auto& blk = rdm.blocks[b];
        double* mat = rho_local[b].data();
        for(size_t p : keep) {
          for(size_t q : keep) {
            const int sgn = compute_fermionic_sign(
                occs[p][0], occs[p][1], occs[q][0], occs[q][1], occs[p][2],
                occs[p][3]);
            mat[rdm.local_of[loc[p]] * blk.dim + rdm.local_of[loc[q]]] +=
                coeff[p] * coeff[q] * double(sgn);
          }
        }
      } catch(...) {
#pragma omp critical
        {
          if(!thread_exception)
            thread_exception = std::current_exception();
        }
        abort_flag.store(true, std::memory_order_relaxed);
      }
    }

#pragma omp critical
    {
      for(size_t b = 0; b < n_blocks; ++b) {
        for(size_t e = 0; e < rho_local[b].size(); ++e) {
          rdm.blocks[b].matrix[e] += rho_local[b][e];
        }
      }
    }
  }

  if(thread_exception) {
    std::rethrow_exception(thread_exception);
  }
}

inline void rotate_imp_rdm_blocks(ImpRdmBlocks& rdm,
                                  const std::vector<double>& rot_mat,
                                  int n_imp_orbitals, int n_active_orbitals,
                                  const std::vector<uint64_t>& basis_states) {
  std::vector<double> tmp;
  for(auto& blk : rdm.blocks) {
    if(blk.dim == 0) continue;
    double max_abs = 0.0;
    for(double v : blk.matrix) max_abs = std::max(max_abs, std::abs(v));
    if(max_abs == 0.0) continue;

    std::vector<double> overlap = compute_overlap_matrix_block(
        rot_mat, n_imp_orbitals, n_active_orbitals, basis_states,
        blk.basis_indices);

    const int n = static_cast<int>(blk.dim);
    tmp.assign(blk.dim * blk.dim, 0.0);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, n,
               n, n, 1.0, blk.matrix.data(), n, overlap.data(), n, 0.0,
               tmp.data(), n);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, n,
               n, n, 1.0, overlap.data(), n, tmp.data(), n, 0.0,
               blk.matrix.data(), n);
  }
}

template <size_t N>
ImpRdmBlocks compute_impurity_rdm_blocks_from_state(
    size_t n_imp, size_t n_active, const std::vector<macis::wfn_t<N>>& dets,
    const std::vector<double>& C, const std::vector<double>& orb_rot) {
  if(dets.size() != C.size()) {
    throw std::runtime_error(
        "compute_impurity_rdm_blocks_from_state: dets/C size mismatch");
  }
  if(n_imp > 32) {
    throw std::runtime_error(
        "compute_impurity_rdm_blocks_from_state: current uint64_t packing "
        "assumes n_imp <= 32 per spin block");
  }

  const size_t imp_hilb_dim = size_t{1} << (2 * n_imp);
  std::vector<uint64_t> basis_states(imp_hilb_dim);
  for(size_t i = 0; i < imp_hilb_dim; ++i) {
    basis_states[i] = static_cast<uint64_t>(imp_hilb_dim - 1 - i);
  }
  std::cout << "Basis size: " << imp_hilb_dim << "\n";

  auto t0 = std::chrono::high_resolution_clock::now();

  std::vector<ImpBathDecomp<N>> decomposed_dets(dets.size());
  for(size_t i = 0; i < dets.size(); ++i) {
    decomposed_dets[i] = decompose_det(dets[i], n_imp, n_active);
  }

  ImpRdmBlocks rdm = make_imp_rdm_blocks(basis_states, (int)n_imp);
  build_reduced_density_matrix_blocks(decomposed_dets, C, basis_states, rdm);
  rotate_imp_rdm_blocks(rdm, orb_rot, (int)n_imp, (int)n_active, basis_states);

  auto t1 = std::chrono::high_resolution_clock::now();
  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
          .count();
  std::cout << "Time to build reduced density matrix: " << seconds << " s\n";
  std::cout << "Trace: " << rdm.trace() << "\n";

  return rdm;
}

}  // namespace macis