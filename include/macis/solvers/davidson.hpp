/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <lobpcgxx/lobpcg.hpp>
#include <macis/util/mpi.hpp>
#include <random>
#include <sparsexx/matrix_types/csr_matrix.hpp>

#ifdef MACIS_ENABLE_MPI
#include <sparsexx/spblas/pspmbv.hpp>
#endif

#include <sparsexx/spblas/spmbv.hpp>

namespace macis {

template <typename SpMatType>
class SparseMatrixOperator {
  using index_type = typename SpMatType::index_type;

  const SpMatType& m_matrix_;
#ifdef MACIS_ENABLE_MPI
  sparsexx::spblas::spmv_info<index_type> m_spmv_info_;
#endif

 public:
  SparseMatrixOperator(const SpMatType& m) : m_matrix_(m) {
#ifdef MACIS_ENABLE_MPI
    if constexpr(sparsexx::is_dist_sparse_matrix_v<SpMatType>) {
      m_spmv_info_ = sparsexx::spblas::generate_spmv_comm_info(m);
    }
#endif
  }

  void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                       double beta, double* AV, size_t LDAV) const {
#ifdef MACIS_ENABLE_MPI
    if constexpr(sparsexx::is_dist_sparse_matrix_v<SpMatType>) {
      sparsexx::spblas::pgespmv(alpha, m_matrix_, V, beta, AV, m_spmv_info_);
    } else {
#endif
      sparsexx::spblas::gespmbv(m, alpha, m_matrix_, V, LDV, beta, AV, LDAV);
#ifdef MACIS_ENABLE_MPI
    }
#endif
  }
};

template <typename SpMatType>
void diagonal_guess(size_t N, const SpMatType& A, double* X) {
  // Extract diagonal and setup guess
  auto D = extract_diagonal_elements(A);
  auto D_min = std::min_element(D.begin(), D.end());
  auto min_idx = std::distance(D.begin(), D_min);
  X[min_idx] = 1.;
}

#ifdef MACIS_ENABLE_MPI
template <typename SpMatType>
void p_diagonal_guess(size_t N_local, const SpMatType& A, double* X) {
  auto comm = A.comm();
  int world_rank, world_size;
  MPI_Comm_rank(comm, &world_rank);
  MPI_Comm_size(comm, &world_size);

  // Extract diagonal tile
  auto A_diagonal_tile = A.diagonal_tile_ptr();
  if(!A_diagonal_tile) throw std::runtime_error("Diagonal Tile Not Populated");

  // Gather Diagonal
  auto D_local = extract_diagonal_elements(*A_diagonal_tile);

  std::vector<int> remote_counts(world_size), row_starts(world_size + 1, 0);
  for(auto i = 0; i < world_size; ++i) {
    remote_counts[i] = A.row_extent(i);
    row_starts[i + 1] = row_starts[i] + A.row_extent(i);
  }

  std::vector<double> D(row_starts.back());

  MPI_Allgatherv(D_local.data(), D_local.size(), MPI_DOUBLE, D.data(),
                 remote_counts.data(), row_starts.data(), MPI_DOUBLE, comm);

  // Determine min index
  auto D_min = std::min_element(D.begin(), D.end());
  auto min_idx = std::distance(D.begin(), D_min);

  // Zero out guess
  for(size_t i = 0; i < N_local; ++i) X[i] = 0.;

  // Get owner rank
  int owner_rank = min_idx / remote_counts[0];
  if(world_rank == owner_rank) {
    X[min_idx - A.local_row_start()] = 1.;
  }
}
#endif

inline bool gram_schmidt(int64_t N, int64_t K, const double* V_old, int64_t LDV,
                         double* V_new) {
  std::vector<double> inner(K);
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, K,
             1, N, 1., V_old, LDV, V_new, N, 0., inner.data(), K);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, N, 1,
             K, -1., V_old, LDV, inner.data(), K, 1., V_new, N);

  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, K,
             1, N, 1., V_old, LDV, V_new, N, 0., inner.data(), K);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, N, 1,
             K, -1., V_old, LDV, inner.data(), K, 1., V_new, N);

  auto nrm = blas::nrm2(N, V_new, 1);
  if(nrm < 1.E-12) return false;  // Linear Dependence Detected
  blas::scal(N, 1. / nrm, V_new, 1);
  return true;  // Success
}

template <typename Functor>
auto davidson(int64_t N, int64_t max_m, const Functor& op, const double* D,
              double tol, double* X) {
  using hrt_t = std::chrono::high_resolution_clock;
  using dur_t = std::chrono::duration<double, std::milli>;

  // Input validation
  if(!X) throw std::runtime_error("Davidson: No Guess Provided");
  if(N <= 0) throw std::runtime_error("Davidson: Invalid Matrix Size");
  if(!D) throw std::runtime_error("Davidson: No Diagonal Provided");

  // Check for NaN/Inf in diagonal and initial guess
  for(size_t i = 0; i < N; ++i) {
    if(!std::isfinite(D[i]))
      throw std::runtime_error(
          "Davidson: Non-finite Diagonal Element Detected");
    if(!std::isfinite(X[i]))
      throw std::runtime_error("Davidson: Non-finite Initial Guess Detected");
  }

  auto logger = spdlog::get("davidson");
  if(!logger) {
    logger = spdlog::stdout_color_mt("davidson");
  }
  max_m = std::min(max_m, N);

  auto D_min = *std::min_element(D, D + N);
  auto D_max = *std::max_element(D, D + N);
  logger->info("Diagonal range: [{:.12e}, {:.12e}]", D_min, D_max);

  logger->info("[Davidson Eigensolver]:");
  logger->info("  {} = {:6}, {} = {:4}, {} = {:10.5e}", "N", N, "MAX_M", max_m,
               "RES_TOL", tol);

  std::vector<double> V(N * (max_m + 1)), AV(N * (max_m + 1)),
      C((max_m + 1) * (max_m + 1)), LAM(max_m + 1);

  // Copy over guess
  std::copy_n(X, N, V.begin());

  // Compute Initial A*V
  op.operator_action(1, 1., V.data(), N, 0., AV.data(), N);

  // Copy AV(:,0) -> V(:,1) and orthogonalize wrt V(:,0)
  std::copy_n(AV.data(), N, V.data() + N);
  if(!gram_schmidt(N, 1, V.data(), N, V.data() + N)) {
    // throw std::runtime_error("Davidson: Linear Dependence Detected in Initial
    // vectors from gram_schmidt");
    logger->warn(
        "  * WARNING: Linear Dependence Detected in Initial vectors from "
        "gram_schmidt. Attempting recovery...");
    // Attempt to recover by replacing with random vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0, 1);
    for(size_t i = 0; i < N; ++i) V[i + N] = dis(gen);
    if(!gram_schmidt(N, 1, V.data(), N, V.data() + N)) {
      throw std::runtime_error(
          "Davidson: Linear Dependence Detected in Initial vectors from "
          "gram_schmidt after recovery attempt");
    } else
      logger->info("  * Recovery successful.");
  }

  bool converged = false;
  size_t iter = 1;

  // Stagnation detection variables
  size_t stagnant_iter = 0;
  double prev_eigvalue = std::numeric_limits<double>::max();
  const size_t max_stagnant_iter = 30; // Number of iterations to consider stagnation
  const double stagnation_tol = 1e-13; // Tolerance for stagnation detection

  for(int64_t i = 1; i < max_m; ++i, ++iter) {
    const auto k = i + 1;  // Current subspace dimension after new vector

    // AV(:,i) = A * V(:,i)
    auto op_st = hrt_t::now();
    op.operator_action(1, 1., V.data() + i * N, N, 0., AV.data() + i * N, N);
    auto op_en = hrt_t::now();
    dur_t op_dur = op_en - op_st;

    // Rayleigh Ritz
    auto rr_st = hrt_t::now();
    lobpcgxx::rayleigh_ritz(N, k, V.data(), N, AV.data(), N, LAM.data(),
                            C.data(), k);
    auto rr_en = hrt_t::now();
    dur_t rr_dur = rr_en - rr_st;

    // Compute Residual (A - LAM(0)*I) * V(:,0:i) * C(:,0)
    auto res_st = hrt_t::now();
    double* R = V.data() + (i + 1) * N;

    // X = V*C
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, N,
               1, k, 1., V.data(), N, C.data(), k, 0., X, N);

    // R = X
    std::copy_n(X, N, R);

    // R = (AV - LAM[0]*V)*C = AV*C - LAM[0]*X = AV*C - LAM[0]*R
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, N,
               1, k, 1., AV.data(), N, C.data(), k, -LAM[0], R, N);

    // Compute residual norm
    auto res_nrm = blas::nrm2(N, R, 1);

    auto res_en = hrt_t::now();
    dur_t res_dur = res_en - res_st;

    logger->info("iter = {:4}, LAM(0) = {:20.12e}, RNORM = {:20.12e}", i,
                 LAM[0], res_nrm);
    logger->trace(
        "  * OP_DUR = {:.2e} ms, RR_DUR = {:.2e} ms, RES_DUR = {:.2e} ms",
        op_dur.count(), rr_dur.count(), res_dur.count());

    // Check for convergence
    if(res_nrm < tol) {
      converged = true;
      break;
    }

    // Check for stagnation
    double eig_change = std::abs(LAM[0] - prev_eigvalue);
    if(eig_change < stagnation_tol) {
      stagnant_iter++;
      if(stagnant_iter >= max_stagnant_iter) {
        logger->warn("  * WARNING: Davidson Stagnation Detected ({} iterations with < {:.1e} change in eigenvalue)", stagnant_iter, stagnation_tol);
        logger->warn("    Current residual norm: {:.3e} may be best achievable. Convergence threshold = {:.3e}", res_nrm, tol);
        if (res_nrm < 100*tol) {
          converged = true;
        }
        break;
      }
    } else {
      stagnant_iter = 0;  // Reset stagnation counter
    }
    prev_eigvalue = LAM[0];

    // Compute new vector
    // (D - LAM(0)*I) * W = -R ==> W = -(D - LAM(0)*I)**-1 * R

    // Safety check for near-degeneracies in D
    int degenerate_count = 0;
    for(auto j = 0; j < N; ++j) {
      // R[j] = -R[j] / (D[j] - LAM[0]);
      double denominator = D[j] - LAM[0];
      if(std::abs(denominator) < 1e-12) {
        // Prevent division by zero or very small numbers
        denominator = (denominator >= 0) ? 1e-8 : -1e-8;
        degenerate_count++;
      }
      R[j] = -R[j] / denominator;
    }

    if(degenerate_count > 0) {
      logger->warn("  * WARNING: {} near-degenerate diagonal elements detected",
                   degenerate_count);
    }

    // Project new vector out from old vectors
    if(!gram_schmidt(N, k, V.data(), N, R)) {
      logger->warn(
          "  * WARNING: Linear Dependence Detected at Iteration {}. Stopping "
          "early.",
          iter);
      converged = true;
      break;
    }

  }  // Davidson iterations

  if(!converged) throw std::runtime_error("Davidson Did Not Converge!");
  logger->info("Davidson Converged!");

  return std::make_pair(iter, LAM[0]);
}

#ifdef MACIS_ENABLE_MPI
inline bool p_gram_schmidt(int64_t N_local, int64_t K, const double* V_old,
                           int64_t LDV, double* V_new, MPI_Comm comm) {
  std::vector<double> inner(K);
  // Compute local V_old**H * V_new
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, K,
             1, N_local, 1., V_old, LDV, V_new, N_local, 0., inner.data(), K);

  // Reduce inner product
  allreduce(inner.data(), K, MPI_SUM, comm);

  // Project locally
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             N_local, 1, K, -1., V_old, LDV, inner.data(), K, 1., V_new,
             N_local);

  // Repeat
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, K,
             1, N_local, 1., V_old, LDV, V_new, N_local, 0., inner.data(), K);
  allreduce(inner.data(), K, MPI_SUM, comm);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             N_local, 1, K, -1., V_old, LDV, inner.data(), K, 1., V_new,
             N_local);

  // Normalize
  double dot = blas::dot(N_local, V_new, 1, V_new, 1);
  dot = allreduce(dot, MPI_SUM, comm);

  double nrm = std::sqrt(dot);

  if(nrm < 1.E-12) return false;  // Linear Dependence Detected

  blas::scal(N_local, 1. / nrm, V_new, 1);
  return true;  // Success
}

inline void p_rayleigh_ritz(int64_t N_local, int64_t K, const double* X,
                            int64_t LDX, const double* AX, int64_t LDAX,
                            double* W, double* C, int64_t LDC, MPI_Comm comm) {
  int world_rank;
  MPI_Comm_rank(comm, &world_rank);

  // Compute Local inner product
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, K,
             K, N_local, 1., X, LDX, AX, LDAX, 0., C, LDC);

  // Reduce result
  if(LDC != K) throw std::runtime_error("DIE DIE DIE RR");
  allreduce(C, K * K, MPI_SUM, comm);

  // Do local diagonalization on rank-0
  if(!world_rank) {
    lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, K, C, LDC, W);
  }

  // Broadcast results
  MPI_Bcast(W, K, MPI_DOUBLE, 0, comm);
  MPI_Bcast(C, K * K, MPI_DOUBLE, 0, comm);
}

template <typename Functor>
auto p_davidson(int64_t N_local, int64_t max_m, const Functor& op,
                const double* D_local, double tol, double* X_local,
                MPI_Comm comm) {
  using hrt_t = std::chrono::high_resolution_clock;
  using dur_t = std::chrono::duration<double, std::milli>;

  // Input validation
  if(N_local and !X_local)
    throw std::runtime_error("Davidson: No Guess Provided");
  if(N_local <= 0) throw std::runtime_error("Davidson: Invalid Matrix Size");
  if(!D_local) throw std::runtime_error("Davidson: No Diagonal Provided");

  // Check for NaN/Inf in diagonal and initial guess
  for(size_t i = 0; i < N_local; ++i) {
    if(!std::isfinite(D_local[i]))
      throw std::runtime_error(
          "Davidson: Non-finite Diagonal Element Detected");
    if(!std::isfinite(X_local[i]))
      throw std::runtime_error("Davidson: Non-finite Initial Guess Detected");
  }

  int world_rank, world_size;
  MPI_Comm_rank(comm, &world_rank);
  MPI_Comm_size(comm, &world_size);

  auto logger = spdlog::get("davidson");
  if(!logger) {
    logger = world_rank ? spdlog::null_logger_mt("davidson")
                        : spdlog::stdout_color_mt("davidson");
  }

  logger->info("[Davidson Eigensolver]:");
  logger->info("  {} = {:6}, {} = {:4}, {} = {:10.5e}", "N_LOCAL", N_local,
               "MAX_M", max_m, "RES_TOL", tol);

  // Allocations
  std::vector<double> V_local(N_local * (max_m + 1)),
      AV_local(N_local * (max_m + 1)), C((max_m + 1) * (max_m + 1)),
      LAM(max_m + 1);

  // Copy over guess
  std::copy_n(X_local, N_local, V_local.begin());

  // Compute initial A*V
  op.operator_action(1, 1., V_local.data(), N_local, 0., AV_local.data(),
                     N_local);

  // Copy AV(:,0) -> V(:,1) and orthogonalize wrt V(:,0)
  std::copy_n(AV_local.data(), N_local, V_local.data() + N_local);
  if(!p_gram_schmidt(N_local, 1, V_local.data(), N_local,
                     V_local.data() + N_local, comm)) {
    // throw std::runtime_error("Davidson: Linear Dependence Detected in Initial
    // vectors from p_gram_schmidt");
    logger->warn(
        "  * WARNING: Linear Dependence Detected in Initial vectors from "
        "p_gram_schmidt. Attempting recovery...");
    // Attempt to recover by replacing with random vector
    std::mt19937 gen(world_rank + 1);
    std::normal_distribution<double> dis(0, 1);
    for(size_t i = 0; i < N_local; ++i) V_local[i + N_local] = dis(gen);
    if(!p_gram_schmidt(N_local, 1, V_local.data(), N_local,
                       V_local.data() + N_local, comm)) {
      throw std::runtime_error(
          "Davidson: Linear Dependence Detected in Initial vectors from "
          "p_gram_schmidt after recovery attempt");
    } else
      logger->info("  * Recovery successful.");
  }

  bool converged = false;
  int64_t iter = 1;

  // Stagnation detection variables
  double prev_eigvalue = std::numeric_limits<double>::max();
  size_t stagnant_iter = 0;
  const size_t max_stagnant_iter = 30; // Number of iterations to consider stagnation
  const double stagnation_tol = 1e-13; // Tolerance for stagnation detection


  for(int64_t i = 1; i < max_m; ++i, ++iter) {
    const auto k = i + 1;  // Current subspace dimension after new vector

    // AV(:,i) = A * V(:,i)
    auto op_st = hrt_t::now();
    op.operator_action(1, 1., V_local.data() + i * N_local, N_local, 0.,
                       AV_local.data() + i * N_local, N_local);
    auto op_en = hrt_t::now();
    dur_t op_dur = op_en - op_st;

    // Rayleigh Ritz
    auto rr_st = hrt_t::now();
    p_rayleigh_ritz(N_local, k, V_local.data(), N_local, AV_local.data(),
                    N_local, LAM.data(), C.data(), k, comm);
    auto rr_en = hrt_t::now();
    dur_t rr_dur = rr_en - rr_st;

    // Compute Residual (A - LAM(0)*I) * V(:,0:i) * C(:,0)
    auto res_st = hrt_t::now();
    double* R_local = V_local.data() + (i + 1) * N_local;

    // X = V*C
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               N_local, 1, k, 1., V_local.data(), N_local, C.data(), k, 0.,
               X_local, N_local);

    // R = X
    std::copy_n(X_local, N_local, R_local);

    // R = (AV - LAM[0]*V)*C = AV*C - LAM[0]*X = AV*C - LAM[0]*R
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               N_local, 1, k, 1., AV_local.data(), N_local, C.data(), k,
               -LAM[0], R_local, N_local);

    // Compute residual norm
    auto res_dot = blas::dot(N_local, R_local, 1, R_local, 1);
    res_dot = allreduce(res_dot, MPI_SUM, comm);
    auto res_nrm = std::sqrt(res_dot);

    auto res_en = hrt_t::now();
    dur_t res_dur = res_en - res_st;

    logger->info("iter = {:4}, LAM(0) = {:20.12e}, RNORM = {:20.12e}", i,
                 LAM[0], res_nrm);
    logger->trace(
        "  * OP_DUR = {:.2e} ms, RR_DUR = {:.2e} ms, RES_DUR = {:.2e} ms",
        op_dur.count(), rr_dur.count(), res_dur.count());

    // Check for convergence
    if(res_nrm < tol) {
      converged = true;
      break;
    }

    // Check for stagnation in eigenvalue convergence
    double eig_change = std::abs(LAM[0] - prev_eigvalue);
    if(eig_change < stagnation_tol) {
      stagnant_iter++;
      if(stagnant_iter >= max_stagnant_iter) {
        logger->warn("  * WARNING: Davidson Stagnation Detected ({} iterations with < {:.1e} change in eigenvalue)", stagnant_iter, stagnation_tol);
        logger->warn("    Current residual norm: {:.3e} may be best achievable. Convergence threshold {:.3e}", res_nrm, tol);
        if (res_nrm < 100*tol) {
          converged = true;
        }
        break;
      }
    } else {
      stagnant_iter = 0;  // Reset stagnation counter
    }
    prev_eigvalue = LAM[0];

    // Compute new vector
    // (D - LAM(0)*I) * W = -R ==> W = -(D - LAM(0)*I)**-1 * R

    // Safety check for near-degeneracies in D
    int degenerate_count = 0;

    for(auto j = 0; j < N_local; ++j) {
      // R_local[j] = -R_local[j] / (D_local[j] - LAM[0]);
      double denominator = D_local[j] - LAM[0];
      if(std::abs(denominator) < 1e-12) {
        // Prevent division by zero or very small numbers
        denominator = (denominator >= 0) ? 1e-8 : -1e-8;
        degenerate_count++;
      }
      R_local[j] = -R_local[j] / denominator;
    }

    size_t total_degenerate_count = 0;
    if(degenerate_count > 0) {
      total_degenerate_count = allreduce(degenerate_count, MPI_SUM, comm);
      logger->warn("  * WARNING: {} near-degenerate diagonal elements detected",
                   total_degenerate_count);
    }

    // Project new vector out form old vectors
    if(!p_gram_schmidt(N_local, k, V_local.data(), N_local, R_local, comm)) {
      logger->warn(
          "  * WARNING: Linear Dependence Detected at Iteration {}. Stopping "
          "early.",
          iter);
      converged = true;
      break;
    }

  }  // Davidson iterations

  if(!converged) throw std::runtime_error("Davidson Did Not Converge!");
  logger->info("Davidson Converged!");

  return std::make_pair(iter, LAM[0]);
}
#endif

}  // namespace macis
