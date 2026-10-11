/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <algorithm>
#include <macis/asci/iteration.hpp>

namespace macis {

template <size_t N, typename index_t = int32_t>
auto asci_refine(ASCISettings asci_settings, MCSCFSettings mcscf_settings,
                 double E0, std::vector<wfn_t<N>> wfn, std::vector<double> X,
                 HamiltonianGenerator<N>& ham_gen,
                 size_t norb MACIS_MPI_CODE(, MPI_Comm comm)) {
  auto logger = spdlog::get("asci_refine");
#ifdef MACIS_ENABLE_MPI
  auto world_rank = comm_rank(comm);
#else
  int world_rank = 0;
#endif
  if(!logger)
    logger = world_rank ? spdlog::null_logger_mt("asci_refine")
                        : spdlog::stdout_color_mt("asci_refine");

  logger->info("[ASCI Refine Settings]:");
  logger->info(
      "  NTDETS = {:6}, NCDETS = {:6}, MAX_REFINE_ITER = {:4}, REFINE_TOL = "
      "{:.2e}",
      wfn.size(), asci_settings.ncdets_max, asci_settings.max_refine_iter,
      asci_settings.refine_energy_tol);

  const std::string fmt_string =
      "iter = {:4}, E0 = {:20.12e}, dE = {:14.6e}, duration = "
      "{:02}min{:05.2f}s";

  logger->info(fmt_string, 0, E0, 0.0, 0, 0.0);

  // Refinement Loop
  const size_t ndets = wfn.size();
  bool converged = false, cycled = false;
  // Period-2 cycle detection. The fixed-size selection can alternate between
  // two determinant sets A -> B -> A, each the best truncation of the other's
  // wavefunction, so |dE| never drops below the tolerance (frozen-H test
  // 2026-10-11: after an orbital rotation E alternated with |dE| = 9.4e-5 for
  // 200 iterations). The test is on the determinant sets, not the energies:
  // a damped oscillation that will still converge also has E_k ~ E_{k-2}.
  // Once the set of iterate k equals that of k-2, refinement stops on the
  // lower of the two states. sorted_prev[0] / [1] = sorted determinant sets
  // of iterates k-1 / k-2; wfn_prev, X_prev = iterate k-1 itself.
  auto sorted_set = [](std::vector<wfn_t<N>> w) {
    std::sort(w.begin(), w.end(), bitset_less_comparator<N>{});
    return w;
  };
  std::vector<std::vector<wfn_t<N>>> sorted_prev = {sorted_set(wfn), {}};
  std::vector<wfn_t<N>> wfn_prev;
  std::vector<double> X_prev;
  double E_prev = E0;
  for(size_t iter = 0; iter < asci_settings.max_refine_iter; ++iter) {
    auto start_time = std::chrono::high_resolution_clock::now();

    wfn_prev = wfn;
    X_prev = X;
    E_prev = E0;
    double E;
    std::tie(E, wfn, X) = asci_iter<N, index_t>(
        asci_settings, mcscf_settings, ndets, E0, std::move(wfn), std::move(X),
        ham_gen, norb MACIS_MPI_CODE(, comm));
    if(wfn.size() != ndets) {
      // The whole-orbit budget may legitimately return a few determinants
      // less than the frozen entry size; growth beyond it still throws.
      if(asci_settings.symmetrize_dets and wfn.size() <= ndets)
        logger->info("Refine wavefunction size {} <= {} (whole-orbit budget)",
                     wfn.size(), ndets);
      else
        throw std::runtime_error(
            "Wavefunction size can't change in refinement");
    }

    const auto E_delta = E - E0;
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    int minutes = static_cast<int>(duration.count()) / 60;
    double seconds = duration.count() - minutes * 60.0;

    logger->info(fmt_string, iter + 1, E, E_delta, minutes, seconds);
    E0 = E;
    if(std::abs(E_delta) < asci_settings.refine_energy_tol) {
      converged = true;
      break;
    }

    // The determinant list is replicated on every rank, so every rank takes
    // the same decision here.
    auto sorted_cur = sorted_set(wfn);
    if(sorted_cur == sorted_prev[1]) {
      logger->warn(
          "ASCI Refine is cycling between two determinant sets, E = {:.12e} "
          "and {:.12e} (|dE| = {:.6e} >= REFINE_TOL); keeping the lower",
          E, E_prev, std::abs(E - E_prev));
      if(E_prev < E) {
        wfn = std::move(wfn_prev);
        X = std::move(X_prev);
        E0 = E_prev;
      }
      converged = cycled = true;
      break;
    }
    sorted_prev[1] = std::move(sorted_prev[0]);
    sorted_prev[0] = std::move(sorted_cur);
  }  // Refinement loop

  if(cycled)
    logger->info("ASCI Refine stopped on a 2-cycle at E = {:.12e}", E0);
  else if(converged)
    logger->info("ASCI Refine Converged!");
  else
    throw std::runtime_error("ACCI Refine did not converge");

  return std::make_tuple(E0, wfn, X);
}

}  // namespace macis
