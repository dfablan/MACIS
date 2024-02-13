/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/asci/iteration.hpp>
#include <macis/wavefunction_io.hpp>

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

  const std::string fmt_string = "iter = {:4}, E0 = {:20.12e}, dE = {:14.6e}";

  logger->info(fmt_string, 0, E0, 0.0);

  // Refinement Loop
  const size_t ndets = wfn.size();
  bool converged = false;
  for(size_t iter = 0; iter < asci_settings.max_refine_iter; ++iter) {
    double E;
    // macis::write_wavefunction("dets_bef_refine.dat", 12, wfn, X);
    //    std::cout << "E0 = " << E0 << std::endl ;
    //    std::cout << "E = " << E << std::endl ;
    //    std::cout << "ndets = " << ndets << std::endl ;
    std::tie(E, wfn, X) = asci_iter<N, index_t>(
        asci_settings, mcscf_settings, ndets, E0, std::move(wfn), std::move(X),
        ham_gen, norb MACIS_MPI_CODE(, comm));
    //    std::cout << "E0 = " << E0 << std::endl ;
    //    std::cout << "E = " << E << std::endl ;
    // macis::write_wavefunction("dets_aft_refine.dat", 12, wfn, X);
    if(wfn.size() != ndets)
      throw std::runtime_error("Wavefunction size can't change in refinement");

    const auto E_delta = E - E0;
    logger->info(fmt_string, iter + 1, E, E_delta);
    E0 = E;
    if(std::abs(E_delta) < asci_settings.refine_energy_tol) {
      converged = true;
      break;
    }
  }  // Refinement loop

  if(converged)
    logger->info("ASCI Refine Converged!");
  else
    throw std::runtime_error("ACCI Refine did not converge");

  return std::make_tuple(E0, wfn, X);
}

}  // namespace macis
