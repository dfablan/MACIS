#include <iomanip>
#include <iostream>
#include <macis/asci/grow.hpp>
#include <macis/asci/refine.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/hamiltonian_generator/sd_build.hpp>
#include <macis/util/cas.hpp>
#include <macis/util/detail/rdm_files.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/util/fock_matrices.hpp>
#include <macis/util/memory.hpp>
#include <macis/util/moller_plesset.hpp>
#include <macis/util/mpi.hpp>
#include <macis/util/transform.hpp>
#include <macis/wavefunction_io.hpp>
#include <map>
#include <sparsexx/io/write_dist_mm.hpp>

using macis::NumActive;
using macis::NumCanonicalOccupied;
using macis::NumCanonicalVirtual;
using macis::NumElectron;
using macis::NumInactive;
using macis::NumOrbital;
using macis::NumVirtual;

constexpr size_t nwfn_bits = 64;

namespace macis {

/**
 * @brief Structure to hold the parameters of the impurity problem.
 */

double Comp_db_occs(void* params) {
  struct impurity_params* p = static_cast<impurity_params*>(params);

  norb = *(p->norb);
  n_active = *(p->n_active);
  norb2 = norb * norb;
  norb3 = norb2 * norb;
  norb4 = norb2 * norb2;
  macis::ASCISettings asci_settings = *(p->asci_settings);

  std::vector<double> T = *(p->T);
  std::vector<double> V = *(p->V);

  using generator_t = macis::DoubleLoopHamiltonianGenerator<nwfn_bits>;

  generator_t ham_gen(
      matrix_span_t(T.data(), n_active, n_active),
      rank4_span_t(V.data(), n_active, n_active, n_active, n_active));

  double orb_db_occs;

  if(asci_settings.nrots == 0) {
    ham_gen.form_rdms
  }
}

}  // namespace macis