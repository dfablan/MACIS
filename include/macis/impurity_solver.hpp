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

struct impurity_params
{
    size_t* n_active;
    size_t* nbeta;
    size_t* nalpha;
    size_t* n_inactive;
    size_t* norb;
    double* nel;
    size_t* n_imp;
    
    double *abs_tol;
    size_t *maxiter;
    bool *print;
    double *init_shift;

    std::string *ci_exp;
    std::string *asci_wfn_fname;
    bool *compute_asci_E0;
    double *asci_E0;


    macis::MCSCFSettings* mcscf_settings;
    macis::ASCISettings* asci_settings;
    std::vector<double>* occs;
    std::vector<double>* C;
    std::vector<macis::wfn_t<nwfn_bits>>* dets;
    
    double* dstep;
    double* E_core;
    double* E;
    std::vector<double>* T;
    std::vector<double>* V;    
    bool* just_singles;

};


double SolveImpurityED(void *params);

double SolveImpurityASCI(void *params);

} // namespace macis