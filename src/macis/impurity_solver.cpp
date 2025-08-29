#define IMPURITY_SOLVER_CPP
#include "macis/impurity_solver.hpp"


namespace macis {

double SolveImpurityED (void * params){


    struct impurity_params *p = static_cast<impurity_params*> (params);

    size_t norb = * (p->norb);
    size_t n_active = * (p->n_active);
    size_t nalpha = * (p->nalpha);
    size_t nbeta = * (p->nbeta);
    size_t n_inactive = * (p->n_inactive);
    std::vector<double> T = * (p->T);
    std::vector<double> V = *(p->V);
    size_t n_imp = *(p->n_imp);
    double E_core = *(p->E_core);
    macis::MCSCFSettings mcscf_settings = *(p->mcscf_settings);
    macis::ASCISettings asci_settings = *(p->asci_settings);

    size_t norb2 = norb * norb;
    size_t norb3 = norb2 * norb;
    size_t norb4 = norb2 * norb2;   

    std::vector<macis::wfn_t<nwfn_bits>> dets;
    std::vector<double> C;  
    std::vector<double> occs(n_active, 0);

    // Copy integrals into active subsets 
    std::vector<double> T_active(n_active * n_active);
    std::vector<double> V_active(n_active * n_active * n_active * n_active) ;
    // Compute active-space Hamiltonian and inactive Fock matrix
    std::vector<double> F_inactive(norb2);
    macis::active_hamiltonian(NumOrbital(norb), NumActive(n_active),
                              NumInactive(n_inactive), T.data(), norb, V.data(),
                              norb, F_inactive.data(), norb, T_active.data(),
                              n_active, V_active.data(), n_active) ;




    // Compute Inactive energy
    auto E_inactive = macis::inactive_energy(NumInactive(n_inactive), T.data(),
                                             norb, F_inactive.data(), norb);
    // console->info("E(inactive) = {:.12f}", E_inactive);
    std::cout<<"E(inactive) = "<< E_inactive << std::endl;

    // Storage for active RDMs
    std::vector<double> active_ordm(n_active * n_active);
    std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

    double E0 = 0 ;

    using generator_t = macis::DoubleLoopHamiltonianGenerator<nwfn_bits>;

    std::vector<double> C_local;
    E0 = macis::CASRDMFunctor<generator_t>::rdms(
            mcscf_settings, NumOrbital(n_active), nalpha, nbeta,
            T_active.data(), V_active.data(), active_ordm.data(),
            active_trdm.data(), C_local MACIS_MPI_CODE(, MPI_COMM_WORLD));
    E0 += E_inactive + E_core;
    
    dets = macis::generate_hilbert_space<generator_t::nbits>(
        n_active, nalpha, nbeta);
    
    C=C_local;

    // Occupation numbers
    for(int i = 0; i < n_active; i++) {
      occs[i] = active_ordm[i + i * n_active]; 
    }

    *(p->occs) = occs;
    *(p->C) = C;
    *(p->dets) = dets;



    
    return E0;
}


double SolveImpurityASCI (void * params){


    // struct impurity_params *p = (struct impurity_params *)params;
    struct impurity_params *p = static_cast<impurity_params*> (params);

    bool compute_asci_E0 = *(p->compute_asci_E0);
    double asci_E0 = *(p->asci_E0);
    std::string asci_wfn_fname = *(p->asci_wfn_fname);
    size_t norb = *(p->norb);
    size_t n_active =* (p->n_active);
    size_t nalpha = *(p->nalpha);
    size_t nbeta = *(p->nbeta);
    size_t n_inactive = *(p->n_inactive);
    std::vector<double> T = *(p->T);
    std::vector<double> V = *(p->V);
    size_t n_imp = *(p->n_imp);
    double E_core = *(p->E_core);
    macis::MCSCFSettings mcscf_settings = *(p->mcscf_settings);
    macis::ASCISettings asci_settings = *(p->asci_settings);
    std::vector<double> occs = *(p->occs);
    std::vector<double> C = *(p->C);
    std::vector<macis::wfn_t<nwfn_bits>> dets = *(p->dets);

    size_t norb2 = norb * norb;
    size_t norb3 = norb2 * norb;
    size_t norb4 = norb2 * norb2;   

    // Copy integrals into active subsets
    std::vector<double> T_active(n_active * n_active);
    std::vector<double> V_active(n_active * n_active * n_active * n_active) ;
    // Compute active-space Hamiltonian and inactive Fock matrix
    std::vector<double> F_inactive(norb2);
    macis::active_hamiltonian(NumOrbital(norb), NumActive(n_active),
                              NumInactive(n_inactive), T.data(), norb, V.data(),
                              norb, F_inactive.data(), norb, T_active.data(),
                              n_active, V_active.data(), n_active) ;

    // Compute Inactive energy
    auto E_inactive = macis::inactive_energy(NumInactive(n_inactive), T.data(),
                                             norb, F_inactive.data(), norb);

    // Storage for active RDMs
    std::vector<double> active_ordm(n_active * n_active);
    std::vector<double> active_trdm;

    double E0 = 0 ;

    using generator_t = macis::DoubleLoopHamiltonianGenerator<nwfn_bits>;

    generator_t ham_gen(
       macis::matrix_span<double>(T_active.data(), n_active, n_active),
       macis::rank4_span<double>(V_active.data(), n_active, n_active, n_active, n_active));

    std::cout<<"this line exists and asci_wfn_name = "<< asci_wfn_fname << std::endl;
    if(asci_wfn_fname.size()) 
    {
      // Read wave function from standard file
      // console->info("Reading Guess Wavefunction From {}", asci_wfn_fname);
      std::cout<<"Reading Guess Wavefunction From "<< asci_wfn_fname << std::endl;
      macis::read_wavefunction(asci_wfn_fname, dets, C);
      // std::cout << dets[0].to_ullong() << std::endl;
      if(compute_asci_E0) 
      {
        // console->info("*  Calculating E0");
        std::cout<<"*  Calculating E0 \n";
        E0 = 0;
        for(auto ii = 0; ii < dets.size(); ++ii) 
        {
          double tmp = 0.0;
          for(auto jj = 0; jj < dets.size(); ++jj) 
          {
            tmp += ham_gen.matrix_element(dets[ii], dets[jj]) * C[jj];
          }
          E0 += C[ii] * tmp;
        }
      } 
      else 
      {
        // console->info("*  Reading E0");
        std::cout<<"*  Reading E0 \n";
        E0 = asci_E0 - E_core - E_inactive;
      }
    } 
    else 
    {
    // HF Guess
    // console->info("Generating HF Guess for ASCI");
    std::cout<<"Generating HF Guess for ASCI \n";
    dets = {macis::canonical_hf_determinant<nwfn_bits>(nalpha, nalpha)};
    // std::cout << dets[0].to_ullong() << std::endl;
    E0 = ham_gen.matrix_element(dets[0], dets[0]);
    C = {1.0};
    }
    std::cout<<"ASCI Guess Size = "<< dets.size() << std::endl;
    std::cout<<"ASCI E0 = "<< E0 + E_core + E_inactive << std::endl;
    // console->info("ASCI Guess Size = {}", dets.size());
    // console->info("ASCI E0 = {:.10e}", E0 + E_core + E_inactive);

    // Perform the ASCI calculation
    // Growth phase
    std::cout << "GROWTH PHASE \n";
    std::tie(E0, dets, C) = macis::asci_grow(
        asci_settings, mcscf_settings, E0, std::move(dets), std::move(C),
        ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));
    // Refinement phase
    std::cout << "REFINEMENT PHASE \n";
    if(asci_settings.max_refine_iter) {
      std::tie(E0, dets, C) = macis::asci_refine(
          asci_settings, mcscf_settings, E0, std::move(dets), std::move(C),
          ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));
    }
    E0 += E_inactive + E_core;
    


    // std::cout << "dets.sizet() = " << std::distance(dets.begin(), dets.end()) << " (" << dets.size() << ")" << std::endl;

    ham_gen.form_rdms(dets.begin(),dets.end(),dets.begin(),dets.end(), C.data(), 
    macis::matrix_span<double>(active_ordm.data(),n_active,n_active), 
    macis::rank4_span<double>(active_trdm.data(),n_active,n_active,n_active,n_active));

    // Occupation numbers
    // std::vector<double> occs(n_active, 1);
    for(int i = 0; i < n_active; i++) {
      occs[i] = active_ordm[i + i * n_active]; 
      // std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
    }

    *(p->occs) = occs;
    *(p->C) = C;
    *(p->dets) = dets;

          
    return E0;
}
} // namespace macis
