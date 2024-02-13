#define CALL_SOLVER_CPP
#include "macis/doping/call_solver.hpp"
#include "macis/doping/fix_mu.hpp"

namespace macis {

double SolveImpurityED (void * params){


    // struct fix_mu_params *p = (struct fix_mu_params *)params;
    struct fix_mu_params *p = static_cast<fix_mu_params*> (params);

    size_t norb = * (p->norb);
    size_t n_active = * (p->n_active);
    size_t nalpha = * (p->nalpha);
    size_t nbeta = * (p->nbeta);
    size_t n_inactive = * (p->n_inactive);
    std::vector<double> T = * (p->T);
    std::vector<double> V = *(p->V);
    bool just_singles = *(p->just_singles);
    size_t n_imp = *(p->n_imp);
    double E_core = *(p->E_core);
    macis::MCSCFSettings mcscf_settings = *(p->mcscf_settings);
    macis::ASCISettings asci_settings = *(p->asci_settings);


    // std::vector<double> * T_ptr = p->T;
    // for(int i = 0; i < norb; i++) {
    //     std::cout << "T_diagonal_" << i << " = " << T_ptr->at(i+norb*i) << std::endl;
    // }

   std::vector<macis::wfn_t<nwfn_bits>> dets;
   std::vector<double> C;  
   std::vector<double> occs(n_active, 0);

    // std::vector<double> occs = *p->occs;
    // std::vector<double> C = *p->C;
    // std::vector<macis::wfn_t<nwfn_bits>> dets = *p->dets;


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

    using generator_t = macis::SDBuildHamiltonianGenerator<nwfn_bits>;

    generator_t ham_gen(
       macis::matrix_span<double>(T.data(), norb, norb),
       macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
    ham_gen.SetJustSingles(just_singles);
    ham_gen.SetNimp(n_imp);
    
    
    std::vector<double> C_local;
    dets = macis::generate_hilbert_space<nwfn_bits>(norb, nalpha, nbeta);
    E0 = macis::selected_ci_diag(
          dets.begin(), dets.end(), ham_gen, mcscf_settings.ci_matel_tol,
          mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol, C_local,
          MACIS_MPI_CODE(MPI_COMM_WORLD, ) true);


    E0 += E_inactive + E_core;
    ham_gen.form_rdms(
          dets.begin(), dets.end(), dets.begin(), dets.end(), C_local.data(),
          macis::matrix_span<double>(active_ordm.data(), norb, norb),
          macis::rank4_span<double>(active_trdm.data(), norb, norb, norb, norb));
    
    
    C=C_local;

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


double SolveImpurityASCI (void * params){


    // struct fix_mu_params *p = (struct fix_mu_params *)params;
    struct fix_mu_params *p = static_cast<fix_mu_params*> (params);

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
    bool just_singles = *(p->just_singles);
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

    using generator_t = macis::SDBuildHamiltonianGenerator<nwfn_bits>;

    generator_t ham_gen(
       macis::matrix_span<double>(T.data(), norb, norb),
       macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
    ham_gen.SetJustSingles(just_singles);
    ham_gen.SetNimp(n_imp);

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
        std::cout<<"Calculating E0"<<std::endl;
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
        std::cout<<"Reading E0"<<std::endl;
        E0 = asci_E0 - E_core - E_inactive;
      }
    } 
    else 
    {
    // HF Guess
    // console->info("Generating HF Guess for ASCI");
    std::cout<<"Generating HF Guess for ASCI"<<std::endl;
    dets = {macis::canonical_hf_determinant<nwfn_bits>(nalpha, nalpha)};
    // std::cout << dets[0].to_ullong() << std::endl;
    E0 = ham_gen.matrix_element(dets[0], dets[0]);
    C = {1.0};
    }

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
