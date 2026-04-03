#define IMPURITY_SOLVER_CPP
#include "macis/impurity_solver.hpp"


namespace macis {

  
  
template <size_t N>
double SolveImpurityED (impurity_params<N>& p){

    size_t& norb =  p.norb;
    double& asci_E0 = p.asci_E0;
    size_t& n_active =  p.n_active;
    size_t& nalpha =  p.nalpha;
    size_t& nbeta =  p.nbeta;
    size_t& n_inactive =  p.n_inactive;
    std::vector<double>& T =  p.T;
    std::vector<double>& Td =  p.Td;
    std::vector<double>& V =  p.V;
    size_t& n_imp =  p.n_imp;
    double& E_core =  p.E_core;
    macis::MCSCFSettings& mcscf_settings =  p.mcscf_settings;
    macis::ASCISettings& asci_settings =  p.asci_settings;

    // std::vector<macis::wfn_t<N>> dets;
    // std::vector<double> C_local;  
    // std::vector<double> occs(n_active, 0);

    std::vector<macis::wfn_t<N>>& dets = p.dets;
    dets.clear();
    std::vector<double>& C_local = p.C;
    C_local.clear();
    std::vector<double>& occs = p.occs;
    occs.assign(n_active, 0);

    std::vector<double>& T_active = p.T_active;
    std::vector<double>& Td_active = p.Td_active;
    std::vector<double>& V_active = p.V_active;
    std::vector<double>& F_inactive = p.F_inactive;
    double& E_inactive = p.E_inactive;

    std::cout << "-------------------------------Entering SolverImpurityED-------------------------------" << std::endl;


    // Storage for active RDMs
    std::vector<double> active_ordm(n_active * n_active);
    std::vector<double> active_ordmd(n_active * n_active);
    std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

    double E0 = 0 ;

    using generator_t = macis::SDBuildHamiltonianGenerator<N>;

    if (p.spin_dep)
      E0 = macis::CASRDMFunctor<generator_t>::rdms(
            mcscf_settings, NumOrbital(n_active), nalpha, nbeta,
            T_active.data(), V_active.data(), active_ordm.data(),
            active_trdm.data(), C_local MACIS_MPI_CODE(, MPI_COMM_WORLD),
            Td_active.data(), active_ordmd.data(), active_trdm.data(),
            active_trdm.data(), active_trdm.data());
    else
      E0 = macis::CASRDMFunctor<generator_t>::rdms(
            mcscf_settings, NumOrbital(n_active), nalpha, nbeta,
            T_active.data(), V_active.data(), active_ordm.data(),
            active_trdm.data(), C_local MACIS_MPI_CODE(, MPI_COMM_WORLD));
    E0 += E_inactive + E_core;
    
    dets = macis::generate_hilbert_space<generator_t::nbits>(
        n_active, nalpha, nbeta);
    
    // Occupation numbers
    for(int i = 0; i < n_active; i++) {
      occs[i] = active_ordm[i + i * n_active]*1./2; 
    }

    return E0;
}

template <size_t N>
double SolveImpurityASCI (impurity_params<N>& p){

    bool& compute_asci_E0 = p.compute_asci_E0;
    double& asci_E0 = p.asci_E0;
    std::string& asci_wfn_fname = p.asci_wfn_fname;
    size_t& norb = p.norb;
    size_t& n_active = p.n_active;
    size_t& nalpha = p.nalpha;
    size_t& nbeta = p.nbeta;
    size_t& n_inactive = p.n_inactive;
    std::vector<double>& T = p.T;
    std::vector<double>& V = p.V;
    size_t& n_imp = p.n_imp;
    double& E_core = p.E_core;
    macis::MCSCFSettings& mcscf_settings = p.mcscf_settings;
    macis::ASCISettings& asci_settings = p.asci_settings;

    std::vector<double>& occs = p.occs;
    occs.assign(n_active, 0);
    std::vector<double>& C_local = p.C;
    C_local.clear();
    std::vector<macis::wfn_t<N>>& dets = p.dets;
    dets.clear();

    std::vector<double>& T_active = p.T_active;
    std::vector<double>& Td_active = p.Td_active;
    std::vector<double>& V_active = p.V_active;
    std::vector<double>& F_inactive = p.F_inactive;
    double& E_inactive = p.E_inactive;

    // Storage for active RDMs
    std::vector<double> active_ordm(n_active * n_active);
    std::vector<double> active_ordmd(n_active * n_active);
    std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

    std::cout << "-------------------------------Entering SolverImpurityASCI-------------------------------" << std::endl;

    double E0 = 0 ;

    using generator_t = macis::SDBuildHamiltonianGenerator<N>;

    generator_t ham_gen(
       macis::matrix_span<double>(T_active.data(), n_active, n_active),
       macis::rank4_span<double>(V_active.data(), n_active, n_active, n_active, n_active));
    
    // Set spin-down one-body matrix if spin-dependent
    if(p.spin_dep) {
      ham_gen.ReadTdo(macis::matrix_span<double>(Td_active.data(), n_active, n_active));
    }
    
    ham_gen.SetJustSingles(p.just_singles);
    ham_gen.SetNimp(n_imp);
    asci_settings.just_singles = p.just_singles;

    if(asci_wfn_fname.size()) 
    {
      // Read wave function from standard file
      // console->info("Reading Guess Wavefunction From {}", asci_wfn_fname);
      std::cout<<"Reading Guess Wavefunction From "<< asci_wfn_fname << std::endl;
      macis::read_wavefunction(asci_wfn_fname, dets, C_local);
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
            tmp += ham_gen.matrix_element(dets[ii], dets[jj]) * C_local[jj];
          }
          E0 += C_local[ii] * tmp;
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
    dets = {macis::canonical_hf_determinant<N>(nalpha, nalpha)};
    // std::cout << dets[0].to_ullong() << std::endl;
    E0 = ham_gen.matrix_element(dets[0], dets[0]);
    C_local = {1.0};
    }
    std::cout<<"ASCI Guess Size = "<< dets.size() << std::endl;
    std::cout<<"ASCI E0 = "<< E0 + E_core + E_inactive << std::endl;
    // console->info("ASCI Guess Size = {}", dets.size());
    // console->info("ASCI E0 = {:.10e}", E0 + E_core + E_inactive);

    //==============PERFORM THE ASCI CALCULATION=========
    {
      // Growth phase
      std::cout << "GROWTH PHASE \n";
      std::tie(E0, dets, C_local) = macis::asci_grow(
          asci_settings, mcscf_settings, E0, std::move(dets), std::move(C_local),
          ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));
      
      // Refinement phase
      std::cout << "REFINEMENT PHASE \n";
      if(asci_settings.max_refine_iter) {
        std::tie(E0, dets, C_local) = macis::asci_refine(
            asci_settings, mcscf_settings, E0, std::move(dets), std::move(C_local),
            ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));
      }
      E0 += E_inactive + E_core;
    } // End ASCI calculation


    // std::cout << "dets.sizet() = " << std::distance(dets.begin(), dets.end()) << " (" << dets.size() << ")" << std::endl;

    ham_gen.form_rdms(dets.begin(),dets.end(),dets.begin(),dets.end(), C_local.data(), 
    macis::matrix_span<double>(active_ordm.data(),n_active,n_active), 
    macis::rank4_span<double>(active_trdm.data(),n_active,n_active,n_active,n_active));

    // Occupation numbers
    // std::vector<double> occs(n_active, 1);
    for(int i = 0; i < n_active; i++) {
      occs[i] = active_ordm[i + i * n_active]*1./2;   //number of electrons in orbital i per spin
      // std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
    }

    return E0;
}

template <size_t N>
double SolveImpurityASCI_rot (impurity_params<N>& p){

    using clock_type = std::chrono::high_resolution_clock;
    using duration_type = std::chrono::duration<double, std::milli>;

    auto start_ASCI_clock = clock_type::now();

    bool& compute_asci_E0 = p.compute_asci_E0;
    double& asci_E0 = p.asci_E0;
    std::string& asci_wfn_fname = p.asci_wfn_fname;
    size_t& norb = p.norb;
    size_t& n_active = p.n_active;
    size_t& nalpha = p.nalpha;
    size_t& nbeta = p.nbeta;
    size_t& n_inactive = p.n_inactive;
    std::vector<double>& T = p.T;
    std::vector<double>& V = p.V;
    size_t& n_imp = p.n_imp;
    double& E_core = p.E_core;
    macis::MCSCFSettings& mcscf_settings = p.mcscf_settings;
    macis::ASCISettings& asci_settings = p.asci_settings;

    std::vector<double>& occs = p.occs;
    occs.assign(n_active, 0);
    std::vector<double>& C_local = p.C;
    C_local.clear();
    std::vector<macis::wfn_t<N>>& dets = p.dets;
    dets.clear();

    std::vector<double> tmp_rot( n_active * n_active, 0. );
    std::vector<double> comp( n_active * n_active, 0. );

    std::vector<double>& orb_rot = p.orb_rot;
    orb_rot.assign( n_active * n_active, 0. );
    for (int i = 0; i < n_active; i++) orb_rot[i + i * n_active] = 1.0;

    std::vector<double>& T_active = p.T_active;
    std::vector<double>& Td_active = p.Td_active;
    std::vector<double>& V_active = p.V_active;
    std::vector<double>& F_inactive = p.F_inactive;
    double& E_inactive = p.E_inactive;

    // Storage for active RDMs
    std::vector<double> active_ordm(n_active * n_active);
    std::vector<double> active_ordmd(n_active * n_active);
    std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

    std::cout << "-------------------------------Entering SolverImpurityASCI_rot (Legacy algorithm)-------------------------------" << std::endl;

    double E0 = 0 ;

    using generator_t = macis::SDBuildHamiltonianGenerator<N>;

    generator_t ham_gen(
       macis::matrix_span<double>(T_active.data(), n_active, n_active),
       macis::rank4_span<double>(V_active.data(), n_active, n_active, n_active, n_active));

    // Set spin-down one-body matrix if spin-dependent
    if(p.spin_dep) {
      ham_gen.ReadTdo(macis::matrix_span<double>(Td_active.data(), n_active, n_active));
    }
    
    ham_gen.SetJustSingles(p.just_singles);
    ham_gen.SetNimp(n_imp);
    asci_settings.just_singles = p.just_singles;

      // HF Guess
      // console->info("Generating HF Guess for ASCI");
      std::cout<<"Generating HF Guess for ASCI \n";
      macis::wfn_t<N> hf_det = macis::canonical_hf_determinant<N>(nalpha, nbeta);
      dets = {hf_det};
      // std::cout << dets[0].to_ullong() << std::endl;
      E0 = ham_gen.matrix_element(dets[0], dets[0]);
      C_local = {1.0};
      std::vector<double> orb_occs(n_active,0.0);
    
    std::cout<<"ASCI Guess Size = "<< dets.size() << std::endl;
    std::cout<<"ASCI E0 = "<< E0 + E_core + E_inactive << std::endl;
    // console->info("ASCI Guess Size = {}", dets.size());
    // console->info("ASCI E0 = {:.10e}", E0 + E_core + E_inactive);

    //==============PERFORM THE ASCI CALCULATION=========
    {

      for (size_t iorb = 0; iorb <= asci_settings.nrots; iorb++)
      {

          auto orbrot_st = clock_type::now();
          
          std::cout<<"\n* Macro It. " << iorb+1 << std::endl;

          //Starting with HF
          dets.clear();
          dets = {hf_det};
          C_local = {1.0};
          E0 = ham_gen.matrix_element(dets[0], dets[0]);
          std::cout<<"ASCI E0 = "<< E0 << std::endl;
          std::cout<<"ASCI E_core = "<< E_core << std::endl;
          std::cout<<"ASCI E_inactive = "<< E_inactive << std::endl;
          std::cout<<"ASCI EHF = "<< E0 + E_core + E_inactive << std::endl;
          std::cout<<"|HF> = " << macis::to_canonical_string(hf_det) << std::endl;

          // Growth phase
          std::cout << "GROWTH PHASE \n";
          std::tie(E0, dets, C_local) = macis::asci_grow(
              asci_settings, mcscf_settings, E0, std::move(dets), std::move(C_local),
              ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));

          // Refinement phase
          std::cout << "REFINEMENT PHASE \n";
          if(asci_settings.max_refine_iter) {
            std::tie(E0, dets, C_local) = macis::asci_refine(
                asci_settings, mcscf_settings, E0, std::move(dets), std::move(C_local),
                ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));
          }
          E0 += E_inactive + E_core;

          
          std::cout<<"\n* @ Macro It. " << iorb+1 << " EASCI: " << E0 << std::endl;

          if (iorb == asci_settings.nrots) break;
          else
          {
            active_ordm.assign( n_active * n_active, 0. );
            active_trdm.assign( n_active * n_active * n_active * n_active, 0. );
            
            

            //Generate RDMs
            ham_gen.form_rdms(dets.begin(),dets.end(),dets.begin(),dets.end(), C_local.data(), 
                macis::matrix_span<double>(active_ordm.data(),n_active,n_active), 
                macis::rank4_span<double>(active_trdm.data(),n_active,n_active,n_active,n_active));

            //Reset auxiliary vectors
            tmp_rot.assign( n_active * n_active, 0. );
            comp.assign( n_active * n_active, 0. );
            //Rotate to new orbitals
            ham_gen.rotate_hamiltonian_ordm_imp_bath( active_ordm.data(), n_imp, tmp_rot.data() , p.spin_dep );
            asci_settings.just_singles = ham_gen.just_singles;
            //Update rotation matrix orb_rot = orb_rot * tmp_rot
            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                      n_active, n_active, n_active, 1.0, orb_rot.data(), n_active,
                      tmp_rot.data(), n_active, 0.0, comp.data(), n_active);
            orb_rot = std::move(comp);
            
            {//Generate new HF determinant in rotated basis
            size_t n_bath = n_active - n_imp;
            //Impurity block
            std::vector<double> ordm_i(n_imp * n_imp);
            std::vector<double> eigvals_i(n_imp);
            //Copy impurity block (active_ordm is column-major)
            for(size_t ii = 0; ii < n_imp; ii++) {
                for(size_t jj = 0; jj < n_imp; jj++) {
                    ordm_i[ii + jj * n_imp] = active_ordm[ii + jj * n_active];
                }
            }
            //Negate for descending eigenvalue order 
            for(auto& x : ordm_i) x *= -1.0;
            //Diagonalize impurity block
            lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, n_imp, ordm_i.data(),
                         n_imp, eigvals_i.data());
            //Restore sign of eigenvalues
            for(auto& x : eigvals_i) x *= -1.0;
            //Bath block
            std::vector<double> ordm_b(n_bath * n_bath);
            std::vector<double> eigvals_b(n_bath);
            //Copy bath block
            for(size_t ii = 0; ii < n_bath; ii++) {
                for(size_t jj = 0; jj < n_bath; jj++) {
                    ordm_b[ii + jj * n_bath] = active_ordm[(ii + n_imp) + (jj + n_imp) * n_active];
                }
            }
            //Negate for descending eigenvalue order
            for(auto& x : ordm_b) x *= -1.0;
            // Diagonalize bath block
            lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, n_bath, ordm_b.data(),
                         n_bath, eigvals_b.data());
            // Restore sign of eigenvalues
            for(auto& x : eigvals_b) x *= -1.0;
            // Store eigenvalues (already in descending order due to sign flip)
            for(size_t ii = 0; ii < n_imp; ii++) {
                orb_occs[ii] = eigvals_i[ii];
            }
            for(size_t ii = n_imp; ii < n_active; ii++) {
                orb_occs[ii] = eigvals_b[ii - n_imp];
            }
            std::cout << "* Impurity and bath 1-RDM eigenvalues: " << std::endl;
            std::cout << "  ";
            for(size_t ii = 0; ii < n_active; ii++) {
                std::cout << " " << orb_occs[ii];
            }
            std::cout << std::endl;
            hf_det = macis::hf_determinant_byocc<N>(nalpha, nbeta, orb_occs);
            }
          
            macis::util::write_matrix(orb_rot.data(), n_active, n_active,
                                     "rot_matrix_" + std::to_string(iorb) + ".dat", true);

            // Rediagonalize
            // E0 = selected_ci_diag( dets.begin(), dets.end(), ham_gen, mcscf_settings.ci_matel_tol,
                      //  mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol, C_local,
                      //  MACIS_MPI_CODE( MPI_COMM_WORLD, ) true, mcscf_settings.ci_nstates);
            
          }

        auto orbrot_en = clock_type::now();
        duration_type total_rot_time = orbrot_en - orbrot_st;
        int minutes = static_cast<int>(total_rot_time.count() / 60000.0);
        double seconds = (total_rot_time.count() / 1000.0) - (minutes * 60.0);
        std::cout << "\n  Total time for ASCI Macro Iteration " << iorb+1 << ": "
                  << minutes << " minutes " << seconds << " seconds" << std::endl;
        }
      }

    // std::cout << "dets.sizet() = " << std::distance(dets.begin(), dets.end()) << " (" << dets.size() << ")" << std::endl;

    if (asci_settings.nrots == 0)
      ham_gen.form_rdms(dets.begin(),dets.end(),dets.begin(),dets.end(), C_local.data(), 
      macis::matrix_span<double>(active_ordm.data(),n_active,n_active), 
      macis::rank4_span<double>(active_trdm.data(),n_active,n_active,n_active,n_active));
    else
      active_ordm = evaluate_ordm( dets, C_local, ham_gen, orb_rot );

    // Occupation numbers
    // std::vector<double> occs(n_active, 1);
    for(int i = 0; i < n_active; i++) {
      occs[i] = active_ordm[i + i * n_active]*1./2;   //number of electrons in orbital i per spin
      std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
    }

    double curr_nel_per_spin = std::accumulate(occs.begin(), occs.begin()+ n_imp, 0.0)/n_imp; //DEBUG
    std::cout << "* Number of electrons on impurity (per orbital per spin) = " << curr_nel_per_spin << std::endl;

    bool print_ordm = true;
    if (print_ordm)
    {
        std::ofstream ofile_ordm( "active_ordm.dat");
        
        ofile_ordm.precision(std::numeric_limits<double>::max_digits10);
        for (int i = 0; i < n_active; i++)
          {
          for (int j = 0; j < n_active; j++)
            ofile_ordm << std::scientific << active_ordm[i + j * n_active] << " ";
          ofile_ordm << std::endl;
          } 
    }
    {
      std::ofstream ofile_rot( "rot_matrix.dat");
      ofile_rot.precision(std::numeric_limits<double>::max_digits10);
      for (int i = 0; i < norb; i++)
      {
        for (int j = 0; j < norb; j++)
          ofile_rot << std::scientific << orb_rot[i + j * n_active] << " ";
        ofile_rot << std::endl;
      }
    }

    auto end_ASCI_clock = clock_type::now();
    duration_type total_ASCI_time = end_ASCI_clock - start_ASCI_clock;
    int minutes = static_cast<int>(total_ASCI_time.count() / 60000.0);
    double seconds = (total_ASCI_time.count() / 1000.0) - (minutes * 60.0);
    std::cout << "\nTotal time to complete ASCI GS calculation: \n"
              << minutes << " minutes " << seconds << " seconds" << std::endl;
          
    return E0;
}

template <size_t N>
double SolveImpurityCheapASCI (impurity_params<N>& p){

    bool& compute_asci_E0 = p.compute_asci_E0;
    double& asci_E0 = p.asci_E0;
    std::string& asci_wfn_fname = p.asci_wfn_fname;
    size_t& norb = p.norb;
    size_t& n_active = p.n_active;
    size_t& nalpha =  p.nalpha;
    size_t& nbeta =  p.nbeta;
    size_t& n_inactive =  p.n_inactive;
    std::vector<double>& T =  p.T;
    std::vector<double>& V =  p.V;
    size_t& n_imp =  p.n_imp;
    double& E_core =  p.E_core;
    macis::MCSCFSettings& mcscf_settings =  p.mcscf_settings;
    macis::ASCISettings& asci_settings =  p.asci_settings;

    std::vector<macis::wfn_t<N>>& dets =  p.dets;
    std::vector<double>& C_local = p.C;
    C_local.clear();
    std::vector<double>& occs = p.occs;
    occs.assign(n_active, 0);

    std::vector<double>& T_active = p.T_active;
    std::vector<double>& Td_active = p.Td_active;
    std::vector<double>& V_active = p.V_active;
    std::vector<double>& F_inactive = p.F_inactive;
    double& E_inactive = p.E_inactive;

    // Storage for active RDMs
    std::vector<double> active_ordm(n_active * n_active);
    std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

    std::cout << "-------------------------------Entering SolverImpurityCheapASCI-------------------------------" << std::endl;

    double E0 = 0 ;

    using generator_t = macis::SDBuildHamiltonianGenerator<N>;

    generator_t ham_gen(
       macis::matrix_span<double>(T_active.data(), n_active, n_active),
       macis::rank4_span<double>(V_active.data(), n_active, n_active, n_active, n_active));
    
    // Set spin-down one-body matrix if spin-dependent
    if(p.spin_dep) {
      ham_gen.ReadTdo(macis::matrix_span<double>(Td_active.data(), n_active, n_active));
    }
    
    ham_gen.SetJustSingles(p.just_singles);
    ham_gen.SetNimp(n_imp);
    asci_settings.just_singles = p.just_singles;

    E0 =
      selected_ci_diag(dets.begin(), dets.end(), ham_gen, mcscf_settings.ci_matel_tol,
                       mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol, C_local,
                       MACIS_MPI_CODE( MPI_COMM_WORLD, ) true, mcscf_settings.ci_nstates);
    E0 += E_inactive + E_core;
    
    ham_gen.form_rdms(dets.begin(),dets.end(),dets.begin(),dets.end(), C_local.data(), 
                macis::matrix_span<double>(active_ordm.data(),n_active,n_active), 
                macis::rank4_span<double>(active_trdm.data(),n_active,n_active,n_active,n_active));

    // Occupation numbers
    // std::vector<double> occs(n_active, 1);
    for(int i = 0; i < n_active; i++) {
      occs[i] = active_ordm[i + i * n_active]*1./2;   //number of electrons in orbital i per spin
      // std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
    }

    return E0;
}


// Explicit template instantiations for commonly used template parameter
template double SolveImpurityED<64>(impurity_params<64>& p);
template double SolveImpurityASCI<64>(impurity_params<64>& p);
template double SolveImpurityASCI_rot<64>(impurity_params<64>& p);
template double SolveImpurityCheapASCI<64>(impurity_params<64>& p);

} // namespace macis
