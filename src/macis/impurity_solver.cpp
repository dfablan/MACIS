#define IMPURITY_SOLVER_CPP
#include <sstream>

#include "macis/asci/determinant_symmetry.hpp"
#include "macis/impurity_solver.hpp"


namespace macis {

namespace {

/**
 *  @brief Load an ASCI guess wavefunction from p.asci_wfn_fname, if one was requested.
 *
 *  Returns true when a guess was loaded, in which case @p dets, @p C_local and @p E0
 *  are set from the file. Returns false when no guess file was requested, leaving all
 *  three untouched so the caller falls back to its own HF reference.
 *
 *  Every precondition below is a throw rather than a warning: a guess that is silently
 *  ignored or silently misapplied produces a plausible-looking energy for a calculation
 *  nobody asked for, which is exactly the failure this path exists to prevent.
 */
template <size_t N>
bool load_asci_guess(impurity_params<N>& p, std::vector<macis::wfn_t<N>>& dets,
                     std::vector<double>& C_local, double& E0) {
  const std::string& fname = p.asci_wfn_fname;
  if(fname.empty()) return false;

  // A determinant list is meaningful only relative to an orbital basis. With NROTS > 0
  // the solver rotates into the natural-orbital basis of the current macro iteration,
  // while the file was written in whatever basis produced it, so the two do not
  // correspond and the guess would be wrong rather than merely useless. With NROTS == 0
  // no rotation is ever applied (orb_rot stays the identity) and file and FCIDUMP share
  // a basis.
  if(p.asci_settings.nrots > 0)
    throw std::runtime_error(
        "ASCI.WFN_FILE with ASCI.NROTS > 0 is not supported: the guess is expressed in "
        "the orbital basis of the run that wrote it, while NROTS > 0 rotates into the "
        "natural-orbital basis. Set NROTS = 0 or drop ASCI.WFN_FILE.");

  // Recomputing E0 = <C|H|C> for a guess is a dense O(ndets^2) contraction over
  // matrix_element; at production NTDETS_MAX that costs far more than the cold start it
  // is meant to replace. Whoever wrote the guess already knows its energy, so require it.
  if(p.compute_asci_E0)
    throw std::runtime_error(
        "ASCI.WFN_FILE requires ASCI.E0_WFN: recomputing E0 for a guess wavefunction is "
        "an O(ndets^2) dense contraction and is not viable at production ndets. Supply "
        "the total energy reported by the run that wrote the guess.");

  // asci_grow only iterates while wfn.size() < ntdets_max, so a converged guess -- which
  // is at ntdets_max by construction -- skips growth entirely and refinement is the only
  // stage left that diagonalizes anything. Since E0 is now supplied rather than computed,
  // MAX_REFINE_ITER = 0 would return ASCI.E0_WFN verbatim as the answer.
  if(p.asci_settings.max_refine_iter == 0)
    throw std::runtime_error(
        "ASCI.WFN_FILE requires ASCI.MAX_REFINE_ITER > 0: a converged guess is already "
        "at ASCI.NTDETS_MAX, so asci_grow is a no-op and refinement is the only stage "
        "that would diagonalize the guess; with MAX_REFINE_ITER = 0 the solver would "
        "return ASCI.E0_WFN unchanged.");

  std::cout << "Reading Guess Wavefunction From " << fname << std::endl;
  const auto header = macis::read_wavefunction(fname, dets, C_local,
                                              /* check_orbital_bounds = */ true);

  // ASCI generates only single and double excitations, which conserve the per-spin
  // particle number, so a guess in the wrong (N, Sz) sector keeps the entire expansion
  // in that sector -- no error, just the energy of a filling that was never requested.
  for(size_t i = 0; i < dets.size(); ++i) {
    const size_t na = macis::bitset_lo_word(dets[i]).count();
    const size_t nb = macis::bitset_hi_word(dets[i]).count();
    if(na != p.nalpha or nb != p.nbeta)
      throw std::runtime_error(
          "Guess wavefunction " + fname + ": determinant " + std::to_string(i) +
          " has (" + std::to_string(na) + ", " + std::to_string(nb) +
          ") electrons, but this run requested (" + std::to_string(p.nalpha) + ", " +
          std::to_string(p.nbeta) + ")");
  }

  if(header.norb != p.n_active)
    std::cout << "WARNING: guess wavefunction " << fname << " was written for "
              << header.norb << " active orbitals, but this run uses " << p.n_active
              << std::endl;

  // ASCI.E0_WFN is a total energy -- the value this driver reports -- while the solvers
  // work with the active-space reference. Subtracting the *current* E_core/E_inactive is
  // the correct conversion when the bath or mu has moved since the guess was written.
  E0 = p.asci_E0 - p.E_core - p.E_inactive;
  std::cout << "*  Reading E0 \n";

  return true;
}

/**
 *  @brief Validate and finalize orbital-permutation symmetry enforcement
 *  (ASCI.SYMMETRIZE_DETS) at solver entry.
 *
 *  On entry p.asci_settings.sym_group holds the generator permutations read
 *  from the input file; on exit it holds the full expanded group (identity
 *  included). Each generator is validated as an exact symmetry of the
 *  active-space integrals within sym_tol. The check is made on the integrals
 *  themselves, never inferred from any assumption about the bath structure:
 *  any bath that is not permutation-symmetric fails here, whatever produced
 *  it.
 *
 *  Idempotent: expanding an already-expanded group returns the same group, so
 *  repeated solver entries (e.g. the mu root-find wrappers) are safe.
 */
template <size_t N>
void prepare_det_symmetry(impurity_params<N>& p) {
  auto& stg = p.asci_settings;
  if(!stg.symmetrize_dets) return;

  if(stg.nrots > 0 or stg.grow_with_rot)
    throw std::runtime_error(
        "ASCI.SYMMETRIZE_DETS requires ASCI.NROTS = 0 and ASCI.GROW_WITH_ROT "
        "= FALSE: natural-orbital rotations destroy the flavor labels the "
        "permutation group acts on.");

  if(!stg.sym_group or stg.sym_group->empty())
    throw std::runtime_error(
        "ASCI.SYMMETRIZE_DETS = TRUE but no permutations were supplied "
        "(ASCI.SYM_NPERM / ASCI.SYM_PERM_i).");

  const size_t n = p.n_active;
  const size_t n2 = n * n;
  const size_t n3 = n2 * n;
  const auto& gens = *stg.sym_group;

  const auto hf_det = macis::canonical_hf_determinant<N>(p.nalpha, p.nbeta);

  for(size_t ig = 0; ig < gens.size(); ++ig) {
    const auto& g = gens[ig];
    const std::string gname = "permutation " + std::to_string(ig + 1);

    if(!macis::is_valid_permutation(g, n))
      throw std::runtime_error("ASCI.SYMMETRIZE_DETS: " + gname +
                               " is not a bijection on the " +
                               std::to_string(n) + " active orbitals.");

    // One-body invariance: T (and Td if spin-dependent)
    auto check_1body = [&](const std::vector<double>& T, const char* name) {
      double max_dev = 0.0;
      for(size_t q = 0; q < n; ++q)
        for(size_t r = 0; r < n; ++r)
          max_dev =
              std::max(max_dev, std::abs(T[g[r] + g[q] * n] - T[r + q * n]));
      if(max_dev > stg.sym_tol) {
        std::ostringstream oss;
        oss << "ASCI.SYMMETRIZE_DETS: " << gname << " is not a symmetry of "
            << name << ": max deviation " << std::scientific << max_dev
            << " exceeds SYM_TOL = " << stg.sym_tol;
        throw std::runtime_error(oss.str());
      }
    };
    check_1body(p.T_active, "T_active");
    if(p.spin_dep) check_1body(p.Td_active, "Td_active");

    // Two-body invariance (full n^4 sweep). The linearization convention is
    // immaterial: the permutation is applied to all four indices, so
    // invariance in one per-index linearization is invariance in any.
    {
      const auto& V = p.V_active;
      double max_dev = 0.0;
      for(size_t l = 0; l < n; ++l)
        for(size_t k = 0; k < n; ++k)
          for(size_t q = 0; q < n; ++q)
            for(size_t r = 0; r < n; ++r) {
              const double dev =
                  std::abs(V[g[r] + g[q] * n + g[k] * n2 + g[l] * n3] -
                           V[r + q * n + k * n2 + l * n3]);
              if(dev > max_dev) max_dev = dev;
            }
      if(max_dev > stg.sym_tol) {
        std::ostringstream oss;
        oss << "ASCI.SYMMETRIZE_DETS: " << gname
            << " is not a symmetry of V_active: max deviation "
            << std::scientific << max_dev
            << " exceeds SYM_TOL = " << stg.sym_tol;
        throw std::runtime_error(oss.str());
      }
    }

    // The HF reference must map to itself, otherwise the initial seed cannot
    // be closed without changing the reference
    if(macis::permute_orbitals(hf_det, g) != hf_det)
      throw std::runtime_error(
          "ASCI.SYMMETRIZE_DETS: " + gname +
          " does not leave the HF reference determinant invariant. The "
          "filling must close whole flavor multiplets (nalpha - n_imp must "
          "fill complete band groups in the bath prefix).");
  }

  // Expand the generators to the full group and store it for the solvers
  const size_t ngen = gens.size();
  auto group = macis::expand_permutation_group(gens, n);
  stg.sym_group = std::make_shared<const std::vector<std::vector<uint32_t>>>(
      std::move(group));
  std::cout << "* SYMMETRIZE_DETS: " << ngen
            << " input permutation(s) expanded to a group of order "
            << stg.sym_group->size() << std::endl;
}

/**
 *  @brief Close a guess wavefunction under the orbital-permutation group.
 *
 *  Added orbit partners get C = 0: zero-coefficient determinants leave the
 *  guess energy <C|H|C> unchanged, and Davidson re-solves the coefficients in
 *  the closed space anyway.
 */
template <size_t N>
void close_guess_under_group(std::vector<macis::wfn_t<N>>& dets,
                             std::vector<double>& C,
                             const std::vector<std::vector<uint32_t>>& group) {
  std::map<macis::wfn_t<N>, double, macis::bitset_less_comparator<N>> closed;
  for(size_t i = 0; i < dets.size(); ++i) closed.emplace(dets[i], C[i]);
  const size_t n_orig = closed.size();
  for(size_t i = 0; i < dets.size(); ++i)
    for(const auto& g : group)
      closed.emplace(macis::permute_orbitals(dets[i], g), 0.0);
  if(closed.size() == n_orig) return;

  std::cout << "* SYMMETRIZE_DETS: guess wavefunction was not closed under "
               "the permutation group; added "
            << closed.size() - n_orig << " determinants with C = 0 ("
            << n_orig << " -> " << closed.size() << ")" << std::endl;
  dets.clear();
  C.clear();
  dets.reserve(closed.size());
  C.reserve(closed.size());
  for(const auto& [d, c] : closed) {
    dets.push_back(d);
    C.push_back(c);
  }
}

/**
 *  @brief Log the maximum deviation of the 1-RDM from the enforced
 *  orbital-permutation symmetry. Warns but never throws: Davidson-tolerance
 *  asymmetry is expected.
 */
template <size_t N>
void check_rdm_symmetry(const impurity_params<N>& p,
                        const std::vector<double>& ordm) {
  const auto& stg = p.asci_settings;
  if(!stg.symmetrize_dets or !stg.sym_group) return;
  const size_t n = p.n_active;
  double max_dev = 0.0;
  for(const auto& g : *stg.sym_group)
    for(size_t q = 0; q < n; ++q)
      for(size_t r = 0; r < n; ++r)
        max_dev = std::max(max_dev,
                           std::abs(ordm[g[r] + g[q] * n] - ordm[r + q * n]));
  std::cout << "* SYMMETRIZE_DETS: max 1-RDM symmetry deviation = " << max_dev
            << std::endl;
  if(max_dev > 1e-6)
    std::cout << "WARNING: 1-RDM breaks the enforced permutation symmetry by "
              << max_dev << " (> 1e-6); check Davidson convergence."
              << std::endl;
}

}  // namespace


  
  
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

    // Validate and expand the orbital-permutation symmetry group, if enabled
    prepare_det_symmetry(p);

    // A guess wavefunction seeds the expansion when ASCI.WFN_FILE is set; otherwise
    // fall back to the HF reference. load_asci_guess validates the guess and throws on
    // any condition under which it could not be applied faithfully.
    if(!load_asci_guess(p, dets, C_local, E0))
    {
      // HF Guess
      std::cout<<"Generating HF Guess for ASCI \n";
      dets = {macis::canonical_hf_determinant<N>(nalpha, nbeta)};
      E0 = ham_gen.matrix_element(dets[0], dets[0]);
      C_local = {1.0};
    }
    else if(asci_settings.symmetrize_dets and asci_settings.sym_group)
    {
      // The HF fallback is closed by construction (validated in
      // prepare_det_symmetry); a guess from file need not be
      close_guess_under_group(dets, C_local, *asci_settings.sym_group);
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

    // Health metric: how well the 1-RDM respects the enforced symmetry
    check_rdm_symmetry(p, active_ordm);

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

    // Validate and expand the orbital-permutation symmetry group, if enabled.
    // Rejects NROTS > 0, so with symmetrization on the macro loop below runs
    // exactly once and the post-rotation reseed path is unreachable.
    prepare_det_symmetry(p);

      // hf_det is needed whether or not a guess is loaded: it is the reference the
      // macro-iteration loop restarts from in each rotated basis, refreshed via
      // hf_determinant_byocc after every rotation below.
      macis::wfn_t<N> hf_det = macis::canonical_hf_determinant<N>(nalpha, nbeta);
      std::vector<double> orb_occs(n_active,0.0);

      // A guess only seeds the first macro iteration. load_asci_guess requires
      // NROTS == 0, so with a guess that first iteration is also the only one.
      const bool have_guess = load_asci_guess(p, dets, C_local, E0);
      if(!have_guess)
      {
        // HF Guess
        std::cout<<"Generating HF Guess for ASCI \n";
        dets = {hf_det};
        E0 = ham_gen.matrix_element(dets[0], dets[0]);
        C_local = {1.0};
      }
      else if(asci_settings.symmetrize_dets and asci_settings.sym_group)
      {
        // The HF reference is closed by construction (validated in
        // prepare_det_symmetry); a guess from file need not be
        close_guess_under_group(dets, C_local, *asci_settings.sym_group);
      }
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

          //Starting with HF. A guess wavefunction survives into the first macro
          //iteration only: from the second onwards its determinants refer to the
          //previous orbital basis, so the HF reference in the current basis is the
          //only valid restart. (load_asci_guess requires NROTS == 0, so a guess
          //never actually reaches iorb > 0 -- the condition states the invariant.)
          if(iorb > 0 or !have_guess)
          {
            dets.clear();
            dets = {hf_det};
            C_local = {1.0};
            E0 = ham_gen.matrix_element(dets[0], dets[0]);
          }
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
            //Rotate to new orbitals. orb_occs receives the natural-orbital
            //occupations (imp block then bath block, each descending) of
            //the exact basis this call rotates the Hamiltonian into, so
            //hf_determinant_byocc below is guaranteed to be consistent with
            //that basis -- re-diagonalizing the ordm blocks separately here
            //could pick a different (but equally valid) basis within any
            //degenerate/near-degenerate occupation subspace, silently
            //decoupling the HF guess from the rotated Hamiltonian.
            ham_gen.rotate_hamiltonian_ordm_imp_bath( active_ordm.data(), n_imp, tmp_rot.data() , p.spin_dep, orb_occs.data() );
            asci_settings.just_singles = ham_gen.just_singles;
            //Update rotation matrix orb_rot = orb_rot * tmp_rot
            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                      n_active, n_active, n_active, 1.0, orb_rot.data(), n_active,
                      tmp_rot.data(), n_active, 0.0, comp.data(), n_active);
            orb_rot = std::move(comp);
            
            {//Generate new HF determinant in rotated basis
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

    // Health metric: how well the 1-RDM respects the enforced symmetry
    // (symmetrization forces nrots == 0, so this is the unrotated 1-RDM)
    check_rdm_symmetry(p, active_ordm);

    // Occupation numbers
    // std::vector<double> occs(n_active, 1);
    for(int i = 0; i < n_active; i++) {
      occs[i] = active_ordm[i + i * n_active]*1./2;   //number of electrons in orbital i per spin
      std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
    }

    double curr_nel_per_spin = std::accumulate(occs.begin(), occs.begin()+ n_imp, 0.0)/n_imp;
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
      for (int i = 0; i < n_active; i++)
      {
        for (int j = 0; j < n_active; j++)
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
