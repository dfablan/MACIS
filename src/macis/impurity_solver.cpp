#define IMPURITY_SOLVER_CPP
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>

#include "macis/asci/determinant_symmetry.hpp"
#include "macis/impurity_solver.hpp"
#include "macis/parity_sectors.hpp"


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
 *  @brief The reference determinant the ASCI expansion is seeded from.
 *
 *  Plain canonical_hf_determinant fills the first nalpha/nbeta *raw indices*,
 *  which is the Hartree-Fock reference only when the orbitals are already
 *  ordered by energy. A bath fit is under no obligation to emit its poles in
 *  energy order, so that filling can leave a deep bath level empty while
 *  occupying a shallow one (observed: Irrep/Nb15/J_0.1/U_30, an empty level
 *  at eps = -5.561 next to an occupied one at +0.003). Fill by the one-body
 *  diagonal instead, as the legacy ASCI-CI driver did.
 *
 *  The seed is not just a starting point: every quantity H conserves on
 *  determinants (e.g. the per-band parity of a band-diagonal Kanamori bath)
 *  is inherited by the whole expansion from it. At nalpha != nbeta the raw
 *  fill puts the unpaired electron in whichever band the file happens to list
 *  next, which can be the wrong parity sector.
 *
 *  stable_sort on an already ascending diagonal is the identity, so runs
 *  whose orbitals were energy-ordered get the same seed as before, bit for
 *  bit. ASCI.HF_BY_ENERGY = FALSE restores the raw-index fill.
 *
 *  Under SYMMETRIZE_DETS the energy order is always used: it also makes the
 *  reference G-closed for free in the usual case. The generators have been
 *  validated as exact symmetries of T_active (and Td_active), so the diagonals
 *  are constant on each orbit, stable_sort keeps orbit members contiguous, and
 *  the occupied prefix splits an orbit only if the Fermi cut falls inside a
 *  tie group. The seed is closed under the group at the call sites
 *  regardless, so that remaining case is handled rather than refused.
 *
 *  Spin-dependent runs order the beta electrons by the Td_active diagonal.
 */
template <size_t N>
macis::wfn_t<N> asci_reference_determinant(const impurity_params<N>& p) {
  if(!p.asci_settings.hf_by_energy and !p.asci_settings.symmetrize_dets)
    return macis::canonical_hf_determinant<N>(p.nalpha, p.nbeta);

  const size_t n = p.n_active;
  std::vector<double> ens_alpha(n), ens_beta(n);
  for(size_t q = 0; q < n; ++q) {
    ens_alpha[q] = p.T_active[q + q * n];
    ens_beta[q] = p.spin_dep ? p.Td_active[q + q * n] : ens_alpha[q];
  }
  // Each call fills one spin only, so OR-ing them orders the spins separately
  return macis::canonical_hf_determinant<N>(p.nalpha, 0, ens_alpha) |
         macis::canonical_hf_determinant<N>(0, p.nbeta, ens_beta);
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
 *  An asymmetric HF reference is reported but not refused. The invariant the
 *  ASCI machinery actually needs is that the *selected* determinant space be
 *  G-closed, and symmetric_orbit_select provides that unconditionally: it
 *  materializes each surviving orbit whole from its representative, whatever
 *  the seed was. The seed itself is closed at the call sites so the guess and
 *  HF paths stay uniform and the first candidate ranking is unbiased.
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

  const auto hf_det = asci_reference_determinant<N>(p);

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

    // The HF reference need not be individually invariant: the seed is closed
    // under the group at the call sites below, which costs nothing when it
    // already was. Report an asymmetric reference, since it means the orbital
    // numbering does not put whole flavor multiplets in the occupied prefix --
    // worth knowing, but not a reason to refuse.
    if(macis::permute_orbitals(hf_det, g) != hf_det)
      std::cout << "* SYMMETRIZE_DETS: " << gname
                << " does not leave the HF reference determinant invariant "
                   "(the occupied prefix splits a flavor multiplet); the seed "
                   "will be closed under the group instead."
                << std::endl;
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

/**
 *  @brief Side outputs of one solve, kept apart so the parity-sector wrapper
 *  can compare sectors before anything is written to disk.
 */
template <size_t N>
struct SolveExtras {
  std::vector<double> ordm;  // active 1-RDM in the original orbital basis
  bool cold = false;         // started from a built seed, not a guess file
  macis::wfn_t<N> seed;
  double E_seed = std::numeric_limits<double>::quiet_NaN();  // total energy
};

/**
 *  @brief The cold seed: the energy-ordered reference, moved into the target
 *  band-parity sector by parity_seed when a sector is set.
 */
template <size_t N>
macis::wfn_t<N> sector_seed(const impurity_params<N>& p,
                            macis::HamiltonianGenerator<N>& ham_gen,
                            SolveExtras<N>& x) {
  auto d = asci_reference_determinant<N>(p);
  if(const auto& t = p.asci_settings.parity_target) {
    d = macis::parity_seed(d, t->key, *t->labels, ham_gen, p.n_imp);
    std::cout << "* PARITY_SOLVE: seed for sector "
              << macis::parity_key_string(t->key, t->labels->ngroups) << " = "
              << macis::to_canonical_string(d) << std::endl;
  }
  x.cold = true;
  x.seed = d;
  x.E_seed = ham_gen.matrix_element(d, d) + p.E_core + p.E_inactive;
  return d;
}

/**
 *  @brief Throw unless every determinant lies in the target parity sector.
 *  A seed outside it would make the filtered search collapse onto the seed
 *  alone, so this must never pass silently.
 */
template <size_t N>
void require_in_sector(const impurity_params<N>& p,
                       const std::vector<macis::wfn_t<N>>& dets,
                       const std::string& what) {
  const auto& t = p.asci_settings.parity_target;
  if(!t) return;
  const macis::ParityMasks<N> pm(*t->labels);
  for(const auto& d : dets)
    if(pm.key(d) != t->key)
      throw std::logic_error(
          "PARITY_SOLVE: " + what + " determinant " +
          macis::to_canonical_string(d) + " is in sector " +
          macis::parity_key_string(pm.key(d), t->labels->ngroups) +
          ", not in the target " +
          macis::parity_key_string(t->key, t->labels->ngroups));
}

/**
 *  @brief The restart determinant of a macro iteration after the per-band
 *  natural-orbital rotation, in the target parity sector.
 *
 *  hf_determinant_byocc fills the same orbitals for both spins, so at
 *  NALPHA = NBETA it is closed-shell and always all-even: in any other
 *  sector the filtered search would collapse onto it. It is therefore moved
 *  into the sector by parity_seed, judged by the diagonal of the *rotated*
 *  Hamiltonian; in its own sector it is returned unchanged. Decoupled
 *  orbitals never rotate and H conserves their occupations, so they keep
 *  those of the previous seed `prev`, and only the grouped orbitals are
 *  filled by occupation (identical to byocc when there are none).
 */
template <size_t N>
macis::wfn_t<N> parity_restart(const impurity_params<N>& p,
                               const std::vector<double>& orb_occs,
                               const macis::wfn_t<N>& prev,
                               macis::HamiltonianGenerator<N>& ham_gen) {
  const auto& t = *p.asci_settings.parity_target;
  const auto& L = *t.labels;
  std::vector<size_t> idx;
  for(size_t q = 0; q < p.n_active; ++q)
    if(L.group_of[q] >= 0) idx.push_back(q);
  std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
    return orb_occs[a] > orb_occs[b];
  });
  macis::wfn_t<N> d(0);
  size_t na = p.nalpha, nb = p.nbeta;
  for(size_t q = 0; q < p.n_active; ++q)
    if(L.group_of[q] < 0) {
      if(prev[q]) d.set(q), --na;
      if(prev[q + N / 2]) d.set(q + N / 2), --nb;
    }
  for(size_t i = 0; i < na; ++i) d.set(idx[i]);
  for(size_t i = 0; i < nb; ++i) d.set(idx[i] + N / 2);
  return macis::parity_seed(d, t.key, L, ham_gen, p.n_imp);
}

#ifdef MACIS_ENABLE_MPI
std::vector<double> gather_ci_vector(std::vector<double> C_local,
                                     size_t ndets) {
  auto world_size = macis::comm_size(MPI_COMM_WORLD);
  auto world_rank = macis::comm_rank(MPI_COMM_WORLD);
  if(world_size == 1) return C_local;
  std::vector<double> C(ndets);
  const size_t local_count = ndets / world_size;
  MPI_Allgather(C_local.data(), local_count, MPI_DOUBLE, C.data(), local_count,
                MPI_DOUBLE, MPI_COMM_WORLD);
  if(ndets % world_size) {
    const size_t nrem = ndets % world_size;
    auto* C_rem = C.data() + world_size * local_count;
    if(world_rank == world_size - 1)
      std::copy_n(C_local.data() + local_count, nrem, C_rem);
    MPI_Bcast(C_rem, nrem, MPI_DOUBLE, world_size - 1, MPI_COMM_WORLD);
  }
  return C;
}
#else
std::vector<double> gather_ci_vector(std::vector<double> C_local, size_t) {
  return C_local;
}
#endif

bool is_root_rank() {
  int rank = 0;
  MACIS_MPI_CODE(rank = macis::comm_rank(MPI_COMM_WORLD);)
  return rank == 0;
}

template <size_t N>
double solve_ed_one(impurity_params<N>& p, SolveExtras<N>& x){

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

    // Parity-sector ED: the full Hilbert space restricted to the target
    // sector. CASRDMFunctor diagonalizes the whole space, whose Davidson
    // start (the lowest diagonal element) fixes the sector it converges in.
    if(const auto& t = p.asci_settings.parity_target) {
      generator_t ham_gen(
          macis::matrix_span<double>(T_active.data(), n_active, n_active),
          macis::rank4_span<double>(V_active.data(), n_active, n_active,
                                    n_active, n_active));
      if(p.spin_dep)
        ham_gen.ReadTdo(
            macis::matrix_span<double>(Td_active.data(), n_active, n_active));
      const macis::ParityMasks<N> pm(*t->labels);
      // Decoupled orbitals keep the occupation of the energy-ordered seed
      const auto base = asci_reference_determinant<N>(p);
      macis::wfn_t<N> dark_mask;
      for(size_t q = 0; q < n_active; ++q)
        if(t->labels->group_of[q] < 0) {
          dark_mask.set(q);
          dark_mask.set(q + N / 2);
        }
      dets.clear();
      for(const auto& d : macis::generate_hilbert_space<N>(n_active, nalpha,
                                                           nbeta))
        if(pm.key(d) == t->key and (d & dark_mask) == (base & dark_mask))
          dets.push_back(d);
      if(dets.empty())
        throw std::runtime_error(
            "PARITY_SOLVE: parity sector " +
            macis::parity_key_string(t->key, t->labels->ngroups) +
            " is empty at this (NALPHA, NBETA)");
      std::cout << "* PARITY_SOLVE: ED in sector "
                << macis::parity_key_string(t->key, t->labels->ngroups)
                << ", " << dets.size() << " determinants" << std::endl;
      std::vector<double> C_dist;
      E0 = macis::selected_ci_diag(
          dets.begin(), dets.end(), ham_gen, mcscf_settings.ci_matel_tol,
          mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol,
          C_dist MACIS_MPI_CODE(, MPI_COMM_WORLD), true);
      C_local = gather_ci_vector(std::move(C_dist), dets.size());
      ham_gen.form_rdms(
          dets.begin(), dets.end(), dets.begin(), dets.end(), C_local.data(),
          macis::matrix_span<double>(active_ordm.data(), n_active, n_active),
          macis::rank4_span<double>(active_trdm.data(), n_active, n_active,
                                    n_active, n_active));
      E0 += E_inactive + E_core;
      for(size_t i = 0; i < n_active; i++)
        occs[i] = active_ordm[i + i * n_active] / 2.;
      x.ordm = active_ordm;
      return E0;
    }

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

    x.ordm = active_ordm;
    return E0;
}

template <size_t N>
double solve_asci_one(impurity_params<N>& p, SolveExtras<N>& x){

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
      std::cout << "Generating HF Guess for ASCI ("
                << (asci_settings.hf_by_energy or asci_settings.symmetrize_dets
                        ? "filled by one-body energy"
                        : "filled by raw orbital index, ASCI.HF_BY_ENERGY = FALSE")
                << ")" << std::endl;
      dets = {sector_seed<N>(p, ham_gen, x)};
      E0 = ham_gen.matrix_element(dets[0], dets[0]);
      C_local = {1.0};
    }
    // Close whatever seeded the expansion -- HF fallback or guess from file --
    // under the group. E0 above is taken from the HF determinant before this
    // call: close_guess_under_group rebuilds dets in bitset order, so dets[0]
    // afterwards is not necessarily the reference. The added partners carry
    // C = 0, so they leave that E0 correct.
    if(asci_settings.symmetrize_dets and asci_settings.sym_group)
      close_guess_under_group(dets, C_local, *asci_settings.sym_group);
    require_in_sector(p, dets, "seed");
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

    x.ordm = active_ordm;
    return E0;
}

template <size_t N>
double solve_asci_rot_one(impurity_params<N>& p, SolveExtras<N>& x){

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
    macis::wfn_t<N> hf_det = sector_seed<N>(p, ham_gen, x);
    std::vector<double> orb_occs(n_active,0.0);

    // A guess only seeds the first macro iteration. load_asci_guess requires
    // NROTS == 0, so with a guess that first iteration is also the only one.
    const bool have_guess = load_asci_guess(p, dets, C_local, E0);
    if(have_guess) x.cold = false;
    if(!have_guess)
    {
      // HF Guess
      std::cout << "Generating HF Guess for ASCI ("
              << (asci_settings.hf_by_energy or asci_settings.symmetrize_dets
                      ? "filled by one-body energy"
                      : "filled by raw orbital index, ASCI.HF_BY_ENERGY = FALSE")
              << ")" << std::endl;
      dets = {hf_det};
      E0 = ham_gen.matrix_element(dets[0], dets[0]);
      C_local = {1.0};
    }
    // Close whatever seeded the expansion -- HF fallback or guess from file --
    // under the group. E0 above is taken from the HF determinant before this
    // call: close_guess_under_group rebuilds dets in bitset order, so dets[0]
    // afterwards is not necessarily the reference. The added partners carry
    // C = 0, so they leave that E0 correct.
    if(asci_settings.symmetrize_dets and asci_settings.sym_group)
      close_guess_under_group(dets, C_local, *asci_settings.sym_group);
    require_in_sector(p, dets, "seed");
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
            //In a parity sector the natural orbitals are taken per band group
            //as well (rdms.hpp), so every orbital index keeps its band: the
            //labels, masks and search filter stay valid in the rotated basis.
            //Without that, gesvd may mix degenerate occupations of different
            //bands -- the normal case with degenerate bands.
            const auto& parity_target = asci_settings.parity_target;
            if(parity_target)
              std::cout << "* PARITY_SOLVE: per-band natural orbitals; largest "
                           "cross-band 1-RDM element (LABEL_LEAK) = "
                        << macis::max_off_group(active_ordm.data(),
                                                *parity_target->labels)
                        << std::endl;
            ham_gen.rotate_hamiltonian_ordm_imp_bath( active_ordm.data(), n_imp, tmp_rot.data() , p.spin_dep, orb_occs.data(),
                parity_target ? &parity_target->labels->group_of : nullptr );
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
            if(parity_target) {
              hf_det = parity_restart<N>(p, orb_occs, hf_det, ham_gen);
              std::cout << "* PARITY_SOLVE: restart for sector "
                        << macis::parity_key_string(parity_target->key,
                                                    parity_target->labels->ngroups)
                        << " = " << macis::to_canonical_string(hf_det) << std::endl;
              require_in_sector(p, {hf_det}, "restart");
            } else
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

    // active_ordm.dat and rot_matrix.dat are written by SolveImpurityASCI_rot
    // (write_rot_outputs), once, for the returned state: the parity-sector
    // wrapper runs this function once per sector.
    x.ordm = active_ordm;

    auto end_ASCI_clock = clock_type::now();
    duration_type total_ASCI_time = end_ASCI_clock - start_ASCI_clock;
    int minutes = static_cast<int>(total_ASCI_time.count() / 60000.0);
    double seconds = (total_ASCI_time.count() / 1000.0) - (minutes * 60.0);
    std::cout << "\nTotal time to complete ASCI GS calculation: \n"
              << minutes << " minutes " << seconds << " seconds" << std::endl;
          
    return E0;
}

template <size_t N>
using one_solver_t = double (*)(impurity_params<N>&, SolveExtras<N>&);

/// Outcome of one parity sector
template <size_t N>
struct SectorRun {
  uint32_t key = 0;
  bool ok = false;
  std::string error;
  std::string start;
  double E = std::numeric_limits<double>::quiet_NaN();
  SolveExtras<N> x;
  impurity_params<N> params;
};

struct SectorGuess {
  std::string fname;
  double E0;  // total energy of the guess
};

std::string parity_key_letters(uint32_t key, size_t ngroups) {
  std::string s;
  for(size_t g = 0; g < ngroups; ++g) s += ((key >> g) & 1u) ? 'o' : 'e';
  return s;
}

/**
 *  @brief Split a guess wavefunction (ASCI.WFN_FILE, or a charge-sector warm
 *  seed) by parity sector.
 *
 *  A guess that lies in one sector -- any guess written by a parity solve --
 *  is handed to that sector unchanged, with its supplied E0. One that spans
 *  several (e.g. c+_A psi and c+_B psi, the N+1 seeds of the charge-sector
 *  search) is cut into one slice per sector; each slice is diagonalized and
 *  written to <fname>.par_<key>, as charge_sectors' solve_from_seed does.
 *  Mixing sectors in one solve would trap it in the sector of the lowest
 *  diagonal element.
 */
template <size_t N>
std::map<uint32_t, SectorGuess> split_guess_by_sector(
    const impurity_params<N>& p, const std::vector<uint32_t>& keys) {
  std::map<uint32_t, SectorGuess> out;
  if(p.asci_wfn_fname.empty()) return out;
  const auto& L = *p.parity_labels;
  const macis::ParityMasks<N> pm(L);

  std::vector<macis::wfn_t<N>> dets;
  std::vector<double> C;
  macis::read_wavefunction<N>(p.asci_wfn_fname, dets, C, true);
  std::map<uint32_t, std::vector<size_t>> by_key;
  for(size_t i = 0; i < dets.size(); ++i) by_key[pm.key(dets[i])].push_back(i);

  if(by_key.size() == 1) {
    out[by_key.begin()->first] = {p.asci_wfn_fname, p.asci_E0};
    std::cout << "* PARITY_SOLVE: guess " << p.asci_wfn_fname
              << " lies in sector "
              << macis::parity_key_string(by_key.begin()->first, L.ngroups)
              << std::endl;
    return out;
  }

  using generator_t = macis::SDBuildHamiltonianGenerator<N>;
  auto T_active = p.T_active, Td_active = p.Td_active, V_active = p.V_active;
  generator_t ham_gen(
      macis::matrix_span<double>(T_active.data(), p.n_active, p.n_active),
      macis::rank4_span<double>(V_active.data(), p.n_active, p.n_active,
                                p.n_active, p.n_active));
  if(p.spin_dep)
    ham_gen.ReadTdo(
        macis::matrix_span<double>(Td_active.data(), p.n_active, p.n_active));
  ham_gen.SetJustSingles(p.just_singles);
  ham_gen.SetNimp(p.n_imp);

  for(const auto& [key, idx] : by_key) {
    if(std::find(keys.begin(), keys.end(), key) == keys.end()) {
      std::cout << "* PARITY_SOLVE: guess determinants in sector "
                << macis::parity_key_string(key, L.ngroups)
                << " are not used (sector not solved)" << std::endl;
      continue;
    }
    std::vector<macis::wfn_t<N>> sd;
    for(auto i : idx) sd.push_back(dets[i]);
    double E_act;
    std::vector<double> sC;
    if(sd.size() == 1) {
      E_act = ham_gen.matrix_element(sd[0], sd[0]);
      sC = {1.0};
    } else {
      std::vector<double> C_dist;
      E_act = macis::selected_ci_diag(
          sd.begin(), sd.end(), ham_gen, p.mcscf_settings.ci_matel_tol,
          p.mcscf_settings.ci_max_subspace, p.mcscf_settings.ci_res_tol,
          C_dist MACIS_MPI_CODE(, MPI_COMM_WORLD), true);
      sC = gather_ci_vector(std::move(C_dist), sd.size());
    }
    const std::string fname =
        p.asci_wfn_fname + ".par_" + parity_key_letters(key, L.ngroups);
    if(is_root_rank()) macis::write_wavefunction(fname, p.n_active, sd, sC);
    MACIS_MPI_CODE(MPI_Barrier(MPI_COMM_WORLD);)
    out[key] = {fname, E_act + p.E_core + p.E_inactive};
    std::cout << "* PARITY_SOLVE: guess slice for sector "
              << macis::parity_key_string(key, L.ngroups) << ": "
              << sd.size() << " determinants, E = " << std::setprecision(10)
              << out[key].E0 << " -> " << fname << std::endl;
  }
  return out;
}

/**
 *  @brief Print and write (parity_sectors.dat) the per-sector table.
 */
template <size_t N>
void report_parity_sectors(const std::vector<SectorRun<N>>& runs, int winner,
                           const macis::ParityLabels& L, double etol) {
  std::ostringstream os;
  os << std::setprecision(10);
  std::vector<std::string> failed;
  for(size_t i = 0; i < runs.size(); ++i) {
    const auto& r = runs[i];
    os << "PARITY_SECTOR key=" << macis::parity_key_string(r.key, L.ngroups);
    if(!r.ok) {
      os << " FAILED: " << r.error << "\n";
      failed.push_back(macis::parity_key_string(r.key, L.ngroups));
      continue;
    }
    os << " E=" << std::fixed << r.E << " dE=" << std::scientific
       << std::setprecision(3) << r.E - runs[winner].E << std::fixed
       << std::setprecision(10) << " ndets=" << r.params.dets.size()
       << " start=" << r.start;
    if(r.x.cold)
      os << " seed=" << macis::to_canonical_string(r.x.seed)
         << " E_seed=" << r.x.E_seed;
    os << " N_band=[";
    for(size_t g = 0; g < L.ngroups; ++g) {
      double n = 0.;
      for(auto q : L.group_orbs[g]) n += 2. * r.params.occs[q];
      os << (g ? "," : "") << std::setprecision(6) << n;
    }
    os << std::setprecision(10) << "]"
       << (int(i) == winner ? " WINNER" : "") << "\n";
    os.unsetf(std::ios::floatfield);
  }
  for(size_t i = 0; i < runs.size(); ++i)
    if(runs[i].ok and int(i) != winner and
       std::abs(runs[i].E - runs[winner].E) < etol)
      os << "PARITY_TIE sectors "
         << macis::parity_key_string(runs[winner].key, L.ngroups) << " and "
         << macis::parity_key_string(runs[i].key, L.ngroups)
         << " agree within PARITY_ETOL = " << etol
         << ": the ground state may be degenerate across them (e.g. band "
            "swap at odd N), and the returned state then breaks that "
            "symmetry. evaluate_GF averages its Green's function over the "
            "partners the SYMMETRIZE_DETS group relates (GF_BAND_AVERAGE) and "
            "warns otherwise\n";
  if(failed.empty())
    os << "PARITY_COVERAGE COMPLETE\n";
  else {
    os << "PARITY_COVERAGE INCOMPLETE: failed sectors";
    for(const auto& f : failed) os << " " << f;
    os << ". The winner is the lowest of the sectors that converged only.\n";
  }
  std::cout << os.str() << std::flush;
  if(is_root_rank()) {
    std::ofstream f("parity_sectors.dat");
    f << "# Band-parity sectors of the last impurity solve (ASCI.PARITY_SOLVE)"
      << "\n# groups:";
    for(size_t g = 0; g < L.ngroups; ++g) {
      f << " [";
      for(size_t k = 0; k < L.group_orbs[g].size(); ++k)
        f << (k ? "," : "") << L.group_orbs[g][k];
      f << "]";
    }
    f << "\n" << os.str();
  }
}

/**
 *  @brief Solve every band-parity sector separately and keep the lowest
 *  (parity-sector-solve-simple.md, §5).
 *
 *  Each sector runs on its own copy of the parameters, so the in-place
 *  integral handling of one sector cannot leak into the next, with its
 *  ASCI search filtered to the sector and a seed built inside it. A sector
 *  that throws is reported as FAILED and the others still run; the winner is
 *  the lowest converged sector. The winner's state is copied back into p.
 */
template <size_t N>
double solve_parity_sectors(impurity_params<N>& p, one_solver_t<N> solve,
                            SolveExtras<N>& x_out, const char* solver_name) {
  auto& stg = p.asci_settings;
  // NROTS > 0 is fine: solve_asci_rot_one rotates per band group in a sector.
  // The full-space rotation of GROW_WITH_ROT is not blocked by group.
  if(stg.grow_with_rot)
    throw std::runtime_error(
        "ASCI.PARITY_SOLVE requires ASCI.GROW_WITH_ROT = FALSE: that rotation "
        "is taken over the whole active space, mixes the bands and destroys "
        "the parity labels.");
  if(p.n_inactive != 0 or p.n_active != p.norb)
    throw std::runtime_error(
        "ASCI.PARITY_SOLVE requires NINACTIVE = 0 and NACTIVE = NORB");
  const auto labels = p.parity_labels;
  const auto& L = *labels;
  if(L.group_of.size() != p.n_active)
    throw std::logic_error("PARITY_SOLVE: labels built for " +
                           std::to_string(L.group_of.size()) +
                           " orbitals, the solve has " +
                           std::to_string(p.n_active));

  // Expand the permutation group once, so each sector's stabilizer is taken
  // from the full group rather than from the generators
  prepare_det_symmetry(p);

  // Electrons in the band groups; decoupled orbitals keep their seed filling
  const auto base = asci_reference_determinant<N>(p);
  size_t nel_grouped = 0;
  for(size_t q = 0; q < p.n_active; ++q)
    if(L.group_of[q] >= 0) nel_grouped += base[q] + base[q + N / 2];

  std::vector<uint32_t> keys;
  if(p.parity_only.empty())
    keys = macis::enumerate_parity_keys(L, nel_grouped);
  else {
    if(p.parity_only.size() != L.ngroups)
      throw std::runtime_error("ASCI.PARITY_ONLY needs one 0/1 entry per band (" +
                               std::to_string(L.ngroups) + ")");
    uint32_t k = 0;
    size_t pop = 0;
    for(size_t g = 0; g < L.ngroups; ++g) {
      if(p.parity_only[g] != 0 and p.parity_only[g] != 1)
        throw std::runtime_error("ASCI.PARITY_ONLY entries must be 0 or 1");
      k |= uint32_t(p.parity_only[g]) << g;
      pop += p.parity_only[g];
    }
    if(pop % 2 != nel_grouped % 2)
      throw std::runtime_error(
          "ASCI.PARITY_ONLY = " + macis::parity_key_string(k, L.ngroups) +
          " is impossible: the band parities must add up to the number of "
          "band electrons, " + std::to_string(nel_grouped) + ", mod 2");
    keys = {k};
  }

  std::cout << "\n* PARITY_SOLVE: " << keys.size() << " band-parity sector(s) for (NALPHA, NBETA) = ("
            << p.nalpha << ", " << p.nbeta << "), solver " << solver_name
            << std::endl;

  const auto guesses = split_guess_by_sector(p, keys);
  const std::string saved_fname = p.asci_wfn_fname;
  const double saved_E0 = p.asci_E0;
  const bool saved_compute_E0 = p.compute_asci_E0;
  const auto saved_group = stg.sym_group;

  // The solvers rebuild dets/C from scratch; don't carry the previous state
  // into every copy
  p.dets.clear();
  p.C.clear();

  std::vector<SectorRun<N>> runs;
  for(size_t ik = 0; ik < keys.size(); ++ik) {
    const auto key = keys[ik];
    std::cout << "\n* PARITY_SOLVE: sector "
              << macis::parity_key_string(key, L.ngroups) << " (" << ik + 1
              << "/" << keys.size() << ")" << std::endl;
    SectorRun<N> r;
    r.key = key;
    r.params = p;
    auto& ps = r.params;
    ps.asci_settings.parity_target = std::make_shared<const macis::ParityTarget>(
        macis::ParityTarget{labels, key});
    if(stg.symmetrize_dets and stg.sym_group)
      ps.asci_settings.sym_group =
          std::make_shared<const std::vector<std::vector<uint32_t>>>(
              macis::parity_stabilizer(*stg.sym_group, L, key));
    if(auto g = guesses.find(key); g != guesses.end()) {
      ps.asci_wfn_fname = g->second.fname;
      ps.asci_E0 = g->second.E0;
      ps.compute_asci_E0 = false;
      r.start = g->second.fname == saved_fname ? "guess" : "guess_slice";
    } else {
      ps.asci_wfn_fname.clear();
      ps.compute_asci_E0 = true;
      ps.asci_E0 = 0.0;
      r.start = "cold";
    }
    try {
      r.E = solve(ps, r.x);
      r.ok = std::isfinite(r.E);
      if(!r.ok) r.error = "non-finite energy";
    } catch(const std::exception& e) {
      r.error = e.what();
      std::cout << "WARNING: PARITY_SOLVE: sector "
                << macis::parity_key_string(key, L.ngroups)
                << " failed: " << r.error << std::endl;
    }
    runs.push_back(std::move(r));
  }

  int winner = -1;
  for(size_t i = 0; i < runs.size(); ++i)
    if(runs[i].ok and (winner < 0 or runs[i].E < runs[winner].E))
      winner = int(i);
  if(winner < 0) {
    std::string msg = "PARITY_SOLVE: every parity sector failed:";
    for(const auto& r : runs)
      msg += " " + macis::parity_key_string(r.key, L.ngroups) + ": " + r.error +
             ";";
    throw std::runtime_error(msg);
  }
  report_parity_sectors(runs, winner, L, p.parity_etol);

  // The winner becomes the result; restore what only the sector solves saw
  auto& w = runs[winner];
  p = std::move(w.params);
  p.asci_settings.parity_target.reset();
  p.asci_settings.sym_group = saved_group;
  p.asci_wfn_fname = saved_fname;
  p.asci_E0 = saved_E0;
  p.compute_asci_E0 = saved_compute_E0;
  // For evaluate_GF's band average (band_orbit_average_gf)
  p.parity_winner_key = w.key;
  p.parity_tied_keys.clear();
  for(size_t i = 0; i < runs.size(); ++i)
    if(runs[i].ok and int(i) != winner and
       std::abs(runs[i].E - w.E) < p.parity_etol)
      p.parity_tied_keys.push_back(runs[i].key);
  x_out = std::move(w.x);
  return w.E;
}

template <size_t N>
double dispatch_solve(impurity_params<N>& p, one_solver_t<N> solve,
                      SolveExtras<N>& x, const char* solver_name) {
  if(p.parity_labels and !p.asci_settings.parity_target)
    return solve_parity_sectors(p, solve, x, solver_name);
  return solve(p, x);
}

template <size_t N>
void write_rot_outputs(const impurity_params<N>& p,
                       const std::vector<double>& active_ordm) {
  const size_t n_active = p.n_active;
  {
    std::ofstream ofile_ordm("active_ordm.dat");
    ofile_ordm.precision(std::numeric_limits<double>::max_digits10);
    for(size_t i = 0; i < n_active; i++) {
      for(size_t j = 0; j < n_active; j++)
        ofile_ordm << std::scientific << active_ordm[i + j * n_active] << " ";
      ofile_ordm << std::endl;
    }
  }
  {
    std::ofstream ofile_rot("rot_matrix.dat");
    ofile_rot.precision(std::numeric_limits<double>::max_digits10);
    for(size_t i = 0; i < n_active; i++) {
      for(size_t j = 0; j < n_active; j++)
        ofile_rot << std::scientific << p.orb_rot[i + j * n_active] << " ";
      ofile_rot << std::endl;
    }
  }
}

}  // namespace

template <size_t N>
double SolveImpurityED(impurity_params<N>& p) {
  SolveExtras<N> x;
  return dispatch_solve<N>(p, &solve_ed_one<N>, x, "ED");
}

template <size_t N>
double SolveImpurityASCI(impurity_params<N>& p) {
  SolveExtras<N> x;
  return dispatch_solve<N>(p, &solve_asci_one<N>, x, "ASCI");
}

template <size_t N>
double SolveImpurityASCI_rot(impurity_params<N>& p) {
  SolveExtras<N> x;
  const double E = dispatch_solve<N>(p, &solve_asci_rot_one<N>, x, "ASCI_rot");
  write_rot_outputs(p, x.ordm);
  return E;
}

template <size_t N>
void setup_parity_sectors(impurity_params<N>& p, double tol) {
  const auto& stg = p.asci_settings;
  if(stg.grow_with_rot)
    throw std::runtime_error(
        "ASCI.PARITY_SOLVE requires ASCI.GROW_WITH_ROT = FALSE (that rotation "
        "is taken over the whole active space and mixes the bands)");
  if(p.n_inactive != 0 or p.n_active != p.norb)
    throw std::runtime_error(
        "ASCI.PARITY_SOLVE requires NINACTIVE = 0 and NACTIVE = NORB");
  p.parity_labels = std::make_shared<const ParityLabels>(build_parity_labels(
      p.norb, p.n_imp, p.nbands, p.T, p.spin_dep ? &p.Td : nullptr, p.V, tol,
      std::cout));
  std::cout << "* PARITY_SOLVE on: each impurity solve runs "
            << (size_t(1) << (p.parity_labels->ngroups - 1))
            << " band-parity sectors (PARITY_TOL = " << tol << ")"
            << std::endl;
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
template void setup_parity_sectors<64>(impurity_params<64>& p, double tol);

} // namespace macis
