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

  n_imp = *(p->n_imp);
  n_imp2 = n_imp * n_imp;
  n_imp3 = n_imp2 * n_imp;
  n_imp4 = n_imp3 * n_imp;
  macis::ASCISettings asci_settings = *(p->asci_settings);

  std::vector<double> T = *(p->T);
  std::vector<double> V = *(p->V);

  using generator_t = macis::DoubleLoopHamiltonianGenerator<nwfn_bits>;

  generator_t ham_gen(
      matrix_span_t(T.data(), n_imp, n_imp),
      rank4_span_t(V.data(), n_imp, n_imp, n_imp, n_imp));

  double orb_db_occs = 0.0;

  std::vector<double> ordm_u, ordm_d;
  std::vector<double> trdm_uu, trdm_dd, trdm_ud, trdm_du;

  if(asci_settings.nrots == 0) {
      ham_gen.form_rdms(dets.begin(), dets.end(), dets.begin(), dets.end(),
                      C.data(), matrix_span<double>(ordm_u, n_imp, n_imp),
                      matrix_span<double>(ordm_d, n_imp, n_imp),
                      rank4_span<double>(trdm_uu, n_imp, n_imp, n_imp, n_imp),
                      rank4_span<double>(trdm_ud, n_imp, n_imp, n_imp, n_imp),
                      rank4_span<double>(trdm_ud, n_imp, n_imp, n_imp, n_imp),
                      rank4_span<double>(trdm_dd, n_imp, n_imp, n_imp, n_imp));
  
      for (int a = 0; a < n_imp; a++) {
          orb_db_occs += trdm_ud[a + a * n_imp + a * n_imp2 + a * n_imp3];
      }
      orb_db_occs = orb_db_occs/n_imp;
    
      {

      orb_db_occs_bm = 0.0;

      dets = *(p->dets);
      C_local = *(p->C);

      struct wf_pair
      {
        std::string str;
        double coeff;
      };

      std::vector<wf_pair> pairs;
      pairs.reserve(dets.size());
      for (size_t idet = 0; idet < dets.size(); ++idet) {
           wf_pair p = {macis::to_canonical_string(dets[idet]), C_local[idet]};
          pairs.push_back(p);
      }  
      
      std::sort(pairs.begin(), pairs.end(),
              [](const wf_pair& a, const wf_pair& b) { return abs(a.coeff) > abs(b.coeff); });

      for (size_t i = 0; i < n_imp; ++i) {
          if (pairs[idet].str[i] == '2') {
            orb_db_occs_bm += pairs[idet].coeff * pairs[idet].coeff;
          }
      }
      orb_db_occs_bm = orb_db_occs_bm/n_imp;

  }

  std::cout << "Double Occupancies (from 2-RDM) = " << std::setprecision(10) << orb_db_occs << std::endl;
  std::cout << "Double Occupancies (from WF) = " << std::setprecision(10) << orb_db_occs_bm << std::endl;

  }

  return orb_db_occs;


} // close Comp_db_occs


auto comp_observables(void* params, bool db_occs_flag, bool sz_sz_flag, bool tz_tz_flag){

  struct impurity_params* p = static_cast<impurity_params*>(params);

  norb = *(p->norb);
  n_imp = *(p->n_imp);
  n_bands = *(p->nbands);
  n_sites = n_imp/n_bands;
  n_sites2 = n_sites*n_sites;
  n_imp2 = n_imp * n_imp;
  n_imp3 = n_imp2 * n_imp;
  n_imp4 = n_imp3 * n_imp;
  macis::ASCISettings asci_settings = *(p->asci_settings);

  std::vector<double> T = *(p->T);
  std::vector<double> V = *(p->V);

  using generator_t = macis::DoubleLoopHamiltonianGenerator<nwfn_bits>;

  generator_t ham_gen(
      matrix_span_t(T.data(), n_imp, n_imp),
      rank4_span_t(V.data(), n_imp, n_imp, n_imp, n_imp));


  std::vector<double> ordm_u, ordm_d;
  std::vector<double> trdm_uu, trdm_dd, trdm_ud, trdm_du;

  ham_gen.form_rdms(dets.begin(), dets.end(), dets.begin(), dets.end(),
                     C.data(), matrix_span<double>(ordm_u, n_imp, n_imp),
                     matrix_span<double>(ordm_d, n_imp, n_imp),
                     rank4_span<double>(trdm_uu, n_imp, n_imp, n_imp, n_imp),
                     rank4_span<double>(trdm_ud, n_imp, n_imp, n_imp, n_imp),
                     rank4_span<double>(trdm_ud, n_imp, n_imp, n_imp, n_imp),
                     rank4_span<double>(trdm_dd, n_imp, n_imp, n_imp, n_imp));
  
  if(db_occs_flag) {
  
      double orb_db_occs = 0.0;
  
      for (int a = 0; a < n_imp; a++) {
           orb_db_occs += trdm_ud[a + a * n_imp + a * n_imp2 + a * n_imp3];
       }
      orb_db_occs = orb_db_occs/n_imp;
       std::cout << "Double Occupancies (from 2-RDM) = " << std::setprecision(10) << orb_db_occs << std::endl;
   }

  if(sz_sz_flag) {
  
      std::vector<double> sz_sz(n_sites2,0.0);
      for (size_t site_i = 0, site_i < n_sites; site_i++) {
        for (size_t site_j = 0, site_j < n_sites; site_j++) {
          for(size_t band_i = 0; band_i < n_bands; band_i++) {
            for(size_t band_j = 0; band_j < n_bands; band_j++) {
              int a = site_i*n_sites + band_i;
              int b = site_j*n_sites + band_j;
              sz_sz[site_i + site_j*n_sites] += 0.25*(trdm_uu[a + a * n_imp + b * n_imp2 + b * n_imp3]
                                                 - trdm_ud[a + a * n_imp + b * n_imp2 + b * n_imp3]
                                                 - trdm_du[a + a * n_imp + b * n_imp2 + b * n_imp3]
                                                 + trdm_dd[a + a * n_imp + b * n_imp2 + b * n_imp3]);
            } // band_j
          } // band_i
        }
      }
   
      // //print_to_file
      // std::ofstream ofs;
      // ofs.open("sz_sz.dat");
      // ofs << std::setprecision(10);
      // for (size_t site_i = 0; site_i < n_sites; site_i
      //     for (size_t site_j = 0; site_j < n_sites; site_j++)
      //         ofs << site_i << " " << site_j << " " << sz[site_i + site_j*n_sites] << std::endl;
      // ofs.close();
  } // close sz_sz_flag

  if(tz_tz_flag) {
      double sign = 1.0;
      std::vector<double> tz_tz(n_sites2,0.0);
      for (size_t site_i = 0, site_i < n_sites; site_i++) {
        for (size_t site_j = 0, site_j < n_sites; site_j++) {
          for(size_t band_i = 0; band_i < n_bands; band_i++) {
            for(size_t band_j = 0; band_j < n_bands; band_j++) {
              int a = site_i*n_sites + band_i;
              int b = site_j*n_sites + band_j;
              if (band_i == band_j) sign = 1.0;
              else sign = -1.0;
              tz_tz[site_i + site_j*n_sites] += 0.25*( sign * trdm_uu[a + a * n_imp + b * n_imp2 + b * n_imp3]
                                                 +     sign * trdm_ud[a + a * n_imp + b * n_imp2 + b * n_imp3]
                                                 +     sign * trdm_du[a + a * n_imp + b * n_imp2 + b * n_imp3]
                                                 +     sign * trdm_dd[a + a * n_imp + b * n_imp2 + b * n_imp3]);
            } // band_j
          } // band_i
        }
      }
   
  }
  }


class CompObservables {
public:
  std::vector<double> dets;
  std::vector<double> C;

}  // namespace macis