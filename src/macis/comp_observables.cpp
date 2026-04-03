#define COMP_OBSERVABLES_CPP
#include "macis/comp_observables.hpp"

namespace macis {

void Transform_2RDMs(const int norbs, const std::vector<double>& ordm_u,
                     const std::vector<double>& ordm_d,
                     std::vector<double>& trdm_uu, std::vector<double>& trdm_ud,
                     std::vector<double>& trdm_du,
                     std::vector<double>& trdm_dd) {
  // Transforms 2-RDMs from
  // <c^+_a c^+_d c_e c_b> format into
  // <c^+_a c_b c^+_d c_e> format.

  for(int o1 = 0; o1 < norbs; o1++)
    for(int v1 = 0; v1 < norbs; v1++)
      for(int e = 0; e < norbs; e++) {
        trdm_uu[v1 + e * norbs + e * norbs * norbs +
                o1 * norbs * norbs * norbs] += ordm_u[v1 + norbs * o1];
        trdm_dd[v1 + e * norbs + e * norbs * norbs +
                o1 * norbs * norbs * norbs] += ordm_d[v1 + norbs * o1];
      }
}


template<size_t N>
double Comp_db_occs(impurity_params<N>& p) {

  size_t& n_active = (p.n_active);
  size_t n_active2 = n_active * n_active;
  size_t n_active3 = n_active2 * n_active;
  size_t n_active4 = n_active3 * n_active;
  size_t& n_imp = (p.n_imp);
  size_t n_imp2 = n_imp * n_imp;
  size_t n_imp3 = n_imp2 * n_imp;
  size_t n_imp4 = n_imp3 * n_imp;
  size_t& norb = (p.norb);
  size_t norb2 = norb * norb;
  size_t& n_inactive = (p.n_inactive);
  macis::ASCISettings& asci_settings = (p.asci_settings);

  std::vector<double>& T = (p.T);
  std::vector<double>& V = (p.V);

  std::vector<macis::wfn_t<N>>& dets = (p.dets);
  std::vector<double>& C_local = (p.C);

  using generator_t = macis::SDBuildHamiltonianGenerator<N>;

  // Copy integrals into active subsets
  std::vector<double> T_active(n_active * n_active);
  std::vector<double> V_active(n_active * n_active * n_active * n_active);
  // Compute active-space Hamiltonian and inactive Fock matrix
  std::vector<double> F_inactive(norb2);
  macis::active_hamiltonian(NumOrbital(norb), NumActive(n_active),
                            NumInactive(n_inactive), T.data(), norb, V.data(),
                            norb, F_inactive.data(), norb, T_active.data(),
                            n_active, V_active.data(), n_active);

  generator_t ham_gen(
      macis::matrix_span<double>(T_active.data(), n_active, n_active),
      macis::rank4_span<double>(V_active.data(), n_active, n_active, n_active,
                                n_active));
  ham_gen.SetJustSingles(p.just_singles);
  ham_gen.SetNimp(n_imp);

  double orb_db_occs = 0.0;
  double orb_db_occs_bm = 0.0;

  std::vector<double> ordm_u(n_active2, 0.0), ordm_d(n_active2, 0.0);
  std::vector<double> trdm_uu(n_active4, 0.0), trdm_dd(n_active4, 0.0);
  std::vector<double> trdm_ud(n_active4, 0.0), trdm_du(n_active4, 0.0);

  if(asci_settings.nrots == 0) {
    ham_gen.form_rdms(
        dets.begin(), dets.end(), dets.begin(), dets.end(), C_local.data(),
        macis::matrix_span<double>(ordm_u.data(), n_active, n_active),
        macis::matrix_span<double>(ordm_d.data(), n_active, n_active),
        macis::rank4_span<double>(trdm_uu.data(), n_active, n_active, n_active,
                                  n_active),
        macis::rank4_span<double>(trdm_ud.data(), n_active, n_active, n_active,
                                  n_active),
        macis::rank4_span<double>(trdm_du.data(), n_active, n_active, n_active,
                                  n_active),
        macis::rank4_span<double>(trdm_dd.data(), n_active, n_active, n_active,
                                  n_active));

    for(int a = 0; a < n_imp; a++) {
      orb_db_occs += trdm_ud[a + a * n_active + a * n_active2 + a * n_active3];
    }
    orb_db_occs = orb_db_occs / n_imp;

    std::cout << "Double Occupancies (from 2-RDM) = " << std::setprecision(10)
              << orb_db_occs << std::endl;

    {
      struct wf_pair {
        std::string str;
        double coeff;
      };

      std::vector<wf_pair> pairs;
      pairs.reserve(dets.size());
      for(int idet = 0; idet < dets.size(); idet++) {
        wf_pair p = {macis::to_canonical_string(dets[idet]), C_local[idet]};
        pairs.push_back(p);
      }

      std::sort(pairs.begin(), pairs.end(),
                [](const wf_pair& a, const wf_pair& b) {
                  return abs(a.coeff) > abs(b.coeff);
                });

      for(int idet = 0; idet < pairs.size(); ++idet) {
        for(size_t i = 0; i < n_imp; ++i) {
          if(pairs[idet].str[i] == '2') {
            orb_db_occs_bm += pairs[idet].coeff * pairs[idet].coeff;
          }
        }
      }
      orb_db_occs_bm = orb_db_occs_bm / n_imp;
      std::cout << "Double Occupancies (from WF) = " << std::setprecision(10)
                << orb_db_occs_bm << std::endl;
    }
  }

  return orb_db_occs_bm;

}  // close Comp_db_occs

template<size_t N>
CompObservables<N>::CompObservables(impurity_params<N>& p) 
    : norb_(p.norb),
      n_imp_(p.n_imp),
      n_bands_(p.nbands),
      n_active_(p.n_active),
      n_inactive_(p.n_inactive),
      dets_(p.dets),
      C_(p.C),
      orb_rot_(p.orb_rot),
      T_active(p.T_active),
      V_active(p.V_active),
      F_inactive(p.F_inactive)
  {
    // struct impurity_params<N>* p = static_cast<impurity_params<N>*>(params);

    // Initialize dimensions
    n_sites_ = n_imp_ / n_bands_;
    n_sites2_ = n_sites_ * n_sites_;
    n_imp2_ = n_imp_ * n_imp_;
    n_imp3_ = n_imp2_ * n_imp_;
    n_imp4_ = n_imp3_ * n_imp_;
    n_active2_ = n_active_ * n_active_;
    n_active3_ = n_active2_ * n_active_;
    n_active4_ = n_active3_ * n_active_;
    norb2_ = norb_ * norb_;

    // Initialize RDMs
    ordm_u_.resize(n_active2_);
    ordm_d_.resize(n_active2_);
    trdm_uu_.resize(n_active4_);
    trdm_dd_.resize(n_active4_);
    trdm_ud_.resize(n_active4_);
    trdm_du_.resize(n_active4_);

    // Build generator and compute RDMs
    using generator_t = macis::SDBuildHamiltonianGenerator<N>;
    generator_t ham_gen(
        macis::matrix_span<double>(T_active.data(), n_active_, n_active_),
        macis::rank4_span<double>(V_active.data(), n_active_, n_active_,
                                  n_active_, n_active_));

    ham_gen.form_rdms(
        dets_.begin(), dets_.end(), dets_.begin(), dets_.end(), C_.data(),
        macis::matrix_span<double>(ordm_u_.data(), n_active_, n_active_),
        macis::matrix_span<double>(ordm_d_.data(), n_active_, n_active_),
        macis::rank4_span<double>(trdm_uu_.data(), n_active_, n_active_,
                                  n_active_, n_active_),
        macis::rank4_span<double>(trdm_ud_.data(), n_active_, n_active_,
                                  n_active_, n_active_),
        macis::rank4_span<double>(trdm_du_.data(), n_active_, n_active_,
                                  n_active_, n_active_),
        macis::rank4_span<double>(trdm_dd_.data(), n_active_, n_active_,
                                  n_active_, n_active_));

    {  // Possible bug fix
      for(int i = 0; i < n_active4_; i++) {
        trdm_dd_[i] = 2.0 * trdm_dd_[i];
        trdm_uu_[i] = 2.0 * trdm_uu_[i];
        trdm_ud_[i] = 2.0 * trdm_ud_[i];
        trdm_du_[i] = 2.0 * trdm_du_[i];
      }
    }

    Transform_2RDMs(n_active_, ordm_u_, ordm_d_, trdm_uu_, trdm_ud_, trdm_du_,
                    trdm_dd_);

    std::cout << " Constructor done\n" << std::endl;  // DEBUG
  }

  template<size_t N>
  double CompObservables<N>::compute_double_occupancies() const {
    double orb_db_occs = 0.0;

    // db_occs = orb[i,a]orb[i,c] <c+a c+b c+b c+a> orb.T[b,i]orb.T[d,i]
    for(int i = 0; i < n_imp_; i++)
      for(int a = 0; a < n_imp_; a++)
        for(int b = 0; b < n_imp_; b++)
          for(int c = 0; c < n_imp_; c++)
            for(int d = 0; d < n_imp_; d++) {
              orb_db_occs +=
                  orb_rot_[i + a * n_active_] * orb_rot_[i + c * n_active_] *
                  trdm_ud_[a + b * n_active_ + c * n_active2_ +
                           d * n_active3_] *
                  orb_rot_[i + b * n_active_] * orb_rot_[i + d * n_active_];
            }
    return orb_db_occs / n_imp_;
  }

  template<size_t N>
  std::vector<double> CompObservables<N>::compute_db_occs_matrix() const {
    std::vector<double> db_occs_matrix(2, 0.0);

    for(size_t site_i = 0; site_i < n_sites_; site_i++) {
      for(size_t site_j = 0; site_j < n_sites_; site_j++) {
        for(size_t band_i = 0; band_i < n_bands_; band_i++) {
          int i = site_i + n_sites_ * band_i;
          for(size_t band_j = 0; band_j < n_bands_; band_j++) {
            int j = site_j + n_sites_ * band_j;
            for(size_t a = 0; a < n_imp_; a++)
              for(size_t b = 0; b < n_imp_; b++)
                for(size_t c = 0; c < n_imp_; c++)
                  for(size_t d = 0; d < n_imp_; d++) {
		    if(i == j) {
                    	db_occs_matrix[0] +=
                  		orb_rot_[i + a * n_active_] * orb_rot_[i + c * n_active_] *
                  		trdm_ud_[a + b * n_active_ + c * n_active2_ +
                  		         d * n_active3_] *
                  		orb_rot_[i + b * n_active_] * orb_rot_[i + d * n_active_];
		    }
		    else if(i > j) {
                    	db_occs_matrix[1] +=
                    		  orb_rot_[i + a * n_active_] *
                        	  orb_rot_[j + c * n_active_] *
                        		(trdm_uu_[a + b * n_active_ + c * n_active2_ +
                         			  d * n_active3_] +
		                         trdm_du_[a + b * n_active_ + c * n_active2_ +
                		                  d * n_active3_] +
		                         trdm_ud_[a + b * n_active_ + c * n_active2_ +
                		                  d * n_active3_] +
		                         trdm_dd_[a + b * n_active_ + c * n_active2_ +
                		                  d * n_active3_] ) * 
                        	  orb_rot_[i + b * n_active_] *
                                  orb_rot_[j + d * n_active_];
		    }
                  }
          }
        }
      }
    }

    return db_occs_matrix;
  }

  template<size_t N>
  std::vector<double> CompObservables<N>::compute_sz_sz_correlations() const {
    std::vector<double> sz_sz(n_sites2_, 0.0);
    // sz_sz = 0.25 * orb[i,a]orb[j,c] <(n_a - n_d)_a (n_a - n_d)_c> orb.T[b,i]orb.T[d,j]
    for(size_t site_i = 0; site_i < n_sites_; site_i++) {
      for(size_t site_j = 0; site_j < n_sites_; site_j++) {
        for(size_t band_i = 0; band_i < n_bands_; band_i++) {
          int i = site_i + n_sites_ * band_i;
          for(size_t band_j = 0; band_j < n_bands_; band_j++) {
            int j = site_j + n_sites_ * band_j;
            for(size_t a = 0; a < n_imp_; a++)
              for(size_t b = 0; b < n_imp_; b++)
                for(size_t c = 0; c < n_imp_; c++)
                  for(size_t d = 0; d < n_imp_; d++) {
                    sz_sz[site_i + site_j * n_sites_] +=
                        0.25 * orb_rot_[i + a * n_active_] *
                        orb_rot_[j + c * n_active_] *
                        (trdm_uu_[a + b * n_active_ + c * n_active2_ +
                                  d * n_active3_] -
                         trdm_ud_[a + b * n_active_ + c * n_active2_ +
                                  d * n_active3_] -
                         trdm_du_[a + b * n_active_ + c * n_active2_ +
                                  d * n_active3_] +
                         trdm_dd_[a + b * n_active_ + c * n_active2_ +
                                  d * n_active3_]) *
                        orb_rot_[i + b * n_active_] *
                        orb_rot_[j + d * n_active_];
                  }
          }
        }
      }
    }
    return sz_sz;
  }

  template<size_t N>
  std::vector<double> CompObservables<N>::compute_tz_tz_correlations() const {
    std::vector<double> tz_tz(n_sites2_, 0.0);
    for(size_t site_i = 0; site_i < n_sites_; site_i++) {
      for(size_t site_j = 0; site_j < n_sites_; site_j++) {
        for(size_t band_i = 0; band_i < n_bands_; band_i++) {
          int i = site_i + n_sites_ * band_i;
          for(size_t band_j = 0; band_j < n_bands_; band_j++) {
            int j = site_j + n_sites_ * band_j;
            double sign = (band_i == band_j) ? 1.0 : -1.0;
            for(size_t a = 0; a < n_imp_; a++)
              for(size_t b = 0; b < n_imp_; b++)
                for(size_t c = 0; c < n_imp_; c++)
                  for(size_t d = 0; d < n_imp_; d++) {
                    tz_tz[site_i + site_j * n_sites_] +=
                        0.25 * sign * orb_rot_[i + a * n_active_] *
                        orb_rot_[j + c * n_active_] *
                        (trdm_uu_[a + b * n_active_ + c * n_active2_ +
                                  d * n_active3_] +
                         trdm_ud_[a + b * n_active_ + c * n_active2_ +
                                  d * n_active3_] +
                         trdm_du_[a + b * n_active_ + c * n_active2_ +
                                  d * n_active3_] +
                         trdm_dd_[a + b * n_active_ + c * n_active2_ +
                                  d * n_active3_]) *
                        orb_rot_[i + b * n_active_] *
                        orb_rot_[j + d * n_active_];
                  }
          }
        }
      }
    }
    return tz_tz;
  }

  template<size_t N>
  std::vector<double> CompObservables<N>::compute_charge_charge_correlations() const {
    std::vector<double> charge_charge(n_imp2_, 0.0);
    std::vector<double> mean_nn(n_imp2_, 0.0);
    std::vector<double> mean_n_mean_n(n_imp2_, 0.0);
    for(size_t site_i = 0; site_i < n_sites_; site_i++) {
      for(size_t site_j = 0; site_j < n_sites_; site_j++) {
        for(size_t band_i = 0; band_i < n_bands_; band_i++) {
          int i = site_i + n_sites_ * band_i;
          for(size_t band_j = 0; band_j < n_bands_; band_j++) {
            int j = site_j + n_sites_ * band_j;
            for(size_t a = 0; a < n_imp_; a++)
              for(size_t b = 0; b < n_imp_; b++)
                for(size_t c = 0; c < n_imp_; c++)
                  for(size_t d = 0; d < n_imp_; d++) {
                    mean_nn[i + j * n_imp_] +=
                      orb_rot_[i + a * n_active_] *
                        orb_rot_[j + c * n_active_] *
                        (trdm_uu_[a + b * n_active_ + c * n_active2_ +
                                  d * n_active3_] +
                         trdm_ud_[a + b * n_active_ + c * n_active2_ +
                                  d * n_active3_] +
                         trdm_du_[a + b * n_active_ + c * n_active2_ +
                                  d * n_active3_] +
                         trdm_dd_[a + b * n_active_ + c * n_active2_ +
                                  d * n_active3_] ) * 
                        orb_rot_[i + b * n_active_] *
                        orb_rot_[j + d * n_active_];
                    mean_n_mean_n[i + j * n_imp_] += 
                      orb_rot_[i + a * n_active_] * 
                        orb_rot_[j + c * n_active_] *
                        (ordm_u_[a + b * n_active_] * ordm_u_[c + d * n_active_] +
                         ordm_u_[a + b * n_active_] * ordm_d_[c + d * n_active_] +
                         ordm_d_[a + b * n_active_] * ordm_u_[c + d * n_active_] +
                         ordm_d_[a + b * n_active_] * ordm_d_[c + d * n_active_])
                         * orb_rot_[i + b * n_active_] * 
                         orb_rot_[j + d * n_active_];
                  }
            charge_charge[i + j * n_imp_] = mean_nn[i + j * n_imp_] - mean_n_mean_n[i + j * n_imp_];
          }
        }
      }
    }

    // // Debug print mean_nn and mean_n_mean_n
    // for(size_t i = 0; i < n_imp_; i++) {
    //   for(size_t j = 0; j < n_imp_; j++) {
    //     std::cout << "---- Debug Print ----" << std::endl;
    //     std::cout << "mean_nn[" << i << "][" << j << "] = " << mean_nn[i + j * n_imp_] << std::endl;
    //     std::cout << "mean_n_mean_n[" << i << "][" << j << "] = " << mean_n_mean_n[i + j * n_imp_] << std::endl;
    //   }
    // }

    return charge_charge;
  }
  // Explicit template instantiations for commonly used template parameter
  template double Comp_db_occs<64>(impurity_params<64>& p);
  template class CompObservables<64>;

}  // namespace macis
