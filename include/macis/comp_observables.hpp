#pragma once
#include "macis/impurity_solver.hpp"

using macis::NumActive;
using macis::NumCanonicalOccupied;
using macis::NumCanonicalVirtual;
using macis::NumElectron;
using macis::NumInactive;
using macis::NumOrbital;
using macis::NumVirtual;

namespace macis {

void Transform_2RDMs(const int norbs, const std::vector<double>& ordm_u,
                     const std::vector<double>& ordm_d,
                     std::vector<double>& trdm_uu, std::vector<double>& trdm_ud,
                     std::vector<double>& trdm_du,
                     std::vector<double>& trdm_dd);


template<size_t N>
double Comp_db_occs(impurity_params<N>& p);

template <size_t N>
class CompObservables {
 private:
  size_t& norb_;
  size_t& n_imp_;
  size_t& n_bands_;
  size_t n_sites_;
  size_t n_sites2_;
  size_t n_imp2_;
  size_t n_imp3_;
  size_t n_imp4_;
  size_t& n_active_;
  size_t n_active2_;
  size_t n_active3_;
  size_t n_active4_;
  size_t& n_inactive_;
  size_t norb2_;

  std::vector<double> ordm_u_, ordm_d_;
  std::vector<double> trdm_uu_, trdm_dd_, trdm_ud_, trdm_du_;

  std::vector<macis::wfn_t<N>>& dets_;
  std::vector<double>& C_;
  std::vector<double>& orb_rot_;

  std::vector<double>& T_active;
  std::vector<double>& V_active;
  std::vector<double>& F_inactive;

 public:
  CompObservables(impurity_params<N>& p) ;

  double compute_double_occupancies() const ;

  std::vector<double> compute_sz_sz_correlations() const ;

  std::vector<double> compute_tz_tz_correlations() const ;

};

}  // namespace macis