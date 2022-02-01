#pragma once
#include "bitset_operations.hpp"
#include <cassert>

namespace dbwy {

template <size_t N>
std::bitset<N> canonical_hf_determinant( uint32_t nalpha, uint32_t nbeta ) {
  static_assert( (N%2) == 0, "N Must Be Even");
  std::bitset<N> alpha = full_mask<N>(nalpha);
  std::bitset<N> beta  = full_mask<N>(nbeta) << (N/2);
  return alpha | beta;
}

template <size_t N>
void bitset_to_occ_vir( size_t norb, std::bitset<N> state, 
  std::vector<uint32_t>& occ, std::vector<uint32_t>& vir ) {

  occ = bits_to_indices( state );
  const auto nocc = occ.size();
  assert( nocc < norb );

  const auto nvir = norb - nocc;
  vir.resize(nvir);
  auto it = vir.begin();
  for( size_t i = 0; i < norb; ++i )
  if( !state[i] ) *(it++) = i;

}


template <size_t N>
void append_singles( std::bitset<N> state, 
  const std::vector<uint32_t>& occ, const std::vector<uint32_t>& vir,
  std::vector<std::bitset<N>>& singles ) {

  singles.clear();
  const size_t nocc = occ.size();
  const size_t nvir = vir.size();
  const std::bitset<N> one = 1ul;

  for( size_t a = 0; a < nvir; ++a )
  for( size_t i = 0; i < nocc; ++i ) {
    std::bitset<N> ex = (one << occ[i]) ^ (one << vir[a]);
    singles.emplace_back( state ^ ex );
  }

}

template <size_t N>
void append_doubles( std::bitset<N> state, 
  const std::vector<uint32_t>& occ, const std::vector<uint32_t>& vir,
  std::vector<std::bitset<N>>& doubles ) {

  doubles.clear();
  const size_t nocc = occ.size();
  const size_t nvir = vir.size();
  const std::bitset<N> one = 1ul;

  for( size_t a = 0; a < nvir; ++a )
  for( size_t i = 0; i < nocc; ++i ) 
  for( size_t b = a+1; b < nvir; ++b )
  for( size_t j = i+1; j < nocc; ++j ) {
    std::bitset<N> ex = (one << occ[i]) ^ (one << occ[j]) ^
                        (one << vir[a]) ^ (one << vir[b]);
    doubles.emplace_back( state ^ ex );
  }

}

template <size_t N>
void generate_singles_doubles( size_t norb, std::bitset<N> state, 
  std::vector<std::bitset<N>>& singles, std::vector<std::bitset<N>>& doubles ) {

  std::vector<uint32_t> occ_orbs, vir_orbs;
  bitset_to_occ_vir( norb, state, occ_orbs, vir_orbs );

  singles.clear(); doubles.clear();
  append_singles(  state, occ_orbs, vir_orbs, singles );
  append_doubles(  state, occ_orbs, vir_orbs, doubles );

}

template <size_t N>
void generate_singles_doubles_spin( size_t norb, std::bitset<N> state, 
  std::vector<std::bitset<N>>& singles, std::vector<std::bitset<N>>& doubles ) {

  auto state_alpha = truncate_bitset<N/2>(state);
  auto state_beta  = truncate_bitset<N/2>(state >> (N/2));

  std::vector<std::bitset<N/2>> singles_alpha, singles_beta;
  std::vector<std::bitset<N/2>> doubles_alpha, doubles_beta;

  // Generate Spin-Specific singles / doubles
  generate_singles_doubles( norb, state_alpha, singles_alpha, doubles_alpha );
  generate_singles_doubles( norb, state_beta,  singles_beta,  doubles_beta  );

  auto state_alpha_expand = expand_bitset<N>(state_alpha);
  auto state_beta_expand  = expand_bitset<N>(state_beta) << (N/2);

  // Generate Singles in full space
  singles.clear();

  // Single Alpha + No Beta
  for( auto s_alpha : singles_alpha ) {
    auto s_state = expand_bitset<N>(s_alpha);
    s_state = s_state | state_beta_expand;
    singles.emplace_back(s_state);
  }

  // No Alpha + Single Beta
  for( auto s_beta : singles_beta ) {
    auto s_state = expand_bitset<N>(s_beta) << (N/2);
    s_state = s_state | state_alpha_expand;
    singles.emplace_back(s_state);
  }

  // Generate Doubles in full space
  doubles.clear();

  // Double Alpha + No Beta
  for( auto d_alpha : doubles_alpha ) {
    auto d_state = expand_bitset<N>(d_alpha);
    d_state = d_state | state_beta_expand;
    doubles.emplace_back(d_state);
  }

  // No Alpha + Double Beta
  for( auto d_beta : doubles_beta ) {
    auto d_state = expand_bitset<N>(d_beta) << (N/2);
    d_state = d_state | state_alpha_expand;
    doubles.emplace_back(d_state);
  }

  // Single Alpha + Single Beta
  for( auto s_alpha : singles_alpha )
  for( auto s_beta  : singles_beta  ) {
    auto d_state_alpha = expand_bitset<N>(s_alpha);
    auto d_state_beta  = expand_bitset<N>(s_beta) << (N/2);
    doubles.emplace_back( d_state_alpha | d_state_beta );
  }

}

template <size_t N>
void generate_cisd_hilbert_space( size_t norb, std::bitset<N> state, 
  std::vector<std::bitset<N>>& dets ) {

  dets.clear();
  dets.emplace_back(state);
  std::vector<std::bitset<N>> singles, doubles;
  generate_singles_doubles_spin( norb, state, singles, doubles );
  dets.insert( dets.end(), singles.begin(), singles.end() );
  dets.insert( dets.end(), doubles.begin(), doubles.end() );

}

template <size_t N>
std::vector<std::bitset<N>> generate_cisd_hilbert_space( size_t norb, 
  std::bitset<N> state ) {
  std::vector<std::bitset<N>> dets;
  generate_cisd_hilbert_space( norb, state, dets );
  return dets;
}

template <size_t N>
void generate_residues( std::bitset<N> state, std::vector<std::bitset<N>>& res ) {

  auto state_alpha = truncate_bitset<N/2>(state);
  auto state_beta  = truncate_bitset<N/2>(state >> (N/2));

  auto occ_alpha = bits_to_indices(state_alpha, occ_alpha);
  const int nalpha = occ_alpha.size();

  auto occ_beta = bits_to_indices(state_beta, occ_beta);
  const int nbeta  = occ_beta.size();

  std::bitset<N> state_alpha_full = expand_bitset<N>(state_alpha);
  std::bitset<N> state_beta_full  = expand_bitset<N>(state_beta); 
  state_beta_full = state_beta_full << (N/2);


  std::bitset<N/2> one = 1ul;

  // Double alpha
  for( auto i = 0;   i < nalpha; ++i ) 
  for( auto j = i+1; j < nalpha; ++j ) {
    auto mask = (one << occ_alpha[i]) | (one << occ_alpha[j]);
    std::bitset<N> _r = expand_bitset<N>(state_alpha & ~mask);
    res.emplace_back( _r | state_beta_full );
  }

  // Double beta
  for( auto i = 0;   i < nbeta; ++i ) 
  for( auto j = i+1; j < nbeta; ++j ) {
    auto mask = (one << occ_beta[i]) | (one << occ_beta[j]);
    std::bitset<N> _r = expand_bitset<N>(state_beta & ~mask) << (N/2);
    res.emplace_back( _r | state_alpha_full );
  }

  // Mixed
  for( auto i = 0; i < nalpha; ++i) 
  for( auto j = 0; j < nbeta;  ++j) {
    std::bitset<N> mask = expand_bitset<N>(one << occ_alpha[i]);
    mask = mask | (expand_bitset<N>(one << occ_beta[j]) << (N/2));
    res.emplace_back( state & ~mask );
  }

}

}