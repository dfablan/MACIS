#pragma once
#include "bitset_operations.hpp"
#include "cmz_ed/combins.h++"
#include <cassert>
#include <numeric>
#include <algorithm>

namespace dbwy {

template <size_t N>
std::bitset<N> canonical_hf_determinant( uint32_t nalpha, uint32_t nbeta ) {
  static_assert( (N%2) == 0, "N Must Be Even");
  std::bitset<N> alpha = full_mask<N>(nalpha);
  std::bitset<N> beta  = full_mask<N>(nbeta) << (N/2);
  return alpha | beta;
}

template <size_t N>
std::bitset<N> canonical_hf_determinant( uint32_t nalpha, uint32_t nbeta, const std::vector<double> &orb_ens ) {
  static_assert( (N%2) == 0, "N Must Be Even");
  // First, find the sorted indices for the orbital energies
  std::vector<size_t> idx(orb_ens.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
       [&orb_ens](size_t i1, size_t i2) {return orb_ens[i1] < orb_ens[i2];});
  // Next, fill the electrons by energy
  std::bitset<N> alpha(0), beta(0);
  for( int i = 0; i < nalpha; i++ )
    alpha.flip( idx[i] );
  for( int i = 0; i < nbeta; i++ )
    beta.flip( idx[i] + N/2 );
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
void bitset_to_occ_vir_as( size_t norb, std::bitset<N> state, 
  std::vector<uint32_t>& occ, std::vector<uint32_t>& vir,
  const std::vector<uint32_t>& as_orbs ) {

  occ.clear();
  for( const auto i : as_orbs )
    if( state[i] ) occ.emplace_back(i);
  const auto nocc = occ.size();
  assert( nocc <= norb );

  const auto nvir = as_orbs.size() - nocc;
  vir.resize(nvir);
  auto it = vir.begin();
  for( const auto i : as_orbs )
    if( !state[i] ) *(it++) = i;

}


template <size_t N>
void append_singles( std::bitset<N> state, 
  const std::vector<uint32_t>& occ, const std::vector<uint32_t>& vir,
  std::vector<std::bitset<N>>& singles ) {

  const size_t nocc = occ.size();
  const size_t nvir = vir.size();
  const std::bitset<N> one = 1ul;

  singles.clear();
  singles.reserve(nocc*nvir);

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

  const size_t nocc = occ.size();
  const size_t nvir = vir.size();
  const std::bitset<N> one = 1ul;

  doubles.clear();
  const size_t nv2 = (nvir * (nvir-1)) / 2;
  const size_t no2 = (nocc * (nocc-1)) / 2;
  doubles.reserve(nv2 * no2);

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
void generate_singles( size_t norb, std::bitset<N> state, 
  std::vector<std::bitset<N>>& singles ) {

  std::vector<uint32_t> occ_orbs, vir_orbs;
  bitset_to_occ_vir( norb, state, occ_orbs, vir_orbs );

  singles.clear();
  append_singles(  state, occ_orbs, vir_orbs, singles );

}

template <size_t N>
void generate_singles_spin( size_t norb, std::bitset<N> state, 
  std::vector<std::bitset<N>>& singles ) {

  auto state_alpha = truncate_bitset<N/2>(state);
  auto state_beta  = truncate_bitset<N/2>(state >> (N/2));

  std::vector<std::bitset<N/2>> singles_alpha, singles_beta;

  // Generate Spin-Specific singles / doubles
  generate_singles( norb, state_alpha, singles_alpha );
  generate_singles( norb, state_beta,  singles_beta  );

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
void generate_singles_as( size_t norb, std::bitset<N> state, 
  std::vector<std::bitset<N>>& singles, const std::vector<uint32_t>& as_orbs ) {

  std::vector<uint32_t> occ_orbs, vir_orbs;
  bitset_to_occ_vir_as<N>( norb, state, occ_orbs, vir_orbs, as_orbs );

  singles.clear();
  append_singles(  state, occ_orbs, vir_orbs, singles );

}

template <size_t N>
void generate_singles_doubles_as( size_t norb, std::bitset<N> state, 
  std::vector<std::bitset<N>>& singles, std::vector<std::bitset<N>>& doubles,
  const std::vector<uint32_t> &as_orbs ) {

  std::vector<uint32_t> occ_orbs, vir_orbs;
  bitset_to_occ_vir_as<N>( norb, state, occ_orbs, vir_orbs, as_orbs );

  singles.clear(); doubles.clear();
  append_singles(  state, occ_orbs, vir_orbs, singles );
  append_doubles(  state, occ_orbs, vir_orbs, doubles );

}

template <size_t N>
void generate_singles_spin_as( size_t norb, std::bitset<N> state, 
  std::vector<std::bitset<N>>& singles, const std::vector<uint32_t> as_orbs ) {

  auto state_alpha = truncate_bitset<N/2>(state);
  auto state_beta  = truncate_bitset<N/2>(state >> (N/2));

  std::vector<std::bitset<N/2>> singles_alpha, singles_beta;

  // Generate Spin-Specific singles 
  generate_singles_as( norb, state_alpha, singles_alpha, as_orbs );
  generate_singles_as( norb, state_beta,  singles_beta,  as_orbs );

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

}

template <size_t N>
void generate_singles_doubles_spin_as( size_t norb, std::bitset<N> state, 
  std::vector<std::bitset<N>>& singles, std::vector<std::bitset<N>>& doubles,
  const std::vector<uint32_t> &as_orbs ) {

  auto state_alpha = truncate_bitset<N/2>(state);
  auto state_beta  = truncate_bitset<N/2>(state >> (N/2));

  std::vector<std::bitset<N/2>> singles_alpha, singles_beta;
  std::vector<std::bitset<N/2>> doubles_alpha, doubles_beta;

  // Generate Spin-Specific singles / doubles
  generate_singles_doubles_as( norb, state_alpha, singles_alpha, doubles_alpha, as_orbs );
  generate_singles_doubles_as( norb, state_beta,  singles_beta,  doubles_beta,  as_orbs );

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
void generate_singles_spin( size_t norb, std::bitset<N> state, 
  std::vector<std::bitset<N>>& singles ) {

  auto state_alpha = truncate_bitset<N/2>(state);
  auto state_beta  = truncate_bitset<N/2>(state >> (N/2));

  std::vector<std::bitset<N/2>> singles_alpha, singles_beta;

  // Generate Spin-Specific singles / doubles
  generate_singles( norb, state_alpha, singles_alpha );
  generate_singles( norb, state_beta,  singles_beta  );

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
std::vector<std::bitset<N> > generate_full_hilbert_space( 
  size_t norb,
  size_t nalpha,
  size_t nbeta )
{
  std::vector<std::bitset<N> > dets;
  // Make all possible states of Norbs with Nups and Ndos
  // bits set.
  std::vector<unsigned long> up_stts = cmz::ed::BuildCombs( norb, nalpha );
  std::vector<unsigned long> do_stts = cmz::ed::BuildCombs( norb, nbeta );

  for( size_t iup = 0; iup < up_stts.size(); iup++ )
    for( size_t ido = 0; ido < do_stts.size(); ido++ )
    {
      std::bitset<N> st = (std::bitset<N>(do_stts[ido]) << (N/2)) 
                          | std::bitset<N>(up_stts[iup]);
      dets.emplace_back( st );
    }

  return dets;
}




template <size_t N>
void generate_cis_hilbert_space( size_t norb, std::bitset<N> state, 
  std::vector<std::bitset<N>>& dets ) {

  dets.clear();
  dets.emplace_back(state);
  std::vector<std::bitset<N>> singles;
  generate_singles_spin( norb, state, singles );
  dets.insert( dets.end(), singles.begin(), singles.end() );

}


template <size_t N>
std::vector<std::bitset<N>> generate_cis_hilbert_space( size_t norb, 
  std::bitset<N> state ) {
  std::vector<std::bitset<N>> dets;
  generate_cis_hilbert_space( norb, state, dets );
  return dets;
}


template <size_t N>
uint32_t first_occupied_flipped( std::bitset<N> state, std::bitset<N> ex ) {
  return ffs( state & ex ) - 1u;
}

template <size_t N>
double single_excitation_sign( std::bitset<N> state, unsigned p, unsigned q ) {
  std::bitset<N> mask = 0ul;

  if( p > q ) {
    mask = state & ( full_mask<N>(p) ^ full_mask<N>(q+1) );
  } else {
    mask = state & ( full_mask<N>(q) ^ full_mask<N>(p+1) );
  }
  return (mask.count() % 2) ? -1. : 1.;
}

template <size_t N>
inline auto single_excitation_sign_indices( std::bitset<N> bra,
  std::bitset<N> ket, std::bitset<N> ex ) {

  auto o1   = first_occupied_flipped( ket, ex );
  auto v1   = first_occupied_flipped( bra, ex );
  auto sign = single_excitation_sign( ket, v1, o1 );

  return std::make_tuple(o1,v1,sign);
}

template <size_t N>
inline auto doubles_sign_indices( std::bitset<N> bra, std::bitset<N> ket, 
  std::bitset<N> ex ) {

    const auto o1 = first_occupied_flipped( ket, ex );
    const auto v1 = first_occupied_flipped( bra, ex );
    auto sign = single_excitation_sign(ket, v1, o1);

    ket.flip(o1).flip(v1);
    ex.flip(o1).flip(v1);

    const auto o2 = first_occupied_flipped( ket, ex );
    const auto v2 = first_occupied_flipped( bra, ex );
    sign *= single_excitation_sign(ket, v2, o2);

    return std::make_tuple(o1,v1,o2,v2,sign);
}

template <size_t N>
inline auto doubles_sign( std::bitset<N> bra, std::bitset<N> ket, 
  std::bitset<N> ex ) {

  auto [p,q,r,s,sign] = doubles_sign_indices(bra,ket,ex);
  return sign;
}




template <size_t N>
void generate_residues( std::bitset<N> state, std::vector<std::bitset<N>>& res ) {

  auto state_alpha = truncate_bitset<N/2>(state);
  auto state_beta  = truncate_bitset<N/2>(state >> (N/2));

  //auto occ_alpha = bits_to_indices(state_alpha, occ_alpha);
  auto occ_alpha = bits_to_indices(state_alpha);
  const int nalpha = occ_alpha.size();

  //auto occ_beta = bits_to_indices(state_beta, occ_beta);
  auto occ_beta = bits_to_indices(state_beta);
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


template <size_t N>
std::string to_canonical_string( std::bitset<N> state ) {
  static_assert((N%2)==0, "N Odd");
  auto state_alpha = truncate_bitset<N/2>(state);
  auto state_beta  = truncate_bitset<N/2>(state >> (N/2));
  std::string str;
  for( auto i = 0; i < N/2; ++i ) {
    if( state_alpha[i] and state_beta[i] ) str.push_back('2');
    else if( state_alpha[i] )              str.push_back('u');
    else if( state_beta[i]  )              str.push_back('d');
    else                                   str.push_back('0');
  }
  return str;
}

template <size_t N>
std::bitset<N> from_canonical_string( std::string str ) {
    std::bitset<N> state_alpha(0), state_beta(0);
    for( auto i = 0; i < str.length(); ++i ) {
      if( str[i] == '2' ) {
        state_alpha.set(i);
	state_beta.set(i);
      } else if( str[i] == 'u' ) {
        state_alpha.set(i);
      } else if( str[i] == 'd' ) {
	state_beta.set(i);
      }
    }
    auto state = state_alpha | (state_beta << (N/2));
    return state;
}

}
