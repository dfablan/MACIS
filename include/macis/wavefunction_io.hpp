/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <bitset>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <macis/sd_operations.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace macis {

/**
 *  @brief Header line of an ASCII CI wavefunction file.
 *
 *  Returned by read_wavefunction so a caller can check the wavefunction it just
 *  read against the run it is about to seed. The counts are written by
 *  write_wavefunction from the leading determinant, so they describe that
 *  determinant, not an independently recorded target.
 */
struct wavefunction_header {
  size_t nstate;  ///< Number of determinants recorded in the header
  size_t norb;    ///< Number of orbitals the wavefunction was written for
  size_t nalpha;  ///< Alpha electrons in the leading determinant
  size_t nbeta;   ///< Beta electrons in the leading determinant
};

/**
 *  @brief Read an ASCII CI wavefunction file
 *
 *  Format:
 *
 *    <NSTATE> <NORB> <NALPHA> <NBETA>
 *    <COEFF_1> <STR_1>
 *    <COEFF_2> <STR_2>
 *    ...
 *    <COEFF_NSTATE> <STR_NSTATE>
 *    [EOF]
 *
 *  Throws if the file cannot be opened, if the header or a body line is
 *  malformed, if the file contains no determinants, or if a determinant
 *  occupies an orbital that does not fit in a bitset of width @p N.
 *
 *  @tparam N Bitset width of the determinant bitstring
 *
 *  @param[in] fname Name of file to read
 *  @param[out] states The determinants of the wave function
 *  @param[out] coeffs The coefficients of the wave function
 *  @param[in] check_orbital_bounds Reject determinants that occupy an orbital
 *             beyond the N/2 this bitset width can represent. Defaults to false
 *             because tests/ref_data/ch4.wfn.dat is a 34-orbital file that
 *             existing callers read into a 64-bit bitset; 9552 of its
 *             determinants trip this check. New callers should pass true.
 *
 *  @returns The parsed header
 */
template <size_t N>
wavefunction_header read_wavefunction(std::string fname,
                                      std::vector<std::bitset<N>>& states,
                                      std::vector<double>& coeffs,
                                      bool check_orbital_bounds = false) {
  static_assert((N % 2) == 0, "N Odd");
  states.clear();
  coeffs.clear();

  std::ifstream file(fname);
  if(!file.is_open())
    throw std::runtime_error("Could not open wavefunction file: " + fname);

  std::string line;

  wavefunction_header header;
  {
    if(!std::getline(file, line))
      throw std::runtime_error("Wavefunction file is empty: " + fname);
    std::stringstream ss{line};
    std::string nstate_, norb_, nalpha_, nbeta_;
    ss >> nstate_ >> norb_ >> nalpha_ >> nbeta_;
    if(nstate_.empty() or norb_.empty() or nalpha_.empty() or nbeta_.empty())
      throw std::runtime_error(
          "Malformed header in wavefunction file " + fname +
          " (expected \"<NSTATE> <NORB> <NALPHA> <NBETA>\"), got: " + line);
    try {
      header.nstate = std::stoul(nstate_);
      header.norb = std::stoul(norb_);
      header.nalpha = std::stoul(nalpha_);
      header.nbeta = std::stoul(nbeta_);
    } catch(const std::exception&) {
      throw std::runtime_error("Malformed header in wavefunction file " + fname +
                               ": " + line);
    }
  }

  states.reserve(header.nstate);
  coeffs.reserve(header.nstate);
  while(std::getline(file, line)) {
    std::stringstream ss{line};
    std::string c, d;
    ss >> c >> d;
    if(c.empty() and d.empty()) continue;  // Tolerate blank lines
    if(d.empty())
      throw std::runtime_error("Malformed line in wavefunction file " + fname +
                               ": " + line);

    // A determinant string may be longer than the N/2 orbitals this bitset
    // width can hold, but only if the orbitals past that point are empty:
    // from_canonical_string would otherwise fold an alpha occupation into the
    // beta half of the packed word and silently corrupt the determinant.
    for(size_t i = check_orbital_bounds ? N / 2 : d.size(); i < d.size(); ++i)
      if(d[i] != '0')
        throw std::runtime_error(
            "Wavefunction file " + fname + " occupies orbital " +
            std::to_string(i) + ", which does not fit in a " +
            std::to_string(N / 2) + "-orbital determinant (bitset width N = " +
            std::to_string(N) + ")");

    try {
      coeffs.emplace_back(std::stod(c));
    } catch(const std::exception&) {
      throw std::runtime_error("Malformed coefficient in wavefunction file " +
                               fname + ": " + line);
    }
    states.emplace_back(from_canonical_string<N>(d));
  }

  if(states.empty())
    throw std::runtime_error("Wavefunction file contains no determinants: " +
                             fname);

  // The header count and the body must agree. They disagree when a file was
  // truncated -- a job killed while writing a large wavefunction -- which would
  // otherwise be accepted as a smaller, perfectly plausible wavefunction.
  if(states.size() != header.nstate)
    throw std::runtime_error(
        "Wavefunction file " + fname + " declares " +
        std::to_string(header.nstate) + " determinants but contains " +
        std::to_string(states.size()) +
        "; the file is truncated or its header is wrong");

  return header;
}

/**
 *  @brief Write an ASCII CI wavefunction file
 *
 *  Format:
 *
 *    <NSTATE> <NORB> <NALPHA> <NBETA>
 *    <COEFF_1> <STR_1>
 *    <COEFF_2> <STR_2>
 *    ...
 *    <COEFF_NSTATE> <STR_NSTATE>
 *    [EOF]
 *
 *  @tparam N Bitset width of the determinant bitstring
 *
 *  @param[in] fname Name of file to read
 *  @param[in] norb  Number of orbitals
 *  @param[in] states The determinants of the wave function
 *  @param[in] coeffs The coefficients of the wave function
 */
template <size_t N>
void write_wavefunction(std::string fname, size_t norb,
                        const std::vector<std::bitset<N>>& states,
                        const std::vector<double>& coeffs) {
  if(states.size() != coeffs.size())
    throw std::runtime_error("Invalid Wave Function Dimensions");

  if(!states.size()) return;

  const auto nstates = states.size();
  const auto nalpha = (states[0] << N / 2).count();
  const auto nbeta = (states[0] >> N / 2).count();

  std::ofstream file(fname);
  file << nstates << " " << norb << " " << nalpha << " " << nbeta << std::endl;
  file << std::scientific << std::setprecision(16);
  for(size_t i = 0; i < nstates; ++i) {
    file << std::setw(30) << coeffs[i] << " " << to_canonical_string(states[i])
         << " " << std::endl;
  }
}

}  // namespace macis
