/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <macis/util/fcidump.hpp>
#include <regex>
#include <sstream>
#include <string>

// std::vector<std::string> split(const std::string str,
//                                const std::string regex_str) {
//   std::regex regexz(regex_str);
//   std::vector<std::string> list(
//       std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
//       std::sregex_token_iterator());
//   std::vector<std::string> clean_list;
//   std::copy_if(list.begin(), list.end(), std::back_inserter(clean_list),
//                [](auto& s) { return s.size() > 0; });
//   return clean_list;
// }

static std::vector<std::string> tokenize_ws(const std::string& line) {
  std::istringstream iss(line);
  std::vector<std::string> toks;
  std::string t;
  while(iss >> t) toks.push_back(t);
  return toks;
}

namespace {

// True iff the *entire* token is a signed integer. Note that std::stoi alone is
// not sufficient to decide this: it happily parses a leading integer prefix and
// stops at the first foreign character, so stoi("0.5") returns 0 without
// throwing. Requiring the full token to be consumed is what distinguishes an
// orbital index from an integral.
bool is_integer_token(const std::string& str) {
  size_t pos = 0;
  try {
    (void)std::stoi(str, &pos);
  } catch(const std::exception&) {
    return false;
  }
  return pos == str.size();
}

int32_t parse_index(const std::string& str) {
  size_t pos = 0;
  int32_t idx = std::stoi(str, &pos);
  if(pos != str.size())
    throw std::runtime_error("Malformed orbital index: " + str);
  if(idx < 0) throw std::runtime_error("Invalid Orb Idx: " + str);
  return idx;
}

double parse_integral(const std::string& str) {
  size_t pos = 0;
  double val = std::stod(str, &pos);
  if(pos != str.size()) throw std::runtime_error("Malformed integral: " + str);
  return val;
}

}  // namespace

auto fcidump_line(const std::vector<std::string>& tokens) {
  if(tokens.size() != 5) throw std::runtime_error("Invalid FCIDUMP Line");

  // Two layouts are accepted: "<p> <q> <r> <s> <integral>" (what write_fcidump
  // emits) and "<integral> <p> <q> <r> <s>" (the Molpro/PySCF convention).
  // Decide by which end of the line is a bare integer.
  const bool front_is_int = is_integer_token(tokens.front());
  const bool back_is_int = is_integer_token(tokens.back());

  bool idx_first;
  if(front_is_int and not back_is_int)
    idx_first = true;  // 1 1 1 1 0.5
  else if(back_is_int)
    idx_first = false;  // 0.5 1 1 1 1, and the all-integer tie ("4 1 1 1 1"),
                        // where integral-first is the FCIDUMP standard
  else
    throw std::runtime_error("Invalid FCIDUMP Line");  // neither end an index

  const size_t i0 = idx_first ? 0 : 1;  // first index token
  const size_t ii = idx_first ? 4 : 0;  // integral token

  try {
    auto p = parse_index(tokens[i0 + 0]);
    auto q = parse_index(tokens[i0 + 1]);
    auto r = parse_index(tokens[i0 + 2]);
    auto s = parse_index(tokens[i0 + 3]);
    return std::make_tuple(p, q, r, s, parse_integral(tokens[ii]));
  } catch(const std::exception& e) {
    std::ostringstream oss;
    oss << "Invalid FCIDUMP line [" << e.what() << "]:";
    for(const auto& t : tokens) oss << " " << t;
    throw std::runtime_error(oss.str());
  }
}

enum LineClassification { Core, OneBody, TwoBody };

LineClassification line_classification(int p, int q, int r, int s) {
  if(!(p or q or r or s))
    return LineClassification::Core;
  else if(p and q and r and s)
    return LineClassification::TwoBody;
  else
    return LineClassification::OneBody;
}

namespace macis {

uint32_t read_fcidump_norb(std::string fname) {
  std::ifstream file(fname);
  std::string line;
  int32_t max_idx = 0;
  while(std::getline(file, line)) {
    auto tokens = tokenize_ws(line);
    if(tokens.size() != 5) continue;  // not a valid FCIDUMP line

    auto [p, q, r, s, integral] = fcidump_line(tokens);

    max_idx = std::max(max_idx, std::max(p, std::max(q, std::max(r, s))));
  }

  return max_idx;
}

double read_fcidump_core(std::string fname) {
  std::ifstream file(fname);
  std::string line;
  while(std::getline(file, line)) {
    auto tokens = tokenize_ws(line);
    if(tokens.size() != 5) continue;  // not a valid FCIDUMP line

    auto [p, q, r, s, integral] = fcidump_line(tokens);
    auto lc = line_classification(p, q, r, s);
    if(lc == LineClassification::Core) {
      return integral;
    }
  }
  return 0.0;
}

void read_fcidump_1body(std::string fname, col_major_span<double, 2> T) {
  if(T.extent(0) != T.extent(1)) throw std::runtime_error("T must be square");

  auto norb = read_fcidump_norb(fname);
  if(T.extent(0) != norb)
    throw std::runtime_error("T is of improper dimension");

  std::ifstream file(fname);
  std::string line;
  while(std::getline(file, line)) {
    auto tokens = tokenize_ws(line);
    if(tokens.size() != 5) continue;  // not a valid FCIDUMP line

    auto [p, q, r, s, integral] = fcidump_line(tokens);
    auto lc = line_classification(p, q, r, s);
    if(lc == LineClassification::OneBody) {
      p--;
      q--;
      T(p, q) = integral;
      T(q, p) = integral;
    }
  }
}

void read_fcidump_1body(std::string fname, double* T, size_t LDT) {
  auto norb = read_fcidump_norb(fname);
  col_major_span<double, 2> T_map(T, LDT, norb);
  read_fcidump_1body(
      fname, Kokkos::submdspan(T_map, std::pair{0, norb}, Kokkos::full_extent));
}

void read_fcidump_2body(std::string fname, col_major_span<double, 4> V) {
  if(V.extent(0) != V.extent(1)) throw std::runtime_error("V must be square");
  if(V.extent(0) != V.extent(2)) throw std::runtime_error("V must be square");
  if(V.extent(0) != V.extent(3)) throw std::runtime_error("V must be square");

  auto norb = read_fcidump_norb(fname);
  if(V.extent(0) != norb)
    throw std::runtime_error("V is of improper dimension");

  std::ifstream file(fname);
  std::string line;
  while(std::getline(file, line)) {
    auto tokens = tokenize_ws(line);
    if(tokens.size() != 5) continue;  // not a valid FCIDUMP line

    auto [p, q, r, s, integral] = fcidump_line(tokens);
    auto lc = line_classification(p, q, r, s);
    if(lc == LineClassification::TwoBody) {
      p--;
      q--;
      r--;
      s--;
      V(p, q, r, s) = integral;  // (pq|rs)
      V(p, q, s, r) = integral;  // (pq|sr)
      V(q, p, r, s) = integral;  // (qp|rs)
      V(q, p, s, r) = integral;  // (qp|sr)

      V(r, s, p, q) = integral;  // (rs|pq)
      V(s, r, p, q) = integral;  // (sr|pq)
      V(r, s, q, p) = integral;  // (rs|qp)
      V(s, r, q, p) = integral;  // (sr|qp)
    }
  }
}

void read_fcidump_2body(std::string fname, double* V, size_t LDV) {
  auto norb = read_fcidump_norb(fname);
  col_major_span<double, 4> V_map(V, LDV, LDV, LDV, norb);
  auto sl = std::pair{0, norb};
  read_fcidump_2body(fname,
                     Kokkos::submdspan(V_map, sl, sl, sl, Kokkos::full_extent));
}

bool is_2body_diagonal(std::string fname) {
  auto norb = read_fcidump_norb(fname);
  bool all_diag = true;

  std::ifstream file(fname);
  std::string line;
  while(std::getline(file, line)) {
    auto tokens = tokenize_ws(line);
    if(tokens.size() != 5) continue;  // not a valid FCIDUMP line

    auto [p, q, r, s, integral] = fcidump_line(tokens);
    auto lc = line_classification(p, q, r, s);
    if(lc == LineClassification::TwoBody) {
      p--;
      q--;
      r--;
      s--;
      if(p != q || r != s) {
        all_diag = false;
        break;
      }
    }
  }
  return all_diag;
}

void write_fcidump(std::string fname, size_t norb, const double* T, size_t LDT,
                   const double* V, size_t LDV, double E_core) {
  auto logger = spdlog::basic_logger_mt("fcidump", fname);
  logger->set_pattern("%v");
  const std::string fmt_string = "{:8} {:8} {:8} {:8} {:25.16e}";

  // Write two body
  for(size_t i = 0; i < norb; ++i)
    for(size_t j = 0; j < norb; ++j)
      for(size_t k = 0; k < norb; ++k)
        for(size_t l = 0; l < norb; ++l) {
          if(V[i + j * LDV + k * LDV * LDV + l * LDV * LDV * LDV] != 0.0)
            logger->info(fmt_string, i + 1, j + 1, k + 1, l + 1,
                         V[i + j * LDV + k * LDV * LDV + l * LDV * LDV * LDV]);
        }

  // Write one body
  for(size_t i = 0; i < norb; ++i)
    for(size_t j = 0; j < norb; ++j) {
      if(T[i + j * LDT] != 0.0)
        logger->info(fmt_string, i + 1, j + 1, 0, 0, T[i + j * LDT]);
    }

  // Write core
  if(E_core != 0.0) logger->info(fmt_string, 0, 0, 0, 0, E_core);
}

void read_rdms_binary(std::string fname, size_t norb, double* ORDM, size_t LDD1,
                      double* TRDM, size_t LDD2) {
  // Zero out rdms in case of sparse data
  for(size_t i = 0; i < norb; ++i)
    for(size_t j = 0; j < norb; ++j) {
      ORDM[i + j * LDD1] = 0;
    }

  for(size_t i = 0; i < norb; ++i)
    for(size_t j = 0; j < norb; ++j)
      for(size_t k = 0; k < norb; ++k)
        for(size_t l = 0; l < norb; ++l) {
          TRDM[i + j * LDD2 + k * LDD2 * LDD2 + l * LDD2 * LDD2 * LDD2] = 0;
        }

  std::ifstream in_file(fname, std::ios::binary);
  if(!in_file) throw std::runtime_error(fname + " not available");

  int _norb_read;
  in_file.read((char*)&_norb_read, sizeof(int));
  if(_norb_read != (int)norb)
    throw std::runtime_error("NORB in RDM file doesn't match " +
                             std::to_string(norb) + " " +
                             std::to_string(_norb_read));

  std::vector<double> raw(norb * norb * norb * norb);

  // Read 1RDM
  in_file.read((char*)raw.data(), norb * norb * sizeof(double));
  for(size_t i = 0; i < norb; ++i)
    for(size_t j = 0; j < norb; ++j) {
      ORDM[i + j * LDD1] = raw[i + j * norb];
    }

  // Read 2RDM
  in_file.read((char*)raw.data(), norb * norb * norb * norb * sizeof(double));
  for(size_t i = 0; i < norb; ++i)
    for(size_t j = 0; j < norb; ++j)
      for(size_t k = 0; k < norb; ++k)
        for(size_t l = 0; l < norb; ++l) {
          TRDM[i + j * LDD2 + k * LDD2 * LDD2 + l * LDD2 * LDD2 * LDD2] =
              raw[i + j * norb + k * norb * norb + l * norb * norb * norb];
        }
}

void write_rdms_binary(std::string fname, size_t norb, const double* ORDM,
                       size_t LDD1, const double* TRDM, size_t LDD2) {
  std::ofstream out_file(fname, std::ios::binary);
  if(!out_file) throw std::runtime_error(fname + " not available");

  int _norb_write = norb;
  out_file.write((char*)&_norb_write, sizeof(int));

  std::vector<double> raw(norb * norb * norb * norb);

  // Pack and Write 1RDM
  for(size_t i = 0; i < norb; ++i)
    for(size_t j = 0; j < norb; ++j) {
      raw[i + j * norb] = ORDM[i + j * LDD1];
    }
  out_file.write((char*)raw.data(), norb * norb * sizeof(double));

  // Pack and Write 2RDM
  for(size_t i = 0; i < norb; ++i)
    for(size_t j = 0; j < norb; ++j)
      for(size_t k = 0; k < norb; ++k)
        for(size_t l = 0; l < norb; ++l) {
          raw[i + j * norb + k * norb * norb + l * norb * norb * norb] =
              TRDM[i + j * LDD2 + k * LDD2 * LDD2 + l * LDD2 * LDD2 * LDD2];
        }
  out_file.write((char*)raw.data(), norb * norb * norb * norb * sizeof(double));
}

}  // namespace macis
