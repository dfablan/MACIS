/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace macis {
namespace util {

/**
 * @brief Print a matrix with proper formatting
 *
 * @tparam T Data type (double, float, etc.)
 * @param mat Matrix data stored as a flat array
 * @param rows Number of rows
 * @param cols Number of columns
 * @param name Optional name for the matrix (will be printed as header)
 * @param is_column_major Whether the matrix is stored in column-major order
 * (default: true)
 * @param width Field width for each element (default: 12)
 * @param output Output stream (default: std::cout)
 */
template <typename T>
void write_matrix(const T* mat, int rows, int cols, const std::string& filename,
                  bool is_column_major = true, int width = 12) {
  std::ofstream output_file(filename);
  output_file.precision(std::numeric_limits<double>::max_digits10);
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      T value;
      if(is_column_major) {
        value = mat[j * rows + i];  // Column-major indexing
      } else {
        value = mat[i * cols + j];  // Row-major indexing
      }
      output_file << std::scientific << std::setw(width) << value << " ";
    }
    output_file << "\n";
  }
  output_file << std::defaultfloat;  // Reset formatting
}

/**
 * @brief Print a matrix stored in a std::vector
 *
 * @tparam T Data type (double, float, etc.)
 * @param mat Matrix data stored as std::vector
 * @param rows Number of rows
 * @param cols Number of columns
 * @param name Optional name for the matrix
 * @param is_column_major Whether the matrix is stored in column-major order
 * (default: true)
 * @param width Field width for each element (default: 12)
 * @param output Output stream (default: std::cout)
 */
template <typename T>
void write_matrix(const std::vector<T>& mat, int rows, int cols,
                  const std::string& filename, bool is_column_major = true,
                  int width = 12) {
  print_matrix(mat.data(), rows, cols, filename, is_column_major, width);
}

/**
 * @brief Print a vector with proper formatting
 *
 * @tparam T Data type (double, float, etc.)
 * @param vec Vector data
 * @param name Optional name for the vector
 * @param width Field width for each element (default: 12)
 * @param output Output stream (default: std::cout)
 */
template <typename T>
void write_vector(const std::vector<T>& vec, const std::string& filename,
                  int width = 12) {
  std::ofstream output_file(filename);
  output_file.precision(std::numeric_limits<double>::max_digits10);
  for(const auto& val : vec) {
    output_file << std::scientific << std::setw(width) << val << " ";
  }
  output_file << "\n" << std::defaultfloat;
}

/**
 * @brief Print a vector from raw pointer
 *
 * @tparam T Data type (double, float, etc.)
 * @param vec Vector data as raw pointer
 * @param size Size of the vector
 * @param name Optional name for the vector
 * @param width Field width for each element (default: 12)
 * @param output Output stream (default: std::cout)
 */
template <typename T>
void write_vector(const T* vec, int size, const std::string& filename,
                  int width = 12) {
  std::ofstream output_file(filename);
  output_file.precision(std::numeric_limits<double>::max_digits10);
  for(int i = 0; i < size; i++) {
    output_file << std::scientific << std::setw(width) << vec[i] << " ";
  }
  output_file << "\n" << std::defaultfloat;
}

}  // namespace util
}  // namespace macis