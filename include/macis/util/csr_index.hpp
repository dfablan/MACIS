/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace macis {

/**
 * @brief Throws if a CSR block of nrows x ncols cannot be indexed by index_t
 * (column indices are stored as index_t).
 */
template <typename index_t>
void check_csr_dims(size_t nrows, size_t ncols) {
  constexpr size_t imax = size_t(std::numeric_limits<index_t>::max());
  if(nrows > imax or ncols > imax)
    throw std::overflow_error(
        "CSR dimensions " + std::to_string(nrows) + " x " +
        std::to_string(ncols) + " exceed the range of the " +
        std::to_string(8 * sizeof(index_t)) +
        "-bit sparse index type; use 64-bit indices (index_t = int64_t)");
}

/**
 * @brief Returns rowptr[row + 1] = prev + nrow, throwing instead of wrapping
 * when the running nonzero count no longer fits in index_t.
 *
 * @param[in] prev:  rowptr[row], i.e. nonzeros stored before this row.
 * @param[in] nrow:  nonzeros in this row.
 * @param[in] row:   Row index (for the error message).
 * @param[in] nrows: Total number of rows (for the error message).
 */
template <typename index_t>
index_t checked_rowptr_next(index_t prev, size_t nrow, size_t row,
                            size_t nrows) {
  constexpr size_t imax = size_t(std::numeric_limits<index_t>::max());
  const size_t next = size_t(prev) + nrow;
  if(next > imax)
    throw std::overflow_error(
        "CSR nonzero count overflows the " +
        std::to_string(8 * sizeof(index_t)) + "-bit sparse index type at row " +
        std::to_string(row) + " of " + std::to_string(nrows) + " (" +
        std::to_string(next) +
        " nonzeros so far); use 64-bit indices (index_t = int64_t)");
  return index_t(next);
}

}  // namespace macis
