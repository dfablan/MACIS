/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include "catch2/catch.hpp"

#include <macis/util/mpi.hpp>
#ifdef MACIS_ENABLE_MPI

#define ROOT_ONLY(comm) \
  int mpi_rank; \
  MPI_Comm_rank(comm,&mpi_rank); \
  if(mpi_rank > 0 ) return;

#else

#define ROOT_ONLY(comm)

#endif


#define REF_DATA_PREFIX "/home/diego/Documents/Trabajo/SISSA/Projects/CI_solver/MACIS_mod/tests/ref_data"
const std::string water_ccpvdz_fcidump = 
  REF_DATA_PREFIX "/h2o.ccpvdz.fci.dat";
const std::string water_ccpvdz_rdms_fname = 
  REF_DATA_PREFIX "/h2o.ccpvdz.cas88.rdms.bin";
const std::string water_ccpvdz_rowptr_fname = 
  REF_DATA_PREFIX "/h2o.ccpvdz.cisd.rowptr.bin";
const std::string water_ccpvdz_colind_fname = 
  REF_DATA_PREFIX "/h2o.ccpvdz.cisd.colind.bin";
const std::string water_ccpvdz_nzval_fname  = 
  REF_DATA_PREFIX "/h2o.ccpvdz.cisd.nzval.bin";
const std::string ch4_wfn_fname =
  REF_DATA_PREFIX "/ch4.wfn.dat";
const std::string hubbard10_fcidump =
  REF_DATA_PREFIX "/hubbard10.fci.dat";