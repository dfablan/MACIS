# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-src"
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build"
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-subbuild/testsweeper-populate-prefix"
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-subbuild/testsweeper-populate-prefix/tmp"
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-subbuild/testsweeper-populate-prefix/src/testsweeper-populate-stamp"
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-subbuild/testsweeper-populate-prefix/src"
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-subbuild/testsweeper-populate-prefix/src/testsweeper-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-subbuild/testsweeper-populate-prefix/src/testsweeper-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-subbuild/testsweeper-populate-prefix/src/testsweeper-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
