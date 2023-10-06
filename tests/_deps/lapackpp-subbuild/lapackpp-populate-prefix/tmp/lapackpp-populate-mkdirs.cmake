# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/lapackpp-src"
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/lapackpp-build"
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/lapackpp-subbuild/lapackpp-populate-prefix"
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/lapackpp-subbuild/lapackpp-populate-prefix/tmp"
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/lapackpp-subbuild/lapackpp-populate-prefix/src/lapackpp-populate-stamp"
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/lapackpp-subbuild/lapackpp-populate-prefix/src"
  "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/lapackpp-subbuild/lapackpp-populate-prefix/src/lapackpp-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/lapackpp-subbuild/lapackpp-populate-prefix/src/lapackpp-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/lapackpp-subbuild/lapackpp-populate-prefix/src/lapackpp-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
