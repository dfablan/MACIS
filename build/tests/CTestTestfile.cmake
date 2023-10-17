# CMake generated Testfile for 
# Source directory: /home/diego/Documents/Trabajo/SISSA/Projects/CI_solver/MACIS_mod/tests
# Build directory: /home/diego/Documents/Trabajo/SISSA/Projects/CI_solver/MACIS_mod/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(MACIS_SERIAL_TEST "/home/diego/Documents/Trabajo/SISSA/Projects/CI_solver/MACIS_mod/build/tests/macis_test")
set_tests_properties(MACIS_SERIAL_TEST PROPERTIES  _BACKTRACE_TRIPLES "/home/diego/Documents/Trabajo/SISSA/Projects/CI_solver/MACIS_mod/tests/CMakeLists.txt;58;add_test;/home/diego/Documents/Trabajo/SISSA/Projects/CI_solver/MACIS_mod/tests/CMakeLists.txt;0;")
add_test(MACIS_MPI_TEST "/usr/bin/mpiexec" "-n" "2" "/home/diego/Documents/Trabajo/SISSA/Projects/CI_solver/MACIS_mod/build/tests/macis_test")
set_tests_properties(MACIS_MPI_TEST PROPERTIES  _BACKTRACE_TRIPLES "/home/diego/Documents/Trabajo/SISSA/Projects/CI_solver/MACIS_mod/tests/CMakeLists.txt;60;add_test;/home/diego/Documents/Trabajo/SISSA/Projects/CI_solver/MACIS_mod/tests/CMakeLists.txt;0;")
subdirs("../_deps/catch2-build")
