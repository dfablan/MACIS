# Test executable for LAPACK matrix convention
add_executable( test_lapack_convention test_lapack_convention.cxx )
add_executable( run_asci_impsolv_dop run_asci_impsolv_dop.cxx ../tests/ini_input.cxx )
add_executable( run_asci_impsolv_mu_vs_n run_asci_impsolv_mu_vs_n.cxx ../tests/ini_input.cxx )

# Link against the macis library for LAPACK dependencies
target_link_libraries( test_lapack_convention macis )
target_link_libraries( run_asci_impsolv_dop macis )
target_link_libraries( run_asci_impsolv_mu_vs_n macis )