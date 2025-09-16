# Test executable for LAPACK matrix convention
add_executable( test_lapack_convention test_lapack_convention.cxx )

# Link against the macis library for LAPACK dependencies
target_link_libraries( test_lapack_convention macis )