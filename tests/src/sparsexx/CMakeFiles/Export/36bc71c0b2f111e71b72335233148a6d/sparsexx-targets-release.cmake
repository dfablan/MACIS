#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "sparsexx::sparsexx" for configuration "Release"
set_property(TARGET sparsexx::sparsexx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sparsexx::sparsexx PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsparsexx.so"
  IMPORTED_SONAME_RELEASE "libsparsexx.so"
  )

list(APPEND _cmake_import_check_targets sparsexx::sparsexx )
list(APPEND _cmake_import_check_files_for_sparsexx::sparsexx "${_IMPORT_PREFIX}/lib/libsparsexx.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
