#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "sparsexx::sparsexx" for configuration "Debug"
set_property(TARGET sparsexx::sparsexx APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(sparsexx::sparsexx PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libsparsexx.so"
  IMPORTED_SONAME_DEBUG "libsparsexx.so"
  )

list(APPEND _cmake_import_check_targets sparsexx::sparsexx )
list(APPEND _cmake_import_check_files_for_sparsexx::sparsexx "${_IMPORT_PREFIX}/lib/libsparsexx.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
