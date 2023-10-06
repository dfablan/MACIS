#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lapackpp" for configuration "Release"
set_property(TARGET lapackpp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(lapackpp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liblapackpp.so"
  IMPORTED_SONAME_RELEASE "liblapackpp.so"
  )

list(APPEND _cmake_import_check_targets lapackpp )
list(APPEND _cmake_import_check_files_for_lapackpp "${_IMPORT_PREFIX}/lib/liblapackpp.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
