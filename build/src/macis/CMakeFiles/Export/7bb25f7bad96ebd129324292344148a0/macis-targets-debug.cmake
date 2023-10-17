#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "macis::macis" for configuration "Debug"
set_property(TARGET macis::macis APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(macis::macis PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libmacis.so"
  IMPORTED_SONAME_DEBUG "libmacis.so"
  )

list(APPEND _cmake_import_check_targets macis::macis )
list(APPEND _cmake_import_check_files_for_macis::macis "${_IMPORT_PREFIX}/lib/libmacis.so" )

# Import target "macis::sparsexx" for configuration "Debug"
set_property(TARGET macis::sparsexx APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(macis::sparsexx PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libsparsexx.so"
  IMPORTED_SONAME_DEBUG "libsparsexx.so"
  )

list(APPEND _cmake_import_check_targets macis::sparsexx )
list(APPEND _cmake_import_check_files_for_macis::sparsexx "${_IMPORT_PREFIX}/lib/libsparsexx.so" )

# Import target "macis::spdlog" for configuration "Debug"
set_property(TARGET macis::spdlog APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(macis::spdlog PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libspdlogd.so.1.12.0"
  IMPORTED_SONAME_DEBUG "libspdlogd.so.1.12"
  )

list(APPEND _cmake_import_check_targets macis::spdlog )
list(APPEND _cmake_import_check_files_for_macis::spdlog "${_IMPORT_PREFIX}/lib/libspdlogd.so.1.12.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
