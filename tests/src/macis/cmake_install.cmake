# Install script for directory: /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/src/macis

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/spdlog-build/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/mdspan-build/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmacis.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmacis.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmacis.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/src/macis/libmacis.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmacis.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmacis.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmacis.so"
         OLD_RPATH "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/src/sparsexx:/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/spdlog-build:/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/lapackpp-build:/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/blaspp-build:/usr/lib/x86_64-linux-gnu/openmpi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmacis.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparsexx.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparsexx.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparsexx.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/src/sparsexx/libsparsexx.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparsexx.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparsexx.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparsexx.so"
         OLD_RPATH "/usr/lib/x86_64-linux-gnu/openmpi/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsparsexx.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so.1.12.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so.1.12"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/spdlog-build/libspdlog.so.1.12.0"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/spdlog-build/libspdlog.so.1.12"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so.1.12.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libspdlog.so.1.12"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/spdlog-build/libspdlog.so")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/include")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local" TYPE DIRECTORY FILES "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/include" FILES_MATCHING REGEX "/[^/]*\\.hpp$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/macis" TYPE FILE FILES "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/include/macis/macis_config.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/macis/macis-targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/macis/macis-targets.cmake"
         "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/src/macis/CMakeFiles/Export/7bb25f7bad96ebd129324292344148a0/macis-targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/macis/macis-targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/macis/macis-targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/macis" TYPE FILE FILES "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/src/macis/CMakeFiles/Export/7bb25f7bad96ebd129324292344148a0/macis-targets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/macis" TYPE FILE FILES "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/src/macis/CMakeFiles/Export/7bb25f7bad96ebd129324292344148a0/macis-targets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/macis" TYPE FILE FILES
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/src/macis/macis-config.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/src/macis/macis-config-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/macis/linalg-cmake-modules" TYPE FILE FILES
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindBLAS.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindBLIS.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindIBMESSL.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindIntelMKL.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindLAPACK.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindOpenBLAS.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindReferenceBLAS.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindReferenceLAPACK.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindReferenceScaLAPACK.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindScaLAPACK.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindILP64.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindTBB.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/FindStandardFortran.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/LinAlgModulesMacros.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/LICENSE.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/macis/linalg-cmake-modules/util" TYPE FILE FILES
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/util/BLASUtilities.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/util/blis_int_size.c"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/util/func_check.c"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/util/get_mpi_vendor.c"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/util/ilp64_checker.c"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/util/lapack_ilp64_checker.c"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/util/openblas_int_size.c"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/util/CommonFunctions.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/util/IntrospectMPI.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/util/IntrospectOpenMP.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/util/LAPACKUtilities.cmake"
    "/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src/util/ScaLAPACKUtilities.cmake"
    )
endif()

