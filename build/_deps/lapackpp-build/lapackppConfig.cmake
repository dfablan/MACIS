cmake_minimum_required( VERSION 3.15 )

set( lapackpp_use_cuda   "false" )
set( lapackpp_use_hip    "false" )
set( lapackpp_use_sycl   "false" )

include( CMakeFindDependencyMacro )

find_dependency( blaspp )

if (lapackpp_use_hip)
    find_dependency( rocblas   )
    find_dependency( rocsolver )
endif()

# Export variables.
set( lapackpp_defines   "-DLAPACK_VERSION=31000" )
set( lapackpp_libraries "/usr/lib/x86_64-linux-gnu/liblapack.so;/usr/lib/x86_64-linux-gnu/libblas.so;blaspp" )

include("${CMAKE_CURRENT_LIST_DIR}/lapackppTargets.cmake")
