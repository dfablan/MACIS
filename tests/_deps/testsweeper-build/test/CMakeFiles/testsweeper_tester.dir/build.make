# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests

# Include any dependencies generated for this target.
include _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/flags.make

_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test.cc.o: _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/flags.make
_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test.cc.o: _deps/testsweeper-src/test/test.cc
_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test.cc.o: _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test.cc.o"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test.cc.o -MF CMakeFiles/testsweeper_tester.dir/test.cc.o.d -o CMakeFiles/testsweeper_tester.dir/test.cc.o -c /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-src/test/test.cc

_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/testsweeper_tester.dir/test.cc.i"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-src/test/test.cc > CMakeFiles/testsweeper_tester.dir/test.cc.i

_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/testsweeper_tester.dir/test.cc.s"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-src/test/test.cc -o CMakeFiles/testsweeper_tester.dir/test.cc.s

_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test_sort.cc.o: _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/flags.make
_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test_sort.cc.o: _deps/testsweeper-src/test/test_sort.cc
_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test_sort.cc.o: _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test_sort.cc.o"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test_sort.cc.o -MF CMakeFiles/testsweeper_tester.dir/test_sort.cc.o.d -o CMakeFiles/testsweeper_tester.dir/test_sort.cc.o -c /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-src/test/test_sort.cc

_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test_sort.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/testsweeper_tester.dir/test_sort.cc.i"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-src/test/test_sort.cc > CMakeFiles/testsweeper_tester.dir/test_sort.cc.i

_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test_sort.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/testsweeper_tester.dir/test_sort.cc.s"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-src/test/test_sort.cc -o CMakeFiles/testsweeper_tester.dir/test_sort.cc.s

# Object files for target testsweeper_tester
testsweeper_tester_OBJECTS = \
"CMakeFiles/testsweeper_tester.dir/test.cc.o" \
"CMakeFiles/testsweeper_tester.dir/test_sort.cc.o"

# External object files for target testsweeper_tester
testsweeper_tester_EXTERNAL_OBJECTS =

_deps/testsweeper-build/test/testsweeper_tester: _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test.cc.o
_deps/testsweeper-build/test/testsweeper_tester: _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/test_sort.cc.o
_deps/testsweeper-build/test/testsweeper_tester: _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/build.make
_deps/testsweeper-build/test/testsweeper_tester: _deps/testsweeper-build/libtestsweeper.so
_deps/testsweeper-build/test/testsweeper_tester: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
_deps/testsweeper-build/test/testsweeper_tester: /usr/lib/x86_64-linux-gnu/libpthread.so
_deps/testsweeper-build/test/testsweeper_tester: _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable testsweeper_tester"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testsweeper_tester.dir/link.txt --verbose=$(VERBOSE)
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build/test && cp -a /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-src/test/run_tests.py /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-src/test/ref /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build/test/

# Rule to build all files generated by this target.
_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/build: _deps/testsweeper-build/test/testsweeper_tester
.PHONY : _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/build

_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/clean:
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build/test && $(CMAKE_COMMAND) -P CMakeFiles/testsweeper_tester.dir/cmake_clean.cmake
.PHONY : _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/clean

_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/depend:
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-src/test /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build/test /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : _deps/testsweeper-build/test/CMakeFiles/testsweeper_tester.dir/depend
