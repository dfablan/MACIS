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

# Utility rule file for ContinuousTest.

# Include any custom commands dependencies for this target.
include src/lobpcgxx/CMakeFiles/ContinuousTest.dir/compiler_depend.make

# Include the progress variables for this target.
include src/lobpcgxx/CMakeFiles/ContinuousTest.dir/progress.make

src/lobpcgxx/CMakeFiles/ContinuousTest:
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/src/lobpcgxx && /usr/local/bin/ctest -D ContinuousTest

ContinuousTest: src/lobpcgxx/CMakeFiles/ContinuousTest
ContinuousTest: src/lobpcgxx/CMakeFiles/ContinuousTest.dir/build.make
.PHONY : ContinuousTest

# Rule to build all files generated by this target.
src/lobpcgxx/CMakeFiles/ContinuousTest.dir/build: ContinuousTest
.PHONY : src/lobpcgxx/CMakeFiles/ContinuousTest.dir/build

src/lobpcgxx/CMakeFiles/ContinuousTest.dir/clean:
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/src/lobpcgxx && $(CMAKE_COMMAND) -P CMakeFiles/ContinuousTest.dir/cmake_clean.cmake
.PHONY : src/lobpcgxx/CMakeFiles/ContinuousTest.dir/clean

src/lobpcgxx/CMakeFiles/ContinuousTest.dir/depend:
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/src/lobpcgxx /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/src/lobpcgxx /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/src/lobpcgxx/CMakeFiles/ContinuousTest.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/lobpcgxx/CMakeFiles/ContinuousTest.dir/depend

