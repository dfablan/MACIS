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
CMAKE_SOURCE_DIR = /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild

# Utility rule file for linalg-cmake-modules-populate.

# Include any custom commands dependencies for this target.
include CMakeFiles/linalg-cmake-modules-populate.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/linalg-cmake-modules-populate.dir/progress.make

CMakeFiles/linalg-cmake-modules-populate: CMakeFiles/linalg-cmake-modules-populate-complete

CMakeFiles/linalg-cmake-modules-populate-complete: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-install
CMakeFiles/linalg-cmake-modules-populate-complete: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-mkdir
CMakeFiles/linalg-cmake-modules-populate-complete: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-download
CMakeFiles/linalg-cmake-modules-populate-complete: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-update
CMakeFiles/linalg-cmake-modules-populate-complete: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-patch
CMakeFiles/linalg-cmake-modules-populate-complete: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-configure
CMakeFiles/linalg-cmake-modules-populate-complete: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-build
CMakeFiles/linalg-cmake-modules-populate-complete: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-install
CMakeFiles/linalg-cmake-modules-populate-complete: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-test
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'linalg-cmake-modules-populate'"
	/usr/local/bin/cmake -E make_directory /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/CMakeFiles
	/usr/local/bin/cmake -E touch /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/CMakeFiles/linalg-cmake-modules-populate-complete
	/usr/local/bin/cmake -E touch /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-done

linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-update:
.PHONY : linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-update

linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-build: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No build step for 'linalg-cmake-modules-populate'"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-build && /usr/local/bin/cmake -E echo_append
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-build && /usr/local/bin/cmake -E touch /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-build

linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-configure: linalg-cmake-modules-populate-prefix/tmp/linalg-cmake-modules-populate-cfgcmd.txt
linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-configure: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "No configure step for 'linalg-cmake-modules-populate'"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-build && /usr/local/bin/cmake -E echo_append
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-build && /usr/local/bin/cmake -E touch /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-configure

linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-download: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-gitinfo.txt
linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-download: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'linalg-cmake-modules-populate'"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps && /usr/local/bin/cmake -P /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/linalg-cmake-modules-populate-prefix/tmp/linalg-cmake-modules-populate-gitclone.cmake
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps && /usr/local/bin/cmake -E touch /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-download

linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-install: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-build
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "No install step for 'linalg-cmake-modules-populate'"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-build && /usr/local/bin/cmake -E echo_append
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-build && /usr/local/bin/cmake -E touch /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-install

linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Creating directories for 'linalg-cmake-modules-populate'"
	/usr/local/bin/cmake -Dcfgdir= -P /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/linalg-cmake-modules-populate-prefix/tmp/linalg-cmake-modules-populate-mkdirs.cmake
	/usr/local/bin/cmake -E touch /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-mkdir

linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-patch: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-patch-info.txt
linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-patch: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-update
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No patch step for 'linalg-cmake-modules-populate'"
	/usr/local/bin/cmake -E echo_append
	/usr/local/bin/cmake -E touch /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-patch

linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-update:
.PHONY : linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-update

linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-test: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-install
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No test step for 'linalg-cmake-modules-populate'"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-build && /usr/local/bin/cmake -E echo_append
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-build && /usr/local/bin/cmake -E touch /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-test

linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-update: linalg-cmake-modules-populate-prefix/tmp/linalg-cmake-modules-populate-gitupdate.cmake
linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-update: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-update-info.txt
linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-update: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-download
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Performing update step for 'linalg-cmake-modules-populate'"
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-src && /usr/local/bin/cmake -Dcan_fetch=YES -P /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/linalg-cmake-modules-populate-prefix/tmp/linalg-cmake-modules-populate-gitupdate.cmake

linalg-cmake-modules-populate: CMakeFiles/linalg-cmake-modules-populate
linalg-cmake-modules-populate: CMakeFiles/linalg-cmake-modules-populate-complete
linalg-cmake-modules-populate: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-build
linalg-cmake-modules-populate: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-configure
linalg-cmake-modules-populate: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-download
linalg-cmake-modules-populate: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-install
linalg-cmake-modules-populate: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-mkdir
linalg-cmake-modules-populate: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-patch
linalg-cmake-modules-populate: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-test
linalg-cmake-modules-populate: linalg-cmake-modules-populate-prefix/src/linalg-cmake-modules-populate-stamp/linalg-cmake-modules-populate-update
linalg-cmake-modules-populate: CMakeFiles/linalg-cmake-modules-populate.dir/build.make
.PHONY : linalg-cmake-modules-populate

# Rule to build all files generated by this target.
CMakeFiles/linalg-cmake-modules-populate.dir/build: linalg-cmake-modules-populate
.PHONY : CMakeFiles/linalg-cmake-modules-populate.dir/build

CMakeFiles/linalg-cmake-modules-populate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/linalg-cmake-modules-populate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/linalg-cmake-modules-populate.dir/clean

CMakeFiles/linalg-cmake-modules-populate.dir/depend:
	cd /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild /home/diego/Documents/SISSA/Projects/CI_solver/MACIS_build/tests/_deps/linalg-cmake-modules-subbuild/CMakeFiles/linalg-cmake-modules-populate.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/linalg-cmake-modules-populate.dir/depend
