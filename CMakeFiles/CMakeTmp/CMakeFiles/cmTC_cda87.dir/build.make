# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /cygdrive/c/Users/t-idkess/.CLion2017.2/system/cygwin_cmake/bin/cmake.exe

# The command to remove a file.
RM = /cygdrive/c/Users/t-idkess/.CLion2017.2/system/cygwin_cmake/bin/cmake.exe -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/CMakeTmp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/CMakeTmp

# Include any dependencies generated for this target.
include CMakeFiles/cmTC_cda87.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cmTC_cda87.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cmTC_cda87.dir/flags.make

CMakeFiles/cmTC_cda87.dir/feature_tests.c.o: CMakeFiles/cmTC_cda87.dir/flags.make
CMakeFiles/cmTC_cda87.dir/feature_tests.c.o: /cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/feature_tests.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/CMakeTmp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/cmTC_cda87.dir/feature_tests.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cmTC_cda87.dir/feature_tests.c.o   -c /cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/feature_tests.c

CMakeFiles/cmTC_cda87.dir/feature_tests.c.i: cmake_force
	@echo "Preprocessing C source to CMakeFiles/cmTC_cda87.dir/feature_tests.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/feature_tests.c > CMakeFiles/cmTC_cda87.dir/feature_tests.c.i

CMakeFiles/cmTC_cda87.dir/feature_tests.c.s: cmake_force
	@echo "Compiling C source to assembly CMakeFiles/cmTC_cda87.dir/feature_tests.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/feature_tests.c -o CMakeFiles/cmTC_cda87.dir/feature_tests.c.s

CMakeFiles/cmTC_cda87.dir/feature_tests.c.o.requires:

.PHONY : CMakeFiles/cmTC_cda87.dir/feature_tests.c.o.requires

CMakeFiles/cmTC_cda87.dir/feature_tests.c.o.provides: CMakeFiles/cmTC_cda87.dir/feature_tests.c.o.requires
	$(MAKE) -f CMakeFiles/cmTC_cda87.dir/build.make CMakeFiles/cmTC_cda87.dir/feature_tests.c.o.provides.build
.PHONY : CMakeFiles/cmTC_cda87.dir/feature_tests.c.o.provides

CMakeFiles/cmTC_cda87.dir/feature_tests.c.o.provides.build: CMakeFiles/cmTC_cda87.dir/feature_tests.c.o


# Object files for target cmTC_cda87
cmTC_cda87_OBJECTS = \
"CMakeFiles/cmTC_cda87.dir/feature_tests.c.o"

# External object files for target cmTC_cda87
cmTC_cda87_EXTERNAL_OBJECTS =

cmTC_cda87.exe: CMakeFiles/cmTC_cda87.dir/feature_tests.c.o
cmTC_cda87.exe: CMakeFiles/cmTC_cda87.dir/build.make
cmTC_cda87.exe: CMakeFiles/cmTC_cda87.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/CMakeTmp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable cmTC_cda87.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cmTC_cda87.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cmTC_cda87.dir/build: cmTC_cda87.exe

.PHONY : CMakeFiles/cmTC_cda87.dir/build

CMakeFiles/cmTC_cda87.dir/requires: CMakeFiles/cmTC_cda87.dir/feature_tests.c.o.requires

.PHONY : CMakeFiles/cmTC_cda87.dir/requires

CMakeFiles/cmTC_cda87.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cmTC_cda87.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cmTC_cda87.dir/clean

CMakeFiles/cmTC_cda87.dir/depend:
	cd /cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/CMakeTmp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/CMakeTmp /cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/CMakeTmp /cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/CMakeTmp /cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/CMakeTmp /cygdrive/c/Users/t-idkess/Documents/KDTree/CMakeFiles/CMakeTmp/CMakeFiles/cmTC_cda87.dir/DependInfo.cmake
.PHONY : CMakeFiles/cmTC_cda87.dir/depend

