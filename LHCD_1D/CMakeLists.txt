cmake_minimum_required(VERSION 3.19)
project(LHCD_1D LANGUAGES CXX)

# ===========================================================================
#  LHCD_1D -- the whole pipeline as one dependency graph.
#
#    LHCD_1D (exe)                             <- LHCD_1D.cpp
#    output.h5               [target: run]     <- LHCD_1D, input_solver.txt
#    figures                 [tools/run.sh]    <- output.h5 (cache + parallel plotters)
#
#  The 1-D parallel-velocity quasilinear diffusion problem: same structure as
#  LHCD_2D/CMakeLists.txt minus snapshot cleanup (the solver stores its
#  time snapshots as aux fields inside the single output.h5, so there is
#  nothing separate to clean).
#
#  CMake owns staleness for the compile and solve.  Plotting lives in
#  tools/run.sh: one shared-cache build, then every plotter in parallel.
#
#       cmake -S . -B build      (with the ASGarD venv active)
#       cmake --build build                        # compile
#       cmake --build build --target run           # solve      (if stale)
# ===========================================================================

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "build type: Debug, Release, RelWithDebInfo, or MinSizeRel" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

set(LHCD1D_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INPUT_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/input_data")
set(OUTPUT_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output_data")
set(SOLVER_DECK "${INPUT_DATA_DIR}/input_solver.txt")

# ---------------------------------------------------------------------------
#  OpenMP.  Homebrew's libomp is keg-only (not linked into /opt/homebrew/lib),
#  so hand find_package(OpenMP) its prefix; Linux never enters the branch.
#  The 1-D solver itself does not use OpenMP, but asgard's config requires it,
#  so this must precede find_package(asgard).
# ---------------------------------------------------------------------------
foreach(_libomp /opt/homebrew/opt/libomp /usr/local/opt/libomp)
    if(APPLE AND EXISTS "${_libomp}" AND NOT DEFINED OpenMP_ROOT)
        set(OpenMP_ROOT "${_libomp}")
    endif()
endforeach()

find_package(asgard 0.9.0 REQUIRED)

add_executable(LHCD_1D "${LHCD1D_SOURCE_DIR}/LHCD_1D.cpp")
target_compile_features(LHCD_1D PRIVATE cxx_std_17)
target_link_libraries(LHCD_1D PRIVATE asgard::asgard)
target_compile_options(LHCD_1D PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-O3>)

# ---------------------------------------------------------------------------
#  stage 2 : the solve  (target: run)
#
#  OUTPUT/DEPENDS makes this incremental: the solver re-runs only when
#  output.h5 is older than the LHCD_1D binary or the solver deck.
# ---------------------------------------------------------------------------
# The solve runs FROM output_data (WORKING_DIRECTORY), and make cds
# there before any COMMAND executes -- so the directory must exist at
# configure time, not be created by a COMMAND.
file(MAKE_DIRECTORY ${OUTPUT_DATA_DIR})

set(OUTPUT_FILE "${OUTPUT_DATA_DIR}/output.h5")

set(LHCD_1D_SOLVER_OMP_THREADS 2 CACHE STRING
    "OpenMP threads for the ASGarD solve")

if(APPLE)
    # KMP_BLOCKTIME=infinite prevents a libomp thread-suspend segfault
    # (3/3 reproducible without it, 0/3 with it; OMP_DYNAMIC made no
    # difference in isolation).  Linux OpenMP runtimes need neither.
    set(LHCD_1D_SOLVER_ENV
        OMP_NUM_THREADS=${LHCD_1D_SOLVER_OMP_THREADS}
        KMP_BLOCKTIME=infinite)
else()
    set(LHCD_1D_SOLVER_ENV OMP_NUM_THREADS=${LHCD_1D_SOLVER_OMP_THREADS})
endif()

# The solve's real products are output.h5 AND the movie snapshot series, but
# the snapshots have no static filenames CMake could declare (they depend on
# step numbers), and with this generator only the FIRST output of a rule
# carries the recipe -- extra outputs get a no-op recheck.  So the rule's one
# declared OUTPUT is a sentinel touched only after the solver finishes: it
# represents "solve and snapshot series complete".  Deleting movie/ removes it
# and forces a re-solve; an interrupted solve never reaches the touch.
# The sentinel lives inside the directory the solver cleans at startup;
# it survives only because the solver deletes snapshot_*.h5 and nothing
# else.  If that cleanup pattern ever broadens, this name must move.
set(MOVIE_SENTINEL "${OUTPUT_DATA_DIR}/movie/.solve-complete")

add_custom_command(
    OUTPUT  ${MOVIE_SENTINEL}
    COMMAND ${CMAKE_COMMAND} -E env ${LHCD_1D_SOLVER_ENV}
            $<TARGET_FILE:LHCD_1D> -if ${SOLVER_DECK} -of ${OUTPUT_FILE}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DATA_DIR}/movie
    COMMAND ${CMAKE_COMMAND} -E touch ${MOVIE_SENTINEL}
    # From output_data so the solver's relative "movie/" snapshot path (from
    # input_solver.txt's movie_dir) resolves there.
    WORKING_DIRECTORY ${OUTPUT_DATA_DIR}
    DEPENDS LHCD_1D ${SOLVER_DECK}
    COMMENT "Running LHCD_1D  (-if ${SOLVER_DECK})"
    USES_TERMINAL
    VERBATIM)

add_custom_target(run DEPENDS ${MOVIE_SENTINEL})

