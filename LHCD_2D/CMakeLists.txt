cmake_minimum_required(VERSION 3.19)
project(LHCD_2D LANGUAGES CXX)

# ===========================================================================
#  LHCD_2D -- the whole pipeline as one dependency graph.
#
#    LHCD_2D (exe)                             <- LHCD_2D.cpp, solver deck
#    output.h5 + snapshots   [target: run]     <- LHCD_2D, input_solver.txt
#    figures                 [tools/run.sh]    <- output.h5 (cache + parallel plotters)
#
#  Identical structure to ICRF_2D/CMakeLists.txt minus the table-generation
#  stage: LHCD's quasilinear coefficients are analytic (see
#  LHCD_2D/plot/coefficients.py), so there is nothing to pregenerate.
#
#  CMake is the single owner of staleness for every stage: each edge above is
#  an OUTPUT/DEPENDS rule, so `cmake --build build --target run` re-runs
#  only the stages whose inputs actually changed.  Re-running with nothing
#  changed does nothing.
#
#       cmake -S . -B build      (with the ASGarD venv active)
#       cmake --build build                        # compile
#       cmake --build build --target run           # solve      (if stale)
#
#  tools/run.sh drives the pipeline: it owns the environment (venv, core
#  counts), invokes the commands above, then runs the plotting stage itself.
# ===========================================================================

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "build type: Debug, Release, RelWithDebInfo, or MinSizeRel" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

set(LHCD_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INPUT_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/input_data")
set(OUTPUT_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output_data")
set(SOLVER_DECK "${INPUT_DATA_DIR}/input_solver.txt")

# ---------------------------------------------------------------------------
#  OpenMP.  Homebrew's libomp is keg-only (not linked into /opt/homebrew/lib),
#  so hand find_package(OpenMP) its prefix; Linux never enters the branch.
#  LHCD itself does not use OpenMP, but asgard's config requires it, so this
#  must precede find_package(asgard).
# ---------------------------------------------------------------------------
foreach(_libomp /opt/homebrew/opt/libomp /usr/local/opt/libomp)
    if(APPLE AND EXISTS "${_libomp}" AND NOT DEFINED OpenMP_ROOT)
        set(OpenMP_ROOT "${_libomp}")
    endif()
endforeach()

find_package(asgard 0.9.0 REQUIRED)

add_executable(LHCD_2D "${LHCD_SOURCE_DIR}/LHCD_2D.cpp")
target_compile_features(LHCD_2D PRIVATE cxx_std_17)
target_link_libraries(LHCD_2D PRIVATE asgard::asgard)
target_compile_options(LHCD_2D PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-O3>)

# Deliberately NO OBJECT_DEPENDS on the deck: it is a runtime input, tracked
# by the solve rule below, not a compile input.  (The old recompile-on-deck
# hack predates the solve being in the graph; keeping it would recompile the
# binary on every deck edit, and the newer binary would then force a re-solve
# even after the deck was reverted.)

# ---------------------------------------------------------------------------
#  stage 2 : the solve  (target: run)
#
#  OUTPUT/DEPENDS makes this incremental: the solver re-runs only when
#  output.h5 is older than the LHCD_2D binary or the solver deck.  Editing
#  plotting code, or nothing at all, re-runs nothing.
#
#  The pre-clean removes every stale .h5 (snapshots included) so a leftover
#  frame from a previous run can never leak into this run's movies -- and it
#  runs only on the path that actually re-solves.
# ---------------------------------------------------------------------------
set(LHCD_SOLVER_OMP_THREADS 2 CACHE STRING
    "OpenMP threads for the ASGarD solve (2 measured fastest)")

# The solve runs FROM output_data (WORKING_DIRECTORY), and make cds
# there before any COMMAND executes -- so the directory must exist at
# configure time, not be created by a COMMAND.
file(MAKE_DIRECTORY ${OUTPUT_DATA_DIR})

set(OUTPUT_FILE "${OUTPUT_DATA_DIR}/output.h5")

if(APPLE)
    # KMP_BLOCKTIME=infinite prevents a libomp thread-suspend segfault
    # (3/3 reproducible without it, 0/3 with it; OMP_DYNAMIC made no
    # difference in isolation).  Linux OpenMP runtimes need neither.
    set(LHCD_SOLVER_ENV
        OMP_NUM_THREADS=${LHCD_SOLVER_OMP_THREADS}
        KMP_BLOCKTIME=infinite)
else()
    set(LHCD_SOLVER_ENV OMP_NUM_THREADS=${LHCD_SOLVER_OMP_THREADS})
endif()

add_custom_command(
    OUTPUT  ${OUTPUT_FILE}
    COMMAND ${CMAKE_COMMAND} -E env ${LHCD_SOLVER_ENV}
            $<TARGET_FILE:LHCD_2D> -if ${SOLVER_DECK} -of ${OUTPUT_FILE}
    # From output_data so the solver's relative "movie/" snapshot path (from
    # input_solver.txt's movie_dir) resolves.
    WORKING_DIRECTORY ${OUTPUT_DATA_DIR}
    DEPENDS LHCD_2D ${SOLVER_DECK}
    COMMENT "Running LHCD_2D  (-if ${SOLVER_DECK})"
    USES_TERMINAL
    VERBATIM)

add_custom_target(run DEPENDS ${OUTPUT_FILE})

