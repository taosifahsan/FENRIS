cmake_minimum_required(VERSION 3.19)
project(ICRF_2D LANGUAGES CXX)

# ===========================================================================
#  ICRF_2D -- the whole pipeline as one dependency graph.
#
#    build_tables (exe)                        <- build_tables.cpp, GL.hpp
#    tables/*.bin                              <- build_tables, input_build.txt
#    ICRF_2D (exe)                             <- ICRF_2D.cpp, tables, decks
#    output.h5 + snapshots   [target: run]     <- ICRF_2D, input_solver.txt,
#                                                 tables/*.bin
#    figures                 [target: plots]   <- output.h5
#
#  CMake is the single owner of staleness for every stage: each edge above is
#  an OUTPUT/DEPENDS rule, so `cmake --build build --target plots` re-runs
#  only the stages whose inputs actually changed.  Re-running with nothing
#  changed does nothing, all the way up the chain.
#
#       cmake -S . -B build      (with the ASGarD venv active)
#       cmake --build build                        # compile (+ tables if stale)
#       cmake --build build --target run           # solve      (if stale)
#       cmake --build build --target plots         # solve+plot (if stale)
#
#  The tools/run_*.sh scripts are thin wrappers over exactly those commands:
#  they own the environment (modules, venv, core counts) and nothing else.
#
#  OMP_NUM_THREADS for table generation is inherited from the environment of
#  the `cmake --build` invocation; the solver's own thread count is the
#  ICRF_SOLVER_OMP_THREADS cache variable below.
# ===========================================================================

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "build type: Debug, Release, RelWithDebInfo, or MinSizeRel" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

set(ICRF_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INPUT_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/input_data")
set(OUTPUT_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output_data")
set(BUILD_DECK "${INPUT_DATA_DIR}/input_build.txt")
set(SOLVER_DECK "${INPUT_DATA_DIR}/input_solver.txt")

# ---------------------------------------------------------------------------
#  OpenMP.  Homebrew's libomp is keg-only (not linked into /opt/homebrew/lib),
#  so hand find_package(OpenMP) its prefix; Linux never enters the branch.
#  Must precede find_package(asgard), whose config requires OpenMP itself.
# ---------------------------------------------------------------------------
foreach(_libomp /opt/homebrew/opt/libomp /usr/local/opt/libomp)
    if(APPLE AND EXISTS "${_libomp}" AND NOT DEFINED OpenMP_ROOT)
        set(OpenMP_ROOT "${_libomp}")
    endif()
endforeach()

find_package(OpenMP QUIET COMPONENTS CXX)

find_package(asgard REQUIRED)

# ---------------------------------------------------------------------------
#  stage 1 : build_tables
# ---------------------------------------------------------------------------
add_executable(build_tables "${ICRF_SOURCE_DIR}/build_tables.cpp")
target_compile_features(build_tables PRIVATE cxx_std_17)
target_include_directories(build_tables PRIVATE "${ICRF_SOURCE_DIR}")
target_link_libraries(build_tables PRIVATE asgard::asgard)
target_compile_options(build_tables PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-O3>)
if(OpenMP_CXX_FOUND)
    target_link_libraries(build_tables PRIVATE OpenMP::OpenMP_CXX)
endif()

set(TABLE_FILES
    ${OUTPUT_DATA_DIR}/tables/parameters.bin
    ${OUTPUT_DATA_DIR}/tables/L_tab.bin
    ${OUTPUT_DATA_DIR}/tables/I_tab.bin
    ${OUTPUT_DATA_DIR}/tables/res_tab.bin)

# OUTPUT/DEPENDS is what makes this incremental: the COMMAND runs only when an
# OUTPUT is older than the build_tables binary or the build deck.
add_custom_command(
    OUTPUT  ${TABLE_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DATA_DIR}/tables
    COMMAND $<TARGET_FILE:build_tables> -if ${BUILD_DECK}
    WORKING_DIRECTORY ${OUTPUT_DATA_DIR}
    DEPENDS build_tables ${BUILD_DECK}
    COMMENT "Generating coefficient tables (build_tables -> tables/*.bin)"
    VERBATIM)

add_custom_target(generate_tables DEPENDS ${TABLE_FILES})

# ---------------------------------------------------------------------------
#  stage 2 : ICRF_2D solver
# ---------------------------------------------------------------------------
add_executable(ICRF_2D "${ICRF_SOURCE_DIR}/ICRF_2D.cpp")
target_compile_features(ICRF_2D PRIVATE cxx_std_17)
target_include_directories(ICRF_2D PRIVATE "${ICRF_SOURCE_DIR}")
target_link_libraries(ICRF_2D PRIVATE asgard::asgard)

add_dependencies(ICRF_2D generate_tables)

# Deliberately NO OBJECT_DEPENDS on the decks or tables: they are runtime
# inputs, tracked by the solve rule below, not compile inputs.  (The old
# recompile-on-deck hack predates the solve being in the graph; keeping it
# would recompile the binary on every deck or table change, and the newer
# binary would then force a re-solve even after the deck was reverted.)

# ---------------------------------------------------------------------------
#  stage 3 : the solve  (target: run)
#
#  OUTPUT/DEPENDS makes this incremental exactly like the tables: the solver
#  re-runs only when output.h5 is older than the ICRF_2D binary, the solver
#  deck, or a regenerated table.  Editing plotting code, or nothing at all,
#  re-runs nothing.
#
#  The pre-clean removes every stale .h5 (snapshots included) so a leftover
#  frame from a previous run can never leak into this run's movies -- and it
#  runs only on the path that actually re-solves.
# ---------------------------------------------------------------------------
set(ICRF_SOLVER_OMP_THREADS 2 CACHE STRING
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
    set(ICRF_SOLVER_ENV
        OMP_NUM_THREADS=${ICRF_SOLVER_OMP_THREADS}
        KMP_BLOCKTIME=infinite)
else()
    set(ICRF_SOLVER_ENV OMP_NUM_THREADS=${ICRF_SOLVER_OMP_THREADS})
endif()

add_custom_command(
    OUTPUT  ${OUTPUT_FILE}
    COMMAND ${CMAKE_COMMAND}
            "-DOUTPUT_DATA_DIR=${OUTPUT_DATA_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/clean_solver_outputs.cmake"
    COMMAND ${CMAKE_COMMAND} -E env ${ICRF_SOLVER_ENV}
            $<TARGET_FILE:ICRF_2D> -if ${SOLVER_DECK} -of ${OUTPUT_FILE}
    # From output_data so the solver's relative "tables/" and "movie/" paths
    # (the latter from input_solver.txt's movie_dir) both resolve.
    WORKING_DIRECTORY ${OUTPUT_DATA_DIR}
    DEPENDS ICRF_2D ${SOLVER_DECK} ${TABLE_FILES}
    COMMENT "Running ICRF_2D  (-if ${SOLVER_DECK})"
    USES_TERMINAL
    VERBATIM)

add_custom_target(run DEPENDS ${OUTPUT_FILE})

# ---------------------------------------------------------------------------
#  stage 4 : the figures  (target: plots)
#
#  One target: show_all.py renders everything (statics + movies) into a fresh
#  timestamped directory under figures/.  The other plot/*.py entry points are
#  for running by hand, outside CMake.
#
#  Deliberately NOT incremental: a custom target with no OUTPUT file is always
#  considered out of date, so `--target plots` renders every time it is asked
#  to.  The DEPENDS on ${OUTPUT_FILE} still chains the graph -- a stale solve
#  (changed deck, solver, or tables) runs first -- but a fresh solve does not
#  suppress the render.  Skipping happens upstream, never here.
# ---------------------------------------------------------------------------
set(ICRF_PLOT_WORKERS 0 CACHE STRING
    "plotter worker processes (0 = every allocated core)")

add_custom_target(plots
    # Plain python3 from the caller's PATH: activate the ASGarD venv in the
    # shell that runs `cmake --build`.
    COMMAND ${CMAKE_COMMAND} -E env PYTHONUNBUFFERED=1
            python3 "${CMAKE_CURRENT_SOURCE_DIR}/plot/show_all.py"
            -j ${ICRF_PLOT_WORKERS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${OUTPUT_FILE}
    COMMENT "Rendering figures (plot/show_all.py)"
    USES_TERMINAL
    VERBATIM)
