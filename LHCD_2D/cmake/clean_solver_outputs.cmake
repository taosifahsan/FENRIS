if(NOT DEFINED OUTPUT_DATA_DIR)
    message(FATAL_ERROR "OUTPUT_DATA_DIR must be provided")
endif()

# A run owns all HDF5 files beneath output_data. Remove snapshots and the
# previous final solution so movie generation can never mix separate runs.
file(GLOB_RECURSE STALE_SOLVER_OUTPUTS
    LIST_DIRECTORIES false
    "${OUTPUT_DATA_DIR}/*.h5"
    "${OUTPUT_DATA_DIR}/*.hdf5")

if(STALE_SOLVER_OUTPUTS)
    file(REMOVE ${STALE_SOLVER_OUTPUTS})
endif()

file(MAKE_DIRECTORY "${OUTPUT_DATA_DIR}/movie")
