add_executable(test-multiprocess-init test-multiprocess-init.cpp)
target_link_libraries(test-multiprocess-init PRIVATE mpi nccl cudart)