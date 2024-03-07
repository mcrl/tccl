add_executable(main main.cpp)

add_executable(benchmark benchmark.cpp conf.cpp util.cpp job.cpp kernels.cu ibv_helper.cpp sharp_helper.cpp)
target_link_libraries(benchmark mpi numa cudart ibverbs sharp_coll)
target_link_options(benchmark PRIVATE -fopenmp)
target_compile_options(benchmark PRIVATE -mavx2 -O3 -fopenmp -g)

add_library(benchmark_v2 benchmark_v2.cpp conf_v2.cpp kernels_v2.cu util.cpp ibv_helper.cpp)
target_link_libraries(benchmark_v2 mpi numa cudart ibverbs sharp_coll tinyxml2)
target_link_options(benchmark_v2 PRIVATE -fopenmp)
target_compile_options(benchmark_v2 PRIVATE -fopenmp -g)

add_executable(launch_benchmark launch_benchmark.cpp)
target_link_libraries(launch_benchmark benchmark_v2)
target_compile_options(launch_benchmark PRIVATE -O3 -fopenmp -g)

add_executable(launch_single launch_single.cpp)
target_link_libraries(launch_single benchmark_v2)

add_executable(test-kernel-comm test-kernel-comm.cu)
target_link_libraries(test-kernel-comm numa)
#target_compile_options(test PRIVATE -Xptxas -dlcm=cg)

add_executable(test-nccl-hang test-nccl-hang.cu)
target_link_libraries(test-nccl-hang cuda)

add_executable(test-multi-context test-multi-context.cu)
target_link_libraries(test-multi-context cuda)
#target_compile_options(test-multi-context PRIVATE --default-stream per-thread)

add_executable(test-cuda-ipc test-cuda-ipc.cu)
target_link_libraries(test-cuda-ipc mpi cuda)

add_executable(test test.cpp)
target_link_libraries(test tinyxml2 numa benchmark_v2)

add_executable(test-nccl-sendrecv test-nccl-sendrecv.cpp)
target_link_libraries(test-nccl-sendrecv cudart nccl mpi)

add_executable(test-search-launch-conf test-search-launch-conf.cu)
target_link_libraries(test-search-launch-conf cudart numa)