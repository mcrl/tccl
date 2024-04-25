add_library(benchmark_v2 benchmark_v2.cpp conf_v2.cpp kernels_v2.cu util.cpp ibv_helper.cpp)
target_link_libraries(benchmark_v2 mpi numa cudart ibverbs sharp_coll tinyxml2)
target_link_options(benchmark_v2 PRIVATE -fopenmp)
target_compile_options(benchmark_v2 PRIVATE -fopenmp -g)

add_executable(pathfinder launch_benchmark.cpp)
target_link_libraries(pathfinder benchmark_v2 hwloc)
target_compile_options(pathfinder PRIVATE -O3 -fopenmp -g)

add_executable(launch_single launch_single.cpp)
target_link_libraries(launch_single benchmark_v2)