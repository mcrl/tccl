#pragma once

#include <cstddef>

void gpu_copy_wrapper(float* dst, float* src, size_t nbytes);
double gpu_reduce_wrapper(float* d_a, size_t nbytes);