#pragma once

#include <cstddef>

void gpu_copy_wrapper_multi_channel(float** dsts, float** srcs, size_t* nbytess, int nchannels, int gpu_idx);
void gpu_copy_wrapper_single_channel(float* dst, float* src, size_t nbytes, int gpu_idx);
void gpu_copy_wrapper_memcpy(float* dst, float* src, size_t nbytes, int gpu_idx);