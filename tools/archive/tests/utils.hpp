#pragma once

#define LOG_RANK_ANY(...) \
  do { \
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    printf("[Rank %d] ", rank); \
    printf(__VA_ARGS__); \
    printf("\n"); \
  } while (0)
