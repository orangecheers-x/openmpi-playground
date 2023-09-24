#include <mpi.h>
#include <cstdio>
#include <cassert>

int rank, world_size;

int get_next(int x) {
  return (x + 1) % world_size;
}

int get_prev(int x) {
  return (x - 1 + world_size) % world_size;
}

void mybarrier() {
  if(rank == 0) {
    int data = 0;
    MPI_Send(&data, 1, MPI_INT, get_next(rank), 0, MPI_COMM_WORLD);
    MPI_Recv(&data, 1, MPI_INT, get_prev(rank), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    data = 1;
    MPI_Send(&data, 1, MPI_INT, get_next(rank), 0, MPI_COMM_WORLD);
  }else{
    int data = -1;
    MPI_Recv(&data, 1, MPI_INT, get_prev(rank), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    assert(data == 0);
    MPI_Send(&data, 1, MPI_INT, get_next(rank), 0, MPI_COMM_WORLD);
    MPI_Recv(&data, 1, MPI_INT, get_prev(rank), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    assert(data == 1);
    if(rank != world_size) {
      MPI_Send(&data, 1, MPI_INT, get_next(rank), 0, MPI_COMM_WORLD);
    }
  }
  return ;
}

