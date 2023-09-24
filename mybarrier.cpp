#include <mpi.h>
#include <cstdio>
#include <cassert>
#include <unistd.h>
#include <ctime>

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
    if(rank != world_size-1) {
      MPI_Send(&data, 1, MPI_INT, get_next(rank), 0, MPI_COMM_WORLD);
    }
  }
  return ;
}

int main() {
  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  srand(rank);
  for(int i = 0; i <= 100; i++) {
    sleep(rand() % 8 + 2);
    time_t t = time(nullptr) - 1695553172;
    printf("[%ld] Worker %d finished task %d.\n", t, rank, i);
    mybarrier();
    t = time(nullptr) - 1695553172;
    printf("[%ld] Worker %d start next task %d.\n", t, rank, i+1);
  }
  MPI_Finalize();
}

