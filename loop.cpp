#include <mpi.h>
#include <iostream>
#include <unistd.h>


int get_next(int a, int all) {
  return (a + 1) % all;
}

int get_prev(int a, int all) {
  return (a - 1 + all) % all;
}
int main() {
  MPI_Init(nullptr, nullptr);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << "Worker " << rank << " started!" << std::endl;
  int d1 = 1;
  if(rank == 0) {
    MPI_Send(&d1, 0, MPI_INT, get_next(rank, world_size), 0, MPI_COMM_WORLD);
  }
  while(true){
    int d = -1;
    MPI_Recv(&d, 0, MPI_INT, get_prev(rank, world_size), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::cout << rank << " Receive message " << d << std::endl;
    sleep(1);
    MPI_Send(&d, 0, MPI_INT, get_next(rank, world_size), 0, MPI_COMM_WORLD);
  }

}