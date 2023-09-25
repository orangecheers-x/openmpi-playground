#include <mpi.h>
#include <cstdio>
#include <vector>
#include <algorithm>

int rank, world_size;

int prank(int x) {
  int sum = -1;
  MPI_Allreduce(&x, &sum, 1, MPI_INT, MPI_SUM,MPI_COMM_WORLD);
  return sum;
}
int d[] = {5,3,56,2,6,4,5,21};
int main() {
  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int tx = d[rank];
  printf("Worker %d generate %d.\n", rank, tx);
  int tans = prank(tx);
  printf("Worker %d get ans: %d.\n", rank, tans);
  MPI_Finalize();
}