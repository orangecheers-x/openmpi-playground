#include <mpi.h>
#include <cstdio>

int rank, world_size;

int highbit(int x) {
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return x - (x >> 1);
}

void mybroadcast() {
  int hb = highbit(rank);
  int msg;
  if(rank != 0) {
    MPI_Recv(&msg, 1, MPI_INT, rank - hb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//    printf("%d received msg %d.\n", rank, msg);
  }else{
    msg = 99;
  }
  if(hb != 0)
    hb <<= 1;
  else
    hb = 1;
  while((hb | rank) < world_size) {
    MPI_Send(&msg, 1, MPI_INT, hb | rank, 0, MPI_COMM_WORLD);
//    printf("%d send msg to %d.\n", rank, hb | rank);
    hb <<= 1;
  }
}

int main() {
  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto t = MPI_Wtime();
  for(int i = 0; i < 1000; i++) {
        mybroadcast();
  }
  auto t2 = MPI_Wtime();
  printf("mybroadcast: %lf\n", t2 - t);

  int d = 99;
  t = MPI_Wtime();
  for(int i = 0; i < 1000; i++) {
        MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  t2 = MPI_Wtime();
  printf("mpi_bcast: %lf\n", t2 - t);

  MPI_Finalize();
}