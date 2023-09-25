#include <mpi.h>
#include <cstdio>
#include <vector>
#include <algorithm>

int rank, world_size;

int prank(int x) {
  std::vector<int> ap;
  if(rank == 0) ap.resize(world_size);
  MPI_Gather(&x, 1, MPI_INT, ap.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<std::pair<int, int>> process;
  std::vector<int> ans;
  if(rank == 0) {
    for(int i = 0;i < ap.size(); i++) {
      process.emplace_back(ap[i], i);
    }
    for(int i = 0;i < ap.size(); i++) {
      printf("%d ", ap[i]);
    }
    printf("\n");
    std::ranges::sort(process, [](auto a, auto b) {
      return a.first < b.first;
    });
    for(int i = 0;i < process.size(); i++)
      process[i].first = i;
    std::ranges::sort(process, [](auto a, auto b) {
      return a.second < b.second;
    });
    ans.reserve(process.size());
    for(auto&& t: process) {
      ans.emplace_back(t.first);
    }
  }
  int tans;
  MPI_Scatter(ans.data(), 1, MPI_INT, &tans, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return tans;
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