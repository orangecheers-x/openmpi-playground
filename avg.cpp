#include <mpi.h>
#include <cstdio>
#include <vector>

int rank, world_size;
int N, num_per_node;

double solve(const std::vector<double> &v) {
  std::vector<double> d;
  d.resize(num_per_node);

  MPI_Scatter(v.data(), num_per_node, MPI_DOUBLE, d.data(), num_per_node, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double sum = 0;
  for(int i = 0;i < 20;i++) {
    for(auto && t : d) {
      sum += t;
    }
  }
  sum /= num_per_node;

  std::vector<double> g;
  g.resize(world_size);
  MPI_Gather(&sum, 1, MPI_DOUBLE, g.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if(rank == 0) {
    sum = 0;
    for (auto &&t : g) {
      sum += t;
    }
    sum /= world_size;
    return sum;
  }
  return 0;
}

int main() {
  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> data;
  N = 10000000;
  srand(time(nullptr));
  num_per_node = N / world_size;
  if (rank == 0) {
    for (int i = 0; i < N; i++)
      data.emplace_back(rand() % 200);
    double ans = 0;
    auto t = MPI_Wtime();
    for(int i = 0;i < 20;i++)
      for(auto && t : data) {
        ans += t;
      }
    ans /= data.size();
    auto t2 = MPI_Wtime();
    printf("ans: %lf, time elapsed: %lf\n", ans, t2-t);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  auto t = MPI_Wtime();
  printf("Worker %d start\n", rank);
  double ans = solve(data);
  auto t2 = MPI_Wtime();
  if(rank == 0) {
    printf("mpians: %lf, time elapsed: %lf\n", ans, t2-t);
  }
  MPI_Finalize();
}