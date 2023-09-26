#include <mpi.h>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <array>
#include <vector>
#include <numeric>
#include <mdspan>

constexpr int N = 10000;
constexpr int INF = 0x7fffffff;
int rank, world_size;
int num_per_node;

void generate_graph(std::mdspan<int, std::dextents<size_t, 2>> g) {
  for(int i = 0;i < N;i++) {
    for(int j = 0;j < N;j++) {
      if(i == j) g[i, j] = 0;
      else g[i, j] = rand() % 20+1;
    }
  }
}

double dijkstra(std::mdspan<int, std::dextents<size_t, 2>> g, int S) {
  std::vector<bool> vis (N,false);
  std::vector<int> dist (N,INF);
  dist[S] = 0;
  for(int i = 1;i < N; i++) {
    int minn = INF, mini = 0;
    for(int j = 0;j < N;j++) {
      if(vis[j]) continue;
      if(minn > dist[j]) {
        minn = dist[j];
        mini = j;
      }
    }
    vis[mini] = true;
    for(int j = 0;j < N;j++) {
      if(vis[j]) continue;
      if(dist[j] > dist[mini] + g[mini, j]) {
        dist[j] = dist[mini] + g[mini, j];
      }
    }
  }
  return static_cast<double>(std::accumulate(dist.begin(), dist.end(), 0)) / N;
}

bool is_in_domination(int x) {
  return x >= num_per_node * rank && x < num_per_node * (rank+1);
}
double parallel_dijkstra(std::mdspan<int, std::dextents<size_t, 2>> g, int S) {
  std::vector<bool> vis (num_per_node,false);
  std::vector<int> dist (num_per_node,INF);
  if(is_in_domination(S))
    dist[S % num_per_node] = 0;
  for(int i = 1;i < N; i++) {
    int minn = INF, mini = 0;
    for(int j = 0;j < num_per_node;j++) {
      if(vis[j]) continue;
      if(minn > dist[j]) {
        minn = dist[j];
        mini = j;
      }
    }
    std::array<int, 2> m {minn, mini + rank*num_per_node};
    std::array<int, 2> gm {};
//    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(m.data(), gm.data(), 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    if(is_in_domination(gm[1])) {
      vis[gm[1] % num_per_node] = true;
    }
    for(int j = 0;j < num_per_node;j++) {
      if(vis[j]) continue;
      if(dist[j] > gm[0] + g[gm[1], j + rank*num_per_node]) {
        dist[j] = gm[0] + g[gm[1], j + rank*num_per_node];
      }
    }
  }
  double tans = static_cast<double>(std::accumulate(dist.begin(), dist.end(), 0)) / num_per_node;
//  printf("Worker %d get tans: %lf\n", rank, tans);
  return tans;
}

int main() {
  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  num_per_node = N / world_size;
  std::vector<int> graph_v(N*N, 0);
  auto graph = std::mdspan(graph_v.data(), N, N);
  if(rank == 0) {
    srand(time(nullptr));
    generate_graph(graph);
    auto t1 = MPI_Wtime();
    double ad;
    int t = 1;
    while(t--)
      ad = dijkstra(graph, 0);
    auto t2 = MPI_Wtime();
    printf("avg dist is %lf, cost %lf seconds.\n", ad, t2-t1);
  }
  MPI_Bcast(graph_v.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  printf("Worker %d received graph!\n", rank);
  auto t1 = MPI_Wtime();
  int t = 1;

  double ans = 0;
  while(t--) {
    double tans = parallel_dijkstra(graph, 0);
    MPI_Reduce(&tans, &ans, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  auto t2 = MPI_Wtime();

  if(rank == 0) {
    printf("mpi avg dist is %lf, cost %lf seconds.\n", ans / world_size, t2-t1);
  }
  MPI_Finalize();
}
