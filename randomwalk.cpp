#include <iostream>
#include <mpi.h>
#include <vector>

const int N = 100;
int world_size, rank;

class Walker {
public:
  int location;
  int remain;
  void walk() {
    if (remain > 0) {
      remain --;
      if(location == N) {
        location = 0;
      }else{
        location ++;
      }
    }
  }
  [[nodiscard]] bool can_walk() const {
    return remain != 0;
  }
  Walker(int location, int remain): location(location), remain(remain){}
  Walker(): location(0), remain(0){}
};


int get_region_id(int x) {
  int region_size = (N + 1) / world_size - 1;
  return x / region_size;
}


std::vector<Walker> walkers;

void walk(Walker& w, std::vector<Walker>& outgoing_walkers, int region_id, int np) {
  while(w.can_walk()) {
    w.walk();
    if(get_region_id(w.location) != region_id) {
      outgoing_walkers.push_back(w);
      break;
    }
  }
  if(!w.can_walk()) {
    printf("Walker stopped at %d\n", w.location);
  }
}

void send_vector(int to, const std::vector<Walker> &v) {
  printf("Worker %d send %d walkers to %d\n", rank, static_cast<int>(v.size()), to);
  MPI_Send(v.data(), v.size() * sizeof(Walker), MPI_BYTE, to, 0, MPI_COMM_WORLD);
}

std::vector<Walker> receive_vector(int from) {
  std::vector<Walker> ans;
  MPI_Status mpiStatus;
  MPI_Probe(from, MPI_ANY_TAG, MPI_COMM_WORLD, &mpiStatus);
  int msgSize;
  MPI_Get_count(&mpiStatus, MPI_BYTE, &msgSize);
  ans.resize(msgSize/ sizeof(Walker));
  MPI_Recv(ans.data(), msgSize, MPI_BYTE, from, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  double stepsum = 0;
  for (auto && w : ans) stepsum += w.remain;
  printf("Worker %d receive %d walkers from %d, with avg remain of %lf\n", rank, static_cast<int>(ans.size()), from, stepsum / ans.size());
  return std::move(ans);
}

int get_next(int x) {
  return (x + 1) % world_size;
}

int get_prev(int x) {
  return (x - 1 + world_size) % world_size;
}


int main() {
  MPI_Init(nullptr, nullptr);

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << rank << " online" << std::endl;

  for(int i = 0;i < 20;i++) {
    walkers.emplace_back(rank * ((N + 1) / world_size - 1), rand() % 40);
  }

  int t = 10;
  while(t--) {
    std::vector<Walker> outgoing;
    for(auto && walker : walkers) {
      walk(walker, outgoing, rank, world_size);
    }
    if(rank % 2 == 0) {
      send_vector(get_next(rank), outgoing);
      auto res = receive_vector(get_prev(rank));
      walkers = std::move(res);
    }else{
      auto res = receive_vector(get_prev(rank));
      send_vector(get_next(rank), outgoing);
      walkers = std::move(res);
    }
  }
  MPI_Finalize();

}