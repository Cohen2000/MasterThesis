// Weighted undirected walk. Each incident dyad e is selected proportional to w_e.
// Fixed SplitMix64 stream; integer rejection gives unbiased bounded draws, and the
// neighbour is found by binary search over cumulative incident weights, so every
// transition is exact integer arithmetic.
#include <cstdint>
#include <vector>
#include <algorithm>
static uint64_t next(uint64_t &s) {
    uint64_t z=(s+=0x9e3779b97f4a7c15ULL);
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
}
static uint64_t bounded(uint64_t &s,uint64_t n) {
    uint64_t x, threshold=-n%n;
    do { x=next(s); } while(x<threshold);
    return x%n;
}
// ptr/neighbors/edges: CSR adjacency; cumulative[j] = sum of incident weights of the
// current vertex up to and including position j. weight: volume a first traversal of
// dyad e adds (K_e). Outputs: delta (volume discovered per step, summed over paths),
// volumes (final volume per path), traversals (per path and dyad, optional), executed.
extern "C" void walks(int64_t N,int64_t D,const int64_t *ptr,
 const int64_t *neighbors,const int64_t *edges,const int64_t *cumulative,
 const int64_t *weight,const int64_t *component_volume,const uint64_t *seeds,
 int64_t paths,int64_t L,int64_t *delta,int64_t *volumes,int64_t *traversals,
 int64_t *executed) {
    std::fill(delta,delta+L+1,0);
    for(int64_t p=0;p<paths;p++) {
        uint64_t state=seeds[p]; int64_t node=bounded(state,N), volume=0;
        int64_t maximum=component_volume[node];
        std::vector<uint8_t> seen(D,0);
        if(traversals) std::fill(traversals+p*D,traversals+(p+1)*D,0);
        executed[p]=0;
        for(int64_t s=1;s<=L;s++) {
            int64_t begin=ptr[node],end=ptr[node+1];
            int64_t x=bounded(state,cumulative[end-1]);
            int64_t j=std::upper_bound(cumulative+begin,cumulative+end,x)-cumulative;
            int64_t e=edges[j]; node=neighbors[j]; executed[p]++;
            if(traversals) traversals[p*D+e]++;
            if(!seen[e]) { seen[e]=1; volume+=weight[e]; delta[s]+=weight[e]; }
            // Only volume summaries: remaining values are now exactly constant.
            if(!traversals && volume==maximum) break;
        }
        volumes[p]=volume;
    }
}
