// steal.metal: own-slice + steal block claiming for ops with uneven blocks (design D9, §5.3; plan M7, #44) —
// probes/p10_claim_protocol.metal's mode 2 as a helper a kernel's block loop can opt into (STEAL=1).
//
// Every *nominal* SIMD-group v owns the contiguous slice [v·n/NOM, (v+1)·n/NOM) of the op's blocks behind its own
// cursor (a device atomic, zeroed before the dispatch by steal_reset). A SIMD-group drains its own slice first (the
// CAS is uncontended in the common case), then visits the other slices in a per-SIMD-group order (a stride coprime
// with NOM, so thieves fan out over different victims) and steals what is left. SIMD-groups beyond the nominal crew
// (surplus threadgroups) own nothing and only steal; missing ones cost nothing but their slice, which the others
// drain. Every block is claimed exactly once; which SIMD-group computes it never matters (a block's output does not
// depend on the claimant), and the op's consumer is the next dispatch, so no fence is needed. Lane 0 claims, the
// claim reaches the 32 lockstep lanes through simd_broadcast_first.
//
// Cost: one atomic CAS per block plus a scan of NOM cursors when the work runs out — enabled per op only where a
// per-op trace shows tail skew and the A/B gains (decode-kernels.md §7: none of the dense targets' ops does).
#define STEAL_NONE 0xFFFFFFFFu

struct StealScan { uint k; uint v; uint lo; uint hi; };     // victims visited, the current victim and its slice

static inline uint steal_stride(uint sg, uint nominal) {   // a stride coprime with `nominal`, varying with the SIMD-group
  const uint primes[16] = {5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u, 41u, 43u, 47u, 53u, 59u, 61u};
  for (uint i = 0; i < 16u; i++) {
    const uint s = primes[(sg + i) & 15u];
    uint a = s, b = nominal;
    while (b) { const uint t = a % b; a = b; b = t; }
    if (a == 1u) return s;
  }
  return 1u;
}

static inline StealScan steal_begin() { StealScan s = {0u, STEAL_NONE, 0u, 0u}; return s; }

// The next block for this SIMD-group, or STEAL_NONE when every slice is drained.
static inline uint steal_next(device atomic_uint* cursors, uint n_blocks, uint nominal, uint sg, uint lane, thread StealScan& s) {
  const uint stride = steal_stride(sg, nominal);
  while (s.k < nominal) {
    if (s.v == STEAL_NONE) {
      if (s.k == 0u && sg >= nominal) { s.k = 1u; continue; }                 // a surplus SIMD-group owns no slice
      const uint v = (sg + s.k * stride) % nominal;
      s.v = v; s.lo = v * n_blocks / nominal; s.hi = (v + 1u) * n_blocks / nominal;
    }
    uint b = STEAL_NONE;
    if (lane == 0u) {
      uint cur = atomic_load_explicit(&cursors[s.v], memory_order_relaxed);
      while (s.lo + cur < s.hi) {
        if (atomic_compare_exchange_weak_explicit(&cursors[s.v], &cur, cur + 1u, memory_order_relaxed, memory_order_relaxed)) { b = s.lo + cur; break; }
      }
    }
    b = simd_broadcast_first(b);
    if (b != STEAL_NONE) return b;
    s.v = STEAL_NONE;
    s.k++;
  }
  return STEAL_NONE;
}

// Zero the cursors before a stealing dispatch (one tiny dispatch, or the step's serial tail).
kernel void steal_reset(device uint* cursors [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) cursors[gid] = 0u;
}
