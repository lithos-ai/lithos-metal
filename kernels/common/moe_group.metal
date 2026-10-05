// Compact routing pairs by expert, preserving each original token/slot index.
// Algorithmic reference: Mirage PR #786, expert_queue.cuh build_table; this is
// an independent Metal implementation, with bounded loops and no atomic sums.
// Reference commit: c28cac618b98fda1e0f1d590923b5f69b4ef3603.
#ifndef QUEUE_WORDS
#define QUEUE_WORDS 1
#endif
kernel void moe_group(device const int* ids [[buffer(0)]], device int* table [[buffer(1)]],
                      constant uint& rows [[buffer(2)]],
#if STEP_STATE
                      device const StepState* st [[buffer(15)]],
#endif
                      uint tid [[thread_index_in_threadgroup]]) {
#if STEP_STATE
  if (st->done) return;
  const uint active = min(st->t_this_step, uint(MAX_T));
#else
  const uint active = rows;
#endif
  const uint pairs = active * TOP_K;
  for(uint i=tid;i<QUEUE_WORDS;i+=256u) table[1u+MAX_T*TOP_K*(GROUP_T+2u)+i]=0;
  threadgroup int selected[MAX_T * TOP_K];
  threadgroup uint offsets[EXPERTS + 1];
  for (uint i=tid; i<pairs; i+=256u) selected[i]=ids[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e=tid; e<EXPERTS; e+=256u) {
    uint count=0u;
    for (uint i=0u; i<pairs; i++) count += uint(selected[i] == int(e));
    offsets[e]=(count+GROUP_T-1u)/GROUP_T;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid==0u) {
    uint sum=0u;
    for (uint e=0u; e<EXPERTS; e++) { uint count=offsets[e]; offsets[e]=sum; sum+=count; }
    offsets[EXPERTS]=sum; table[0]=int(sum);
    table[1u+MAX_T*TOP_K*(GROUP_T+2u)]=0; // queue for independent expert tasks
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e=tid; e<EXPERTS; e+=256u) {
    uint found=0u;
    for (uint i=0u; i<pairs; i++) if (selected[i]==int(e)) {
      const uint at=1u+(offsets[e]+found/GROUP_T)*(GROUP_T+2u);
      table[at]=int(e); table[at+1u]=int(found%GROUP_T+1u);
      table[at+2u+found%GROUP_T]=int(i); found++;
    }
  }
}
