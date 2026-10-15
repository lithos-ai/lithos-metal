// Copyright (c) 2023-2026 The ggml authors (MIT; see third_party/TURBOQUANT-LICENSE).
// adapted from TheTom/llama-cpp-turboquant ggml/src/ggml-turbo-quant.c,
// ggml/src/ggml-quants.c, ggml/src/ggml-metal/turbo-wht.h @ 6e09de1b44ae328f641d1367bb4c875a8ffab43f
// KV_CACHE_TYPE: 0 BF16, 1 Q8_0 (32/34 bytes), 2 Turbo4_0 (128/66 bytes), 3 Q4_0 (32/18 bytes).
#ifndef KV_CACHE_TYPE
#define KV_CACHE_TYPE 0
#endif
#ifndef K_CACHE_TYPE
#define K_CACHE_TYPE KV_CACHE_TYPE
#endif
#ifndef V_CACHE_TYPE
#define V_CACHE_TYPE KV_CACHE_TYPE
#endif
#ifndef TURBO4_ROTATED
#define TURBO4_ROTATED (!DRAFT)
#endif
#define K_TURBO4_ROTATED (K_CACHE_TYPE == 2 && TURBO4_ROTATED)
#define V_TURBO4_ROTATED (V_CACHE_TYPE == 2 && TURBO4_ROTATED)
constant float kv_signs1[128] = {-1.0f,1.0f,1.0f,-1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,-1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,-1.0f,1.0f,1.0f,1.0f,-1.0f,1.0f,1.0f,-1.0f,-1.0f,-1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,1.0f,-1.0f,-1.0f,1.0f,-1.0f,1.0f,1.0f,1.0f,1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,1.0f,-1.0f,1.0f,1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f,-1.0f,1.0f,-1.0f,-1.0f,-1.0f,1.0f,-1.0f,-1.0f,-1.0f,1.0f,-1.0f,-1.0f,-1.0f,1.0f,1.0f,1.0f,-1.0f,-1.0f,1.0f,1.0f,1.0f,-1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f};
constant float kv_signs2[128] = {1.0f,1.0f,1.0f,1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f,-1.0f,-1.0f,1.0f,-1.0f,-1.0f,-1.0f,1.0f,1.0f,-1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,-1.0f,1.0f,-1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,-1.0f,-1.0f,-1.0f,1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,1.0f,1.0f,-1.0f,-1.0f,1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f,-1.0f,-1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f,-1.0f,-1.0f,-1.0f,1.0f,-1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,1.0f,1.0f,-1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,1.0f,-1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,1.0f,-1.0f};
constant float kv_centroids[16] = {-0.241529f,-0.182877f,-0.143016f,-0.111036f,
 -0.083292f,-0.058050f,-0.034299f,-0.011349f,0.011349f,0.034299f,0.058050f,0.083292f,
 0.111036f,0.143016f,0.182877f,0.241529f};
constant float kv_thresholds[15] = {-0.212203f,-0.162947f,-0.127026f,-0.097164f,
 -0.070671f,-0.046174f,-0.022824f,0,0.022824f,0.046174f,0.070671f,0.097164f,
 0.127026f,0.162947f,0.212203f};
// Read a dimension from the existing contiguous DL-per-lane attention registers.
static inline float kv_gather(const thread float* f, uint dim) {
  float result = 0;
  for (uint e = 0; e < DL; ++e) {
    float v = simd_shuffle(f[e], ushort(dim / DL));
    if (e == dim % DL) result = v;
  }
  return result;
}
// A 128-element block, four consecutive dimensions per lane.
static inline void kv_rotate(thread float* x, uint lane, bool inverse) {
  for (uint e=0; e<4; ++e) x[e] *= inverse ? kv_signs2[lane*4+e] : kv_signs1[lane*4+e];
  for (uint h=1; h<4; h*=2) {
    for (uint i=0; i<4; i+=2*h) for (uint j=i; j<i+h; ++j) {
      float a=x[j], b=x[j+h]; x[j]=a+b; x[j+h]=a-b;
    }
  }
  for (uint h=1; h<32; h*=2) for (uint e=0; e<4; ++e) {
    float a=x[e], b=simd_shuffle_xor(a, h);
    x[e] = (lane & h) ? b-a : a+b;
  }
  for (uint e=0; e<4; ++e) x[e] *= 0.08838834764831845f *
    (inverse ? kv_signs1[lane*4+e] : kv_signs2[lane*4+e]);
}
// Transform each 128-wide block of a head without changing its lane layout.
// Only queries and the final merged output need this, never historical K/V.
static inline void kv_rotate_head(thread float* f, uint lane, bool inverse) {
  for (uint b=0; b<D/128; ++b) {
    float x[4];
    for (uint e=0; e<4; ++e) x[e]=kv_gather(f,b*128+lane*4+e);
    kv_rotate(x,lane,inverse);
    for (uint e=0; e<DL; ++e) {
      uint dim=lane*DL+e, local=dim%128;
      float value=0;
      for (uint i=0; i<4; ++i) {
        float v=simd_shuffle(x[i],ushort(local/4));
        if (i==local%4) value=v;
      }
      if (dim/128==b) f[e]=value;
    }
  }
}

// Reconstruct in registers with exactly the stored FP16 scale. Every row group
// derives fresh K/V from the projection, so no cross-threadgroup cache read is needed.
template<bool RECONSTRUCT, bool ROTATED=false, int TYPE=KV_CACHE_TYPE>
static inline void kv_encode(device ushort* cache, uint row, thread float* f, uint lane, bool write) {
  if constexpr (TYPE == 0) {
    if (write) store_dl(cache + row*D + lane*DL, f);
  } else if constexpr (TYPE == 1) {
    device uchar* dst = write ? (device uchar*)cache + row*(D/32)*34 : nullptr;
    for (uint b=0; b<D/32; ++b) {
      float v = kv_gather(f, b*32+lane);
      float scale = simd_max(abs(v))/127.0f;
      if (write && lane == 0) *(device half*)(dst+b*34) = half(scale);
      float inv = scale > 0 ? 1.0f/scale : 0.0f;
      int q = clamp(int(round(v*inv)), -127, 127);
      if (write) dst[b*34+2+lane] = as_type<uchar>(char(q));
      if (RECONSTRUCT) for (uint e=0; e<DL; ++e) {
        uint dim=lane*DL+e;
        float value=simd_shuffle(float(q),ushort(dim%32));
        if (dim/32==b) f[e]=value*float(half(scale));
      }
    }
  } else if constexpr (TYPE == 3) {
    device uchar* dst = write ? (device uchar*)cache + row*(D/32)*18 : nullptr;
    for (uint b=0; b<D/32; ++b) {
      float v = kv_gather(f, b*32+lane);
      float amax = simd_max(abs(v));
      uint first = simd_min(abs(v) == amax ? lane : 32u);
      float peak = simd_shuffle(v, ushort(first));
      float scale = amax > 0 ? peak / -8.0f : -0.0f;
      float inv = scale != 0 ? precise::divide(1.0f, scale) : 0.0f;
      // Preserve ggml's separate FP32 multiply and bias at half-step ties.
      volatile float scaled = v*inv;
      uint q = uint(clamp(int(scaled + 8.5f), 0, 15));
      uint hi = simd_shuffle(q, ushort((lane & 15u) + 16u));
      if (write) {
        if (lane == 0) *(device half*)(dst+b*18) = half(scale);
        if (lane < 16) dst[b*18+2+lane] = uchar(q | (hi << 4));
      }
      if (RECONSTRUCT) for (uint e=0; e<DL; ++e) {
        uint dim=lane*DL+e;
        float value=simd_shuffle(float(int(q)-8), ushort(dim%32));
        if (dim/32==b) f[e]=value*float(half(scale));
      }
    }
  } else {
    device uchar* dst = write ? (device uchar*)cache + row*(D/128)*66 : nullptr;
    for (uint b=0; b<D/128; ++b) {
      float x[4], ss=0;
      for (uint e=0; e<4; ++e) { x[e]=kv_gather(f,b*128+lane*4+e); ss+=x[e]*x[e]; }
      float norm=sqrt(simd_sum(ss));
      for (uint e=0; e<4; ++e) x[e] = norm > 1e-10f ? x[e]/norm : 0;
      kv_rotate(x,lane,false);
      uint codes[4]; ss=0;
      for (uint e=0; e<4; ++e) {
        uint q=0; for (uint i=0; i<15; ++i) q += uint(x[e]>=kv_thresholds[i]);
        codes[e]=q; ss+=kv_centroids[q]*kv_centroids[q];
      }
      float scale=norm/sqrt(simd_sum(ss));
      if (write) {
        if (lane==0) *(device half*)(dst+b*66)=half(scale);
        for (uint e=0; e<2; ++e) dst[b*66+2+lane*2+e]=uchar(codes[2*e] | (codes[2*e+1]<<4));
      }
      if (RECONSTRUCT) {
        for (uint e=0; e<4; ++e) x[e]=kv_centroids[codes[e]];
        if (!ROTATED) kv_rotate(x,lane,true);
        for (uint e=0; e<DL; ++e) {
          uint dim=lane*DL+e, local=dim%128;
          float value=0;
          for (uint i=0; i<4; ++i) {
            float v=simd_shuffle(x[i],ushort(local/4));
            if (i==local%4) value=v;
          }
          if (dim/128==b) f[e]=value*float(half(scale));
        }
      }
    }
  }
}
template<int TYPE=KV_CACHE_TYPE>
static inline void kv_store(device ushort* cache, uint row, const thread float* f, uint lane) {
  float copy[DL]; for (uint e=0; e<DL; ++e) copy[e]=f[e];
  kv_encode<false, false, TYPE>(cache,row,copy,lane,true);
}
template<int TYPE=KV_CACHE_TYPE>
static inline void kv_roundtrip(thread float* f, uint lane) {
  kv_encode<true, false, TYPE>(nullptr,0,f,lane,false);
}
template<int TYPE=KV_CACHE_TYPE>
static inline void kv_load(device const ushort* cache, uint row, thread float* f, uint lane) {
  if constexpr (TYPE == 0) {
    load_dl(cache + row*D + lane*DL, f);
  } else if constexpr (TYPE == 1) {
    device const uchar* src=(device const uchar*)cache + row*(D/32)*34;
    for (uint e=0; e<DL; ++e) {
      uint dim=lane*DL+e, b=dim/32;
      f[e]=float(*(device const half*)(src+b*34))*float(as_type<char>(src[b*34+2+dim%32]));
    }
  } else if constexpr (TYPE == 3) {
    device const uchar* src=(device const uchar*)cache + row*(D/32)*18;
    for (uint e=0; e<DL; ++e) {
      uint dim=lane*DL+e, b=dim/32, local=dim%32;
      uint q=(src[b*18+2+local%16] >> ((local/16)*4)) & 15;
      f[e]=float(*(device const half*)(src+b*18))*float(int(q)-8);
    }
  } else {
    device const uchar* src=(device const uchar*)cache + row*(D/128)*66;
    for (uint b=0; b<D/128; ++b) {
      float x[4];
      float scale=float(*(device const half*)(src+b*66));
      for (uint e=0; e<4; ++e) {
        uint dim=lane*4+e, q=(src[b*66+2+dim/2] >> ((dim%2)*4)) & 15;
        x[e]=kv_centroids[q];
      }
      kv_rotate(x,lane,true);
      for (uint e=0; e<DL; ++e) {
        uint dim=lane*DL+e, local=dim%128;
        float value=0;
        for (uint i=0; i<4; ++i) {
          float v=simd_shuffle(x[i],ushort(local/4));
          if (i==local%4) value=v;
        }
        if (dim/128==b) f[e]=value*scale;
      }
    }
  }
}
template<int TYPE=KV_CACHE_TYPE>
static inline void kv_copy(device ushort* cache, uint row, device const ushort* src, uint lane) {
  float f[DL]; load_dl(src,f); kv_store<TYPE>(cache,row,f,lane);
}

// Attention-domain helpers keep standalone codec behavior and packed bytes
// unchanged. Fresh rows use exactly the same FP16 corrected norm as the cache.
template<int TYPE=KV_CACHE_TYPE, bool ROTATED=(TYPE == 2 && TURBO4_ROTATED)>
static inline void kv_attention_load(device const ushort* cache, uint row, thread float* f, uint lane) {
  if constexpr (ROTATED) {
    device const uchar* src=(device const uchar*)cache + row*(D/128)*66;
    for (uint e=0; e<DL; ++e) {
      uint dim=lane*DL+e, b=dim/128, local=dim%128;
      uint q=(src[b*66+2+local/2] >> ((local%2)*4)) & 15;
      f[e]=kv_centroids[q]*float(*(device const half*)(src+b*66));
    }
  } else {
      kv_load<TYPE>(cache,row,f,lane);
  }
}
template<int TYPE=KV_CACHE_TYPE, bool ROTATED=(TYPE == 2 && TURBO4_ROTATED)>
static inline void kv_attention_encode(device ushort* cache, uint row, thread float* f, uint lane, bool write) {
  kv_encode<true, ROTATED, TYPE>(cache,row,f,lane,write);
}
