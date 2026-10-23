# m2_max_30c kernel overrides

The 30-core M2 Max backend uses the shared native shader implementations in
`kernels/common/`. Tensor acceleration and automatic megakernel fusion remain
disabled. Place future chip-specific templates here with the same relative
names as the common sources; they override only this exact chip variant.
