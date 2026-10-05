# m3_pro kernel overrides

Place chip-specific Metal templates here using the same relative names as
`kernels/common/`. This backend selects these files before the shared templates.
The initial implementation uses the shared sources; its Python backend owns
lowering and scheduling choices.
