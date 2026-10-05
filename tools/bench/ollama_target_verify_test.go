// Benchmark overlay for Ollama v0.35.1's mlxrunner package.
// Copy into that pinned source tree; inference code and kernels stay unchanged.
package mlxrunner

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
)

func TestTargetVerifyBenchmark(t *testing.T) {
	dir := os.Getenv("VERIFY_BENCH_DIR")
	if dir == "" {
		t.Skip("set VERIFY_BENCH_DIR")
	}
	var inputs struct {
		Prefix   []int32 `json:"prefix_ids"`
		Batch    []int32 `json:"batch_ids"`
		Contexts []int   `json:"contexts"`
	}
	data, err := os.ReadFile(filepath.Join(dir, "inputs.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &inputs); err != nil {
		t.Fatal(err)
	}
	reps := 20
	if s := os.Getenv("VERIFY_REPS"); s != "" {
		reps, _ = strconv.Atoi(s)
	}
	if reps < 1 || len(inputs.Batch) != 8 {
		t.Fatal("invalid benchmark inputs")
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		var r Runner
		if err := r.Load("monolith-benchmark-qwen38-nvfp4-bf16"); err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		fmt.Println("BENCH_LOADED", mlx.ActiveMemory())
		caches := r.Model.NewCaches()
		defer func() {
			for _, c := range caches {
				c.Free()
			}
		}()
		states := func() []*mlx.Array {
			var x []*mlx.Array
			for _, c := range caches {
				x = append(x, c.State()...)
			}
			return x
		}
		eval := func() { mlx.Scoped(func() { mlx.Eval(states()...) }) }
		position := 0
		forward := func(ids []int32, pos int, head bool) *mlx.Array {
			b := &batch.Batch{InputIDs: mlx.FromValues(ids, 1, len(ids)), SeqOffsets: []int32{int32(pos)}, SeqQueryLens: []int32{int32(len(ids))}}
			h, _ := r.Model.Forward(b, caches)
			if head {
				return r.Model.Unembed(h)
			}
			return h
		}
		for _, ctx := range inputs.Contexts {
			for position < ctx {
				n := min(128, ctx-position)
				p := position
				mlx.Scoped(func() {
					out := mlx.ScopedArrays(func() []*mlx.Array { return []*mlx.Array{forward(inputs.Prefix[p:p+n], p, false)} })
					mlx.Eval(append(out, states()...)...)
				})
				position += n
			}
			fmt.Println("BENCH_PREFILLED", ctx)
			snapshots := make([]cache.Snapshot, len(caches))
			for i, c := range caches {
				snapshots[i] = c.Snapshot(0)
			}
			eval()
			restore := func() {
				for i, c := range caches {
					if !c.Restore(snapshots[i], ctx) {
						t.Fatalf("restore layer %d", i)
					}
				}
				eval()
			}
			for _, capture := range []bool{false, true} {
				walls := []float64{}
				hashes := []string{}
				memory := []uint64{}
				for rep := -5; rep < reps; rep++ {
					restore()
					if capture {
						for _, c := range caches {
							offsets := make([]int, 8)
							for i := range offsets {
								offsets[i] = ctx + i + 1
							}
							c.PrepareSnapshots(offsets)
						}
					}
					mlx.Scoped(func() {
						started := time.Now()
						out := mlx.ScopedArrays(func() []*mlx.Array { return []*mlx.Array{forward(inputs.Batch, ctx, true)} })
						mlx.Eval(append(out, states()...)...)
						elapsed := float64(time.Since(started).Nanoseconds()) / 1e6
						shape := out[0].Dims()
						if len(shape) != 3 || shape[0] != 1 || shape[1] != 8 {
							t.Fatalf("logit shape %v", shape)
						}
						if rep >= 0 {
							values := out[0].AsType(mlx.DTypeFloat32).Floats()
							bytes := make([]byte, 4*len(values))
							for i, v := range values {
								if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
									t.Fatal("nonfinite logit")
								}
								binary.LittleEndian.PutUint32(bytes[i*4:], math.Float32bits(v))
							}
							sum := sha256.Sum256(bytes)
							walls = append(walls, elapsed)
							hashes = append(hashes, fmt.Sprintf("%x", sum))
							if rep == 0 {
								name := fmt.Sprintf("ollama-logits-%d-%t.safetensors", ctx, capture)
								if err := mlx.SaveSafetensors(filepath.Join(dir, name), map[string]*mlx.Array{"logits": out[0]}); err != nil {
									t.Fatal(err)
								}
							}
						}
					})
					if capture {
						for _, c := range caches {
							for _, s := range c.TakeSnapshots() {
								if s != nil {
									s.Close()
								}
							}
						}
					}
					for _, c := range caches {
						if c.Offset() != ctx+8 {
							t.Fatalf("cache offset %d", c.Offset())
						}
					}
					if rep >= 0 {
						memory = append(memory, uint64(mlx.ActiveMemory()))
					}
				}
				for _, h := range hashes {
					if h != hashes[0] {
						t.Fatal("non-deterministic replay")
					}
				}
				if memory[len(memory)-1] > memory[0]+64*1024*1024 {
					t.Fatalf("memory grew across fixed replays: %v", memory)
				}
				lib, _ := mlx.LoadedLibraryPath()
				row := map[string]any{"engine": "ollama-mlx", "version": "0.35.1", "fp8_mode": "bf16-materialized", "context": ctx, "target_rows": 8, "scope": "embedding + 64 decoder layers + final norm + all-eight-row vocabulary projection + final cache/state updates", "capture_per_token_states": capture, "wall_ms": walls, "active_memory_samples": memory, "logits_sha256": hashes[0], "library": lib, "active_memory_bytes": mlx.ActiveMemory()}
				line, _ := json.Marshal(row)
				f, e := os.OpenFile(filepath.Join(dir, "ollama.jsonl"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
				if e != nil {
					t.Fatal(e)
				}
				f.Write(append(line, '\n'))
				f.Close()
				fmt.Println(string(line))
			}
			restore()
			mlx.Scoped(func() {
				var all []*mlx.Array
				for j := 0; j < 8; j++ {
					out := mlx.ScopedArrays(func() []*mlx.Array { return []*mlx.Array{forward(inputs.Batch[j:j+1], ctx+j, true)} })
					mlx.Eval(append(out, states()...)...)
					all = append(all, out[0])
				}
				logits := mlx.Concatenate(all, 1)
				if err := mlx.SaveSafetensors(filepath.Join(dir, fmt.Sprintf("ollama-serial-logits-%d.safetensors", ctx)), map[string]*mlx.Array{"logits": logits}); err != nil {
					t.Fatal(err)
				}
			})
			restore()
			for _, s := range snapshots {
				if s != nil {
					s.Close()
				}
			}
		}
	})
}
