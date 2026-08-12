package model

import (
	"math"

	"inferlab/internal/tensor"
)

// BatchRequest is one sequence's single-token decode step within a batch
// tick: which token is being fed in, at which position, against that
// sequence's own key/value cache buffers (see internal/cache.KVCache.Keys/
// Values — each sequence owns independent buffers, since batching here
// never mixes different sequences' cache contents).
type BatchRequest struct {
	Token              int32
	Pos                int
	KeyCache, ValCache [][]float32
}

// StepBatch computes one decode step for every request together. It lives
// in this package rather than internal/batch because it needs the same
// unexported weights/RoPE-table access Step does; internal/batch owns the
// scheduling policy (which sequences share a tick, admit/evict) on top of
// this primitive.
//
// The position-independent linear projections (QKVO, FFN gate/up/down, the
// final classifier) are computed as one Linear.BatchMatMul call across all
// requests' stacked activations — the shared weight matrix is streamed from
// memory once per layer and amortized over len(reqs) dot products instead
// of once per request, which is the actual memory-bandwidth win batching
// exists to deliver. Attention is deliberately NOT batched: each request
// has its own KV-cache length (sequences enter a batch at different points
// in their generation), and there is no padding/masking scheme here to make
// that shape-uniform — attention stays a plain per-request, per-head loop,
// identical to what Step does for one request. This mirrors, in miniature,
// how systems like vLLM split a batched linear-algebra path from a separate
// ragged/paged attention kernel rather than pretending attention batches
// the same way a feed-forward layer does.
func (m *Model) StepBatch(reqs []BatchRequest) [][]float32 {
	n := len(reqs)
	if n == 0 {
		return nil
	}
	cfg := m.Config
	w := m.weights
	dim, kvDim, headSize, hiddenDim := cfg.Dim, cfg.KVDim(), cfg.HeadSize(), cfg.HiddenDim
	kvMul := cfg.NHeads / cfg.NKVHeads
	invSqrtHead := float32(1.0 / math.Sqrt(float64(headSize)))

	x := make([]float32, n*dim)
	for s, r := range reqs {
		copy(x[s*dim:s*dim+dim], w.TokenEmbeddingRow(r.Token))
	}

	xb := make([]float32, n*dim)
	xb2 := make([]float32, n*dim)
	q := make([]float32, n*dim)
	kTmp := make([]float32, n*kvDim)
	vTmp := make([]float32, n*kvDim)
	hb := make([]float32, n*hiddenDim)
	hb2 := make([]float32, n*hiddenDim)
	scores := make([]float32, cfg.SeqLen) // reused across (request, head) pairs

	for l := 0; l < cfg.NLayers; l++ {
		for s := range reqs {
			tensor.RMSNorm(xb[s*dim:s*dim+dim], x[s*dim:s*dim+dim], w.RMSAttWeight(l))
		}

		w.WQ(l).BatchMatMul(q, xb, n)
		w.WK(l).BatchMatMul(kTmp, xb, n)
		w.WV(l).BatchMatMul(vTmp, xb, n)

		for s, r := range reqs {
			k := r.KeyCache[l][r.Pos*kvDim : r.Pos*kvDim+kvDim]
			v := r.ValCache[l][r.Pos*kvDim : r.Pos*kvDim+kvDim]
			copy(k, kTmp[s*kvDim:s*kvDim+kvDim])
			copy(v, vTmp[s*kvDim:s*kvDim+kvDim])

			qs := q[s*dim : s*dim+dim]
			m.applyRoPE(qs, headSize, r.Pos)
			m.applyRoPE(k, headSize, r.Pos)
		}

		for s, r := range reqs {
			qs := q[s*dim : s*dim+dim]
			out := xb2[s*dim : s*dim+dim]
			for h := 0; h < cfg.NHeads; h++ {
				qh := qs[h*headSize : h*headSize+headSize]
				kvHead := h / kvMul
				active := scores[:r.Pos+1]
				for t := 0; t <= r.Pos; t++ {
					kt := r.KeyCache[l][t*kvDim+kvHead*headSize : t*kvDim+kvHead*headSize+headSize]
					var dot float32
					for i := 0; i < headSize; i++ {
						dot += qh[i] * kt[i]
					}
					active[t] = dot * invSqrtHead
				}
				tensor.Softmax(active)

				oh := out[h*headSize : h*headSize+headSize]
				for i := range oh {
					oh[i] = 0
				}
				for t := 0; t <= r.Pos; t++ {
					vt := r.ValCache[l][t*kvDim+kvHead*headSize : t*kvDim+kvHead*headSize+headSize]
					weight := active[t]
					for i := 0; i < headSize; i++ {
						oh[i] += weight * vt[i]
					}
				}
			}
		}

		w.WO(l).BatchMatMul(xb, xb2, n)
		for i := range x {
			x[i] += xb[i]
		}

		for s := range reqs {
			tensor.RMSNorm(xb[s*dim:s*dim+dim], x[s*dim:s*dim+dim], w.RMSFFNWeight(l))
		}
		w.W1(l).BatchMatMul(hb, xb, n)
		w.W3(l).BatchMatMul(hb2, xb, n)
		tensor.SiLU(hb)
		for i := range hb {
			hb[i] *= hb2[i]
		}
		w.W2(l).BatchMatMul(xb, hb, n)
		for i := range x {
			x[i] += xb[i]
		}
	}

	for s := range reqs {
		tensor.RMSNorm(x[s*dim:s*dim+dim], x[s*dim:s*dim+dim], w.RMSFinalWeight())
	}

	logitsFlat := make([]float32, n*cfg.VocabSize)
	w.Classifier().BatchMatMul(logitsFlat, x, n)

	out := make([][]float32, n)
	for s := range reqs {
		out[s] = logitsFlat[s*cfg.VocabSize : s*cfg.VocabSize+cfg.VocabSize]
	}
	return out
}
