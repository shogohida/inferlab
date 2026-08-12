// Package model implements the Llama-2-style transformer forward pass used
// by llama2.c-format checkpoints: RMSNorm, rotary positional embeddings
// (RoPE), grouped-query causal self-attention, and a SwiGLU feed-forward
// block. It has no notion of a persistent KV cache — Step takes plain
// key/value buffers and reads/writes exactly one position's worth of data,
// so internal/cache (long-lived buffers reused across generation steps) and
// ForwardSequence (fresh buffers, full recompute — the reference/"uncached"
// path) are both just different callers of the same primitive.
package model

import (
	"math"

	"inferlab/internal/loader"
	"inferlab/internal/tensor"
)

// Model is an immutable, loaded checkpoint plus its precomputed RoPE tables.
// It holds its weights behind the Weights interface so the same Step logic
// runs over either dense fp32 or int8 weight-only quantized tensors.
type Model struct {
	Config  loader.Config
	weights Weights

	ropeCos [][]float32 // [pos][headSize/2]
	ropeSin [][]float32
}

// New builds a Model directly from a parsed dense fp32 checkpoint.
func New(ck *loader.Checkpoint) *Model {
	return NewFromWeights(ck.Config, denseWeights{cfg: ck.Config, w: ck.Weights})
}

// NewFromWeights builds a Model over any Weights implementation — used by
// New (dense fp32) and by internal/quant (int8 weight-only) alike.
func NewFromWeights(cfg loader.Config, w Weights) *Model {
	m := &Model{Config: cfg, weights: w}
	m.ropeCos, m.ropeSin = precomputeRoPE(cfg.SeqLen, cfg.HeadSize())
	return m
}

func precomputeRoPE(seqLen, headSize int) (cosT, sinT [][]float32) {
	half := headSize / 2
	cosT = make([][]float32, seqLen)
	sinT = make([][]float32, seqLen)
	for pos := 0; pos < seqLen; pos++ {
		cosT[pos] = make([]float32, half)
		sinT[pos] = make([]float32, half)
		for i := 0; i < half; i++ {
			freq := float32(1.0 / math.Pow(10000, float64(2*i)/float64(headSize)))
			val := float32(pos) * freq
			cosT[pos][i] = float32(math.Cos(float64(val)))
			sinT[pos][i] = float32(math.Sin(float64(val)))
		}
	}
	return cosT, sinT
}

// applyRoPE rotates every headSize-sized head chunk of vec in place using
// position pos's precomputed frequency table. The same table applies to
// every head (query or key) because the rotation frequency depends only on
// the element's offset within its own head, not which head or how many
// heads vec spans — this is what lets a single table serve both the
// full-width query vector and the (possibly narrower, under GQA) key
// vector.
func (m *Model) applyRoPE(vec []float32, headSize, pos int) {
	half := headSize / 2
	cos := m.ropeCos[pos]
	sin := m.ropeSin[pos]
	for base := 0; base+headSize <= len(vec); base += headSize {
		for i := 0; i < half; i++ {
			fcr, fci := cos[i], sin[i]
			v0, v1 := vec[base+2*i], vec[base+2*i+1]
			vec[base+2*i] = v0*fcr - v1*fci
			vec[base+2*i+1] = v0*fci + v1*fcr
		}
	}
}

// Step computes the logits for a single new token at position pos, given
// key/value cache buffers indexed [layer][seqLen*kvDim] that already hold
// valid data for every position < pos. Step writes this position's K/V into
// those buffers before attending over [0, pos] — so cache reuse (or its
// absence) is entirely the caller's concern: a caller that keeps calling
// Step with the same buffers across increasing pos gets real KV-cache
// behavior "for free"; ForwardSequence below gets the uncached reference
// behavior by discarding and rebuilding the buffers on every call instead.
//
// The caller must ensure 0 <= pos < Config.SeqLen and that keyCache/valCache
// are sized [Config.NLayers][Config.SeqLen*Config.KVDim()] — Step trusts
// this the same way tensor.MatMul trusts its slice-length arguments; it is
// an internal invariant between tightly-coupled packages, not a public API
// boundary that needs its own validation.
func (m *Model) Step(token int32, pos int, keyCache, valCache [][]float32) []float32 {
	cfg := m.Config
	w := m.weights
	dim := cfg.Dim
	kvDim := cfg.KVDim()
	headSize := cfg.HeadSize()
	hiddenDim := cfg.HiddenDim
	kvMul := cfg.NHeads / cfg.NKVHeads

	x := make([]float32, dim)
	copy(x, w.TokenEmbeddingRow(token))

	xb := make([]float32, dim)
	xb2 := make([]float32, dim)
	q := make([]float32, dim)
	hb := make([]float32, hiddenDim)
	hb2 := make([]float32, hiddenDim)
	scores := make([]float32, cfg.SeqLen)
	invSqrtHead := float32(1.0 / math.Sqrt(float64(headSize)))

	for l := 0; l < cfg.NLayers; l++ {
		tensor.RMSNorm(xb, x, w.RMSAttWeight(l))

		w.WQ(l).MatMul(q, xb)

		k := keyCache[l][pos*kvDim : pos*kvDim+kvDim]
		v := valCache[l][pos*kvDim : pos*kvDim+kvDim]
		w.WK(l).MatMul(k, xb)
		w.WV(l).MatMul(v, xb)

		m.applyRoPE(q, headSize, pos)
		m.applyRoPE(k, headSize, pos)

		for h := 0; h < cfg.NHeads; h++ {
			qh := q[h*headSize : h*headSize+headSize]
			kvHead := h / kvMul
			active := scores[:pos+1]
			for t := 0; t <= pos; t++ {
				kt := keyCache[l][t*kvDim+kvHead*headSize : t*kvDim+kvHead*headSize+headSize]
				var dot float32
				for i := 0; i < headSize; i++ {
					dot += qh[i] * kt[i]
				}
				active[t] = dot * invSqrtHead
			}
			tensor.Softmax(active)

			out := xb2[h*headSize : h*headSize+headSize]
			for i := range out {
				out[i] = 0
			}
			for t := 0; t <= pos; t++ {
				vt := valCache[l][t*kvDim+kvHead*headSize : t*kvDim+kvHead*headSize+headSize]
				weight := active[t]
				for i := 0; i < headSize; i++ {
					out[i] += weight * vt[i]
				}
			}
		}

		w.WO(l).MatMul(xb, xb2)
		tensor.Add(x, x, xb)

		tensor.RMSNorm(xb, x, w.RMSFFNWeight(l))
		w.W1(l).MatMul(hb, xb)
		w.W3(l).MatMul(hb2, xb)
		tensor.SiLU(hb)
		for i := range hb {
			hb[i] *= hb2[i] // SwiGLU: silu(gate) * up
		}
		w.W2(l).MatMul(xb, hb)
		tensor.Add(x, x, xb)
	}

	tensor.RMSNorm(x, x, w.RMSFinalWeight())

	logits := make([]float32, cfg.VocabSize)
	w.Classifier().MatMul(logits, x)
	return logits
}

// NewCacheBuffers allocates zeroed key/value cache buffers sized for one
// full sequence: [Config.NLayers][Config.SeqLen * Config.KVDim()].
func (m *Model) NewCacheBuffers() (keyCache, valCache [][]float32) {
	kvDim := m.Config.KVDim()
	keyCache = make([][]float32, m.Config.NLayers)
	valCache = make([][]float32, m.Config.NLayers)
	for l := 0; l < m.Config.NLayers; l++ {
		keyCache[l] = make([]float32, m.Config.SeqLen*kvDim)
		valCache[l] = make([]float32, m.Config.SeqLen*kvDim)
	}
	return keyCache, valCache
}

// ForwardSequence recomputes the full prefix from scratch — fresh cache
// buffers, one Step call per position from 0 up to len(tokens)-1 — and
// returns the logits at the final position. This is the uncached reference
// path: functionally what Step already gives you if you never reuse its
// cache buffers across calls, made explicit as its own entry point because
// internal/cache's correctness test needs exactly this as its baseline.
func (m *Model) ForwardSequence(tokens []int32) []float32 {
	keyCache, valCache := m.NewCacheBuffers()
	var logits []float32
	for pos, tok := range tokens {
		logits = m.Step(tok, pos, keyCache, valCache)
	}
	return logits
}
