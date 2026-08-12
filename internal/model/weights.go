package model

import (
	"inferlab/internal/loader"
	"inferlab/internal/tensor"
)

// Linear performs y = W @ x for one fixed-shape weight matrix, independent
// of the underlying numeric representation. Step calls MatMul exactly once
// per projection per layer — at that granularity (O(layers) calls, each
// doing O(dim*hiddenDim)-ish work) interface dispatch overhead is
// negligible, which is what makes an interface here a reasonable choice
// instead of duplicating Step's ~150 lines of control flow once per
// precision. BatchMatMul is the same operation applied independently across
// nSeqs stacked rows of x, used by StepBatch (see batch_step.go) for the
// position-independent projections — the actual memory-bandwidth win
// internal/batch's scheduler exists to deliver.
type Linear interface {
	MatMul(out, x []float32)
	BatchMatMul(out, x []float32, nSeqs int)
}

// Weights is everything Step needs from a loaded model, independent of how
// the underlying tensors are stored. denseWeights (below, dense fp32) and
// quant.Weights (int8 weight-only) both implement it, so the exact same
// forward-pass code in Step runs either precision unmodified.
type Weights interface {
	TokenEmbeddingRow(token int32) []float32
	RMSAttWeight(layer int) []float32
	RMSFFNWeight(layer int) []float32
	RMSFinalWeight() []float32
	WQ(layer int) Linear
	WK(layer int) Linear
	WV(layer int) Linear
	WO(layer int) Linear
	W1(layer int) Linear
	W2(layer int) Linear
	W3(layer int) Linear
	Classifier() Linear
}

type denseLinear struct {
	w    []float32
	n, d int
}

func (l denseLinear) MatMul(out, x []float32) { tensor.MatMul(out, x, l.w, l.n, l.d) }
func (l denseLinear) BatchMatMul(out, x []float32, nSeqs int) {
	tensor.BatchMatMul(out, x, l.w, nSeqs, l.n, l.d)
}

// denseWeights adapts loader.Weights' flat fp32 arrays to the Weights
// interface.
type denseWeights struct {
	cfg loader.Config
	w   loader.Weights
}

func (d denseWeights) TokenEmbeddingRow(token int32) []float32 {
	dim := d.cfg.Dim
	return d.w.TokenEmbedding[int(token)*dim : int(token)*dim+dim]
}
func (d denseWeights) RMSAttWeight(l int) []float32 {
	dim := d.cfg.Dim
	return d.w.RMSAttWeight[l*dim : l*dim+dim]
}
func (d denseWeights) RMSFFNWeight(l int) []float32 {
	dim := d.cfg.Dim
	return d.w.RMSFFNWeight[l*dim : l*dim+dim]
}
func (d denseWeights) RMSFinalWeight() []float32 { return d.w.RMSFinalWeight }

func (d denseWeights) WQ(l int) Linear {
	dim := d.cfg.Dim
	return denseLinear{d.w.WQ[l*dim*dim : (l+1)*dim*dim], dim, dim}
}
func (d denseWeights) WK(l int) Linear {
	dim, kvDim := d.cfg.Dim, d.cfg.KVDim()
	return denseLinear{d.w.WK[l*dim*kvDim : (l+1)*dim*kvDim], dim, kvDim}
}
func (d denseWeights) WV(l int) Linear {
	dim, kvDim := d.cfg.Dim, d.cfg.KVDim()
	return denseLinear{d.w.WV[l*dim*kvDim : (l+1)*dim*kvDim], dim, kvDim}
}
func (d denseWeights) WO(l int) Linear {
	dim := d.cfg.Dim
	return denseLinear{d.w.WO[l*dim*dim : (l+1)*dim*dim], dim, dim}
}
func (d denseWeights) W1(l int) Linear {
	dim, hiddenDim := d.cfg.Dim, d.cfg.HiddenDim
	return denseLinear{d.w.W1[l*hiddenDim*dim : (l+1)*hiddenDim*dim], dim, hiddenDim}
}
func (d denseWeights) W2(l int) Linear {
	dim, hiddenDim := d.cfg.Dim, d.cfg.HiddenDim
	return denseLinear{d.w.W2[l*dim*hiddenDim : (l+1)*dim*hiddenDim], hiddenDim, dim}
}
func (d denseWeights) W3(l int) Linear {
	dim, hiddenDim := d.cfg.Dim, d.cfg.HiddenDim
	return denseLinear{d.w.W3[l*hiddenDim*dim : (l+1)*hiddenDim*dim], dim, hiddenDim}
}
func (d denseWeights) Classifier() Linear {
	dim, vocab := d.cfg.Dim, d.cfg.VocabSize
	return denseLinear{d.w.WCLS, dim, vocab}
}
