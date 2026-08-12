// Package quant applies int8 weight-only quantization to a loaded
// checkpoint and exposes the result as a model.Weights implementation, so
// internal/model's Step runs unmodified over either precision. RMSNorm
// weights are kept fp32 — they're small elementwise scale vectors, not
// matmul weights, so quantizing them would save little memory while adding
// error to every layer's normalization. Everything else, including the
// token embedding table (which is roughly half of all parameters at
// TinyStories scale — see Quantize's doc comment), is quantized per-row.
package quant

import (
	"inferlab/internal/loader"
	"inferlab/internal/model"
	"inferlab/internal/tensor"
)

// Weights is an int8 weight-only quantized copy of a loaded checkpoint.
type Weights struct {
	cfg loader.Config

	embedding  *tensor.QuantizedTensor // (vocab_size, dim)
	classifier *tensor.QuantizedTensor // (vocab_size, dim); == embedding if the checkpoint shares them

	rmsAtt   []float32 // (n_layers, dim), fp32
	rmsFFN   []float32 // (n_layers, dim), fp32
	rmsFinal []float32 // (dim,), fp32

	wq, wk, wv, wo, w1, w2, w3 []*tensor.QuantizedTensor // one entry per layer
}

// Quantize builds a Weights from a fully-loaded dense checkpoint. At
// TinyStories scale the token embedding table (vocab_size * dim) is a much
// larger share of total parameters than in a full-size LLM — e.g. for
// stories15M's 32000-word vocabulary and dim=288, the embedding table alone
// is roughly 9.2M of the model's ~15M parameters — so quantizing only the
// attention/FFN matrices would miss most of the memory win this package
// exists to deliver.
func Quantize(ck *loader.Checkpoint) *Weights {
	cfg := ck.Config
	w := ck.Weights
	dim, hiddenDim, kvDim := cfg.Dim, cfg.HiddenDim, cfg.KVDim()

	qw := &Weights{
		cfg:      cfg,
		rmsAtt:   w.RMSAttWeight,
		rmsFFN:   w.RMSFFNWeight,
		rmsFinal: w.RMSFinalWeight,
	}

	qw.embedding = tensor.Quantize(w.TokenEmbedding, cfg.VocabSize, dim)
	if cfg.SharedWeights {
		qw.classifier = qw.embedding
	} else {
		qw.classifier = tensor.Quantize(w.WCLS, cfg.VocabSize, dim)
	}

	for l := 0; l < cfg.NLayers; l++ {
		qw.wq = append(qw.wq, tensor.Quantize(w.WQ[l*dim*dim:(l+1)*dim*dim], dim, dim))
		qw.wk = append(qw.wk, tensor.Quantize(w.WK[l*dim*kvDim:(l+1)*dim*kvDim], kvDim, dim))
		qw.wv = append(qw.wv, tensor.Quantize(w.WV[l*dim*kvDim:(l+1)*dim*kvDim], kvDim, dim))
		qw.wo = append(qw.wo, tensor.Quantize(w.WO[l*dim*dim:(l+1)*dim*dim], dim, dim))
		qw.w1 = append(qw.w1, tensor.Quantize(w.W1[l*hiddenDim*dim:(l+1)*hiddenDim*dim], hiddenDim, dim))
		qw.w2 = append(qw.w2, tensor.Quantize(w.W2[l*dim*hiddenDim:(l+1)*dim*hiddenDim], dim, hiddenDim))
		qw.w3 = append(qw.w3, tensor.Quantize(w.W3[l*hiddenDim*dim:(l+1)*hiddenDim*dim], hiddenDim, dim))
	}
	return qw
}

// ByteSize returns the total bytes occupied by this Weights' int8 tensors
// and per-row fp32 scales (excluding the fp32 RMSNorm vectors, which are
// negligible in size). Used by the benchmark endpoint to report the actual
// measured memory difference between precisions.
func (w *Weights) ByteSize() int {
	total := 0
	add := func(q *tensor.QuantizedTensor) {
		total += len(q.Data) + len(q.Scales)*4
	}
	add(w.embedding)
	if w.classifier != w.embedding {
		add(w.classifier)
	}
	for _, layerSet := range [][]*tensor.QuantizedTensor{w.wq, w.wk, w.wv, w.wo, w.w1, w.w2, w.w3} {
		for _, q := range layerSet {
			add(q)
		}
	}
	return total
}

type quantLinear struct {
	q    *tensor.QuantizedTensor
	n, d int
}

func (l quantLinear) MatMul(out, x []float32) { tensor.QuantMatMul(out, x, l.q, l.n, l.d) }
func (l quantLinear) BatchMatMul(out, x []float32, nSeqs int) {
	tensor.QuantBatchMatMul(out, x, l.q, nSeqs, l.n, l.d)
}

func (w *Weights) TokenEmbeddingRow(token int32) []float32 {
	return w.embedding.DequantizeRow(int(token))
}
func (w *Weights) RMSAttWeight(l int) []float32 {
	dim := w.cfg.Dim
	return w.rmsAtt[l*dim : l*dim+dim]
}
func (w *Weights) RMSFFNWeight(l int) []float32 {
	dim := w.cfg.Dim
	return w.rmsFFN[l*dim : l*dim+dim]
}
func (w *Weights) RMSFinalWeight() []float32 { return w.rmsFinal }

func (w *Weights) WQ(l int) model.Linear { return quantLinear{w.wq[l], w.cfg.Dim, w.cfg.Dim} }
func (w *Weights) WK(l int) model.Linear { return quantLinear{w.wk[l], w.cfg.Dim, w.cfg.KVDim()} }
func (w *Weights) WV(l int) model.Linear { return quantLinear{w.wv[l], w.cfg.Dim, w.cfg.KVDim()} }
func (w *Weights) WO(l int) model.Linear { return quantLinear{w.wo[l], w.cfg.Dim, w.cfg.Dim} }
func (w *Weights) W1(l int) model.Linear { return quantLinear{w.w1[l], w.cfg.Dim, w.cfg.HiddenDim} }
func (w *Weights) W2(l int) model.Linear { return quantLinear{w.w2[l], w.cfg.HiddenDim, w.cfg.Dim} }
func (w *Weights) W3(l int) model.Linear { return quantLinear{w.w3[l], w.cfg.Dim, w.cfg.HiddenDim} }
func (w *Weights) Classifier() model.Linear {
	return quantLinear{w.classifier, w.cfg.Dim, w.cfg.VocabSize}
}
