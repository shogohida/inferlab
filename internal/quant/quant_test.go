package quant

import (
	"testing"

	"inferlab/internal/loader"
	"inferlab/internal/model"
)

func fixtureCheckpoint(shared bool) *loader.Checkpoint {
	dim, hiddenDim, nLayers, nHeads, nKVHeads, vocabSize, seqLen := 6, 8, 2, 2, 1, 5, 6
	kvDim := dim * nKVHeads / nHeads
	gen := func(n int, offset int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(((i+offset)%11)-5) * 0.07
		}
		return out
	}
	cfg := loader.Config{
		Dim: dim, HiddenDim: hiddenDim, NLayers: nLayers,
		NHeads: nHeads, NKVHeads: nKVHeads, VocabSize: vocabSize, SeqLen: seqLen,
		SharedWeights: shared,
	}
	tokenEmbedding := gen(vocabSize*dim, 0)
	w := loader.Weights{
		TokenEmbedding: tokenEmbedding,
		RMSAttWeight:   gen(nLayers*dim, 1),
		WQ:             gen(nLayers*dim*dim, 2),
		WK:             gen(nLayers*dim*kvDim, 3),
		WV:             gen(nLayers*dim*kvDim, 4),
		WO:             gen(nLayers*dim*dim, 5),
		RMSFFNWeight:   gen(nLayers*dim, 6),
		W1:             gen(nLayers*hiddenDim*dim, 7),
		W2:             gen(nLayers*dim*hiddenDim, 8),
		W3:             gen(nLayers*hiddenDim*dim, 9),
		RMSFinalWeight: gen(dim, 10),
	}
	if shared {
		w.WCLS = tokenEmbedding
	} else {
		w.WCLS = gen(vocabSize*dim, 11)
	}
	return &loader.Checkpoint{Config: cfg, Weights: w}
}

func approxEqual(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

func TestQuantizedForwardCloseToDense(t *testing.T) {
	ck := fixtureCheckpoint(true)
	dense := model.New(ck)
	quantized := model.NewFromWeights(ck.Config, Quantize(ck))

	tokens := []int32{1, 3, 0, 4, 2}
	wantLogits := dense.ForwardSequence(tokens)
	gotLogits := quantized.ForwardSequence(tokens)

	if len(wantLogits) != len(gotLogits) {
		t.Fatalf("logits length mismatch: dense=%d quantized=%d", len(wantLogits), len(gotLogits))
	}
	for i := range wantLogits {
		// int8 per-row quantization over several layers accumulates some
		// error; this tolerance is loose enough to allow that while still
		// catching a badly broken quantization path (e.g. wrong dimension
		// ordering, which would produce wildly different logits, not
		// slightly different ones).
		tol := abs32(wantLogits[i])*0.25 + 0.25
		if !approxEqual(wantLogits[i], gotLogits[i], tol) {
			t.Fatalf("logit %d: dense=%v quantized=%v, exceeds tolerance %v", i, wantLogits[i], gotLogits[i], tol)
		}
	}
}

func TestTokenEmbeddingRowCloseToDense(t *testing.T) {
	ck := fixtureCheckpoint(true)
	qw := Quantize(ck)
	dim := ck.Config.Dim

	for token := 0; token < ck.Config.VocabSize; token++ {
		want := ck.Weights.TokenEmbedding[token*dim : token*dim+dim]
		got := qw.TokenEmbeddingRow(int32(token))
		for j := range want {
			tol := abs32(want[j])*0.05 + 0.02
			if !approxEqual(want[j], got[j], tol) {
				t.Fatalf("token %d dim %d: dense=%v quantized=%v, exceeds tolerance %v", token, j, want[j], got[j], tol)
			}
		}
	}
}

func TestSharedWeightsAliasClassifier(t *testing.T) {
	shared := Quantize(fixtureCheckpoint(true))
	if shared.classifier != shared.embedding {
		t.Fatalf("expected classifier to alias embedding when SharedWeights=true")
	}

	unshared := Quantize(fixtureCheckpoint(false))
	if unshared.classifier == unshared.embedding {
		t.Fatalf("expected classifier to be distinct from embedding when SharedWeights=false")
	}
}

func TestByteSizeSmallerThanFP32Equivalent(t *testing.T) {
	ck := fixtureCheckpoint(true)
	qw := Quantize(ck)

	dim, hiddenDim, nLayers, kvDim, vocab := ck.Config.Dim, ck.Config.HiddenDim, ck.Config.NLayers, ck.Config.KVDim(), ck.Config.VocabSize
	fp32Bytes := 4 * (vocab*dim + // embedding (classifier shared, not counted twice)
		nLayers*dim*dim + // wq
		nLayers*dim*kvDim + // wk
		nLayers*dim*kvDim + // wv
		nLayers*dim*dim + // wo
		nLayers*hiddenDim*dim + // w1
		nLayers*dim*hiddenDim + // w2
		nLayers*hiddenDim*dim) // w3

	quantBytes := qw.ByteSize()
	if quantBytes >= fp32Bytes {
		t.Fatalf("quantized ByteSize() = %d, want less than fp32-equivalent %d", quantBytes, fp32Bytes)
	}
	// int8 (1 byte) vs fp32 (4 bytes) per weight, plus a small per-row fp32
	// scale overhead: should land well under half the fp32 size.
	if quantBytes > fp32Bytes/2 {
		t.Fatalf("quantized ByteSize() = %d is not meaningfully smaller than fp32-equivalent %d", quantBytes, fp32Bytes)
	}
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
