package loader

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// fixtureDims is a tiny architecture used across this package's tests: small
// enough to hand-verify by eye, with n_heads != n_kv_heads to exercise the
// grouped-query-attention dimension math.
type fixtureDims struct {
	dim, hiddenDim, nLayers, nHeads, nKVHeads, vocabSize, seqLen int
}

var dims = fixtureDims{dim: 8, hiddenDim: 16, nLayers: 2, nHeads: 2, nKVHeads: 1, vocabSize: 6, seqLen: 8}

func appendI32(buf *bytes.Buffer, v int32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(v))
	buf.Write(b[:])
}

func appendF32Seq(buf *bytes.Buffer, n int, start float32) {
	for i := 0; i < n; i++ {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(start+float32(i)))
		buf.Write(b[:])
	}
}

// buildFixtureBytes serializes a checkpoint with the given dims and
// shared-weights flag, filling every tensor with a distinct, predictable
// float sequence so parsed values can be checked against expectations.
func buildFixtureBytes(d fixtureDims, shared bool) []byte {
	buf := new(bytes.Buffer)
	vocab := int32(d.vocabSize)
	if !shared {
		vocab = -vocab
	}
	appendI32(buf, int32(d.dim))
	appendI32(buf, int32(d.hiddenDim))
	appendI32(buf, int32(d.nLayers))
	appendI32(buf, int32(d.nHeads))
	appendI32(buf, int32(d.nKVHeads))
	appendI32(buf, vocab)
	appendI32(buf, int32(d.seqLen))

	headSize := d.dim / d.nHeads
	sizes := []int{
		d.vocabSize * d.dim,                       // token embedding
		d.nLayers * d.dim,                         // rms att
		d.nLayers * d.dim * d.nHeads * headSize,   // wq
		d.nLayers * d.dim * d.nKVHeads * headSize, // wk
		d.nLayers * d.dim * d.nKVHeads * headSize, // wv
		d.nLayers * d.nHeads * headSize * d.dim,   // wo
		d.nLayers * d.dim,                         // rms ffn
		d.nLayers * d.hiddenDim * d.dim,           // w1
		d.nLayers * d.dim * d.hiddenDim,           // w2
		d.nLayers * d.hiddenDim * d.dim,           // w3
		d.dim,                                     // rms final
		d.seqLen * headSize / 2,                   // freq_cis_real (skipped on read)
		d.seqLen * headSize / 2,                   // freq_cis_imag (skipped on read)
	}
	if !shared {
		sizes = append(sizes, d.vocabSize*d.dim) // wcls
	}
	for i, n := range sizes {
		appendF32Seq(buf, n, float32(i)*1000) // each tensor's values start at a distinct offset
	}
	return buf.Bytes()
}

func TestLoadSharedWeights(t *testing.T) {
	raw := buildFixtureBytes(dims, true)
	ck, err := Load(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if ck.Config.Dim != dims.dim || ck.Config.HiddenDim != dims.hiddenDim ||
		ck.Config.NLayers != dims.nLayers || ck.Config.NHeads != dims.nHeads ||
		ck.Config.NKVHeads != dims.nKVHeads || ck.Config.VocabSize != dims.vocabSize ||
		ck.Config.SeqLen != dims.seqLen {
		t.Fatalf("Config mismatch: got %+v", ck.Config)
	}
	if !ck.Config.SharedWeights {
		t.Fatalf("expected SharedWeights=true for positive vocab_size")
	}
	if len(ck.Weights.TokenEmbedding) != dims.vocabSize*dims.dim {
		t.Fatalf("TokenEmbedding len = %d, want %d", len(ck.Weights.TokenEmbedding), dims.vocabSize*dims.dim)
	}
	if ck.Weights.TokenEmbedding[0] != 0 {
		t.Fatalf("TokenEmbedding[0] = %v, want 0 (first tensor starts at offset 0)", ck.Weights.TokenEmbedding[0])
	}
	// shared weights: WCLS must be the same slice as TokenEmbedding
	if &ck.Weights.WCLS[0] != &ck.Weights.TokenEmbedding[0] {
		t.Fatalf("expected WCLS to alias TokenEmbedding when SharedWeights=true")
	}
}

func TestLoadUnsharedWeights(t *testing.T) {
	raw := buildFixtureBytes(dims, false)
	ck, err := Load(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if ck.Config.SharedWeights {
		t.Fatalf("expected SharedWeights=false for negative vocab_size")
	}
	if ck.Config.VocabSize != dims.vocabSize {
		t.Fatalf("VocabSize = %d, want %d (sign should be stripped)", ck.Config.VocabSize, dims.vocabSize)
	}
	wantLen := dims.vocabSize * dims.dim
	if len(ck.Weights.WCLS) != wantLen {
		t.Fatalf("WCLS len = %d, want %d", len(ck.Weights.WCLS), wantLen)
	}
	// unshared: WCLS is its own tensor, distinct from TokenEmbedding
	if ck.Weights.WCLS[0] == ck.Weights.TokenEmbedding[0] {
		t.Fatalf("WCLS and TokenEmbedding unexpectedly identical for unshared weights")
	}
}

func TestLoadTruncatedInputErrors(t *testing.T) {
	raw := buildFixtureBytes(dims, true)
	truncated := raw[:len(raw)-100]
	if _, err := Load(bytes.NewReader(truncated)); err == nil {
		t.Fatalf("Load() on truncated input: got nil error, want error")
	}
}

func TestLoadShortHeaderErrors(t *testing.T) {
	if _, err := Load(bytes.NewReader([]byte{1, 2, 3})); err == nil {
		t.Fatalf("Load() on too-short header: got nil error, want error")
	}
}

func TestLoadInvalidDimNotDivisibleByHeadsErrors(t *testing.T) {
	bad := dims
	bad.dim = 7 // not divisible by nHeads=2
	raw := buildFixtureBytes(bad, true)
	if _, err := Load(bytes.NewReader(raw)); err == nil {
		t.Fatalf("Load() with dim not divisible by n_heads: got nil error, want error")
	}
}
