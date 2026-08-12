// Package loader parses llama2.c's checkpoint file format: a raw 7-int32
// Config header (no magic number, no version field — this is a fragile,
// non-self-describing format compared to something like GGUF, which is a
// deliberate simplicity trade-off for this project's scope, documented in
// the README) followed by fp32 weight tensors in a fixed order.
package loader

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// Config mirrors llama2.c's Config struct. A negative VocabSize in the raw
// file signals unshared classifier weights (a separate output projection
// distinct from the token embedding table); Load normalizes VocabSize to its
// absolute value and records that fact in SharedWeights.
type Config struct {
	Dim           int
	HiddenDim     int
	NLayers       int
	NHeads        int
	NKVHeads      int
	VocabSize     int
	SeqLen        int
	SharedWeights bool
}

// HeadSize returns the per-attention-head dimension.
func (c Config) HeadSize() int { return c.Dim / c.NHeads }

// KVDim returns the total dimension of the key/value projections, which can
// be smaller than Dim under grouped-query attention (NKVHeads < NHeads).
func (c Config) KVDim() int { return c.Dim * c.NKVHeads / c.NHeads }

// Weights holds every tensor from the checkpoint as flat, row-major
// []float32 slices, one layer's worth concatenated after another in the
// per-layer fields. Per-layer slice boundaries are computed by callers
// (internal/model) using Config's dimensions, matching llama2.c's own
// pointer-arithmetic indexing style.
type Weights struct {
	TokenEmbedding []float32 // (vocab_size, dim)
	RMSAttWeight   []float32 // (n_layers, dim)
	WQ             []float32 // (n_layers, dim, n_heads*head_size)
	WK             []float32 // (n_layers, dim, n_kv_heads*head_size)
	WV             []float32 // (n_layers, dim, n_kv_heads*head_size)
	WO             []float32 // (n_layers, n_heads*head_size, dim)
	RMSFFNWeight   []float32 // (n_layers, dim)
	W1             []float32 // (n_layers, hidden_dim, dim) — gate projection
	W2             []float32 // (n_layers, dim, hidden_dim) — down projection
	W3             []float32 // (n_layers, hidden_dim, dim) — up projection
	RMSFinalWeight []float32 // (dim)
	WCLS           []float32 // (vocab_size, dim); == TokenEmbedding if SharedWeights
}

// Checkpoint is a fully-parsed model: its architecture Config plus Weights.
type Checkpoint struct {
	Config  Config
	Weights Weights
}

const configHeaderBytes = 7 * 4

// Load reads a full llama2.c-format checkpoint from r.
func Load(r io.Reader) (*Checkpoint, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("loader: read checkpoint: %w", err)
	}
	if len(data) < configHeaderBytes {
		return nil, fmt.Errorf("loader: checkpoint too short for config header: got %d bytes, need at least %d", len(data), configHeaderBytes)
	}

	raw := make([]int32, 7)
	for i := range raw {
		raw[i] = int32(binary.LittleEndian.Uint32(data[i*4:]))
	}
	vocabRaw := raw[5]
	vocabSize := vocabRaw
	if vocabSize < 0 {
		vocabSize = -vocabSize
	}
	cfg := Config{
		Dim:           int(raw[0]),
		HiddenDim:     int(raw[1]),
		NLayers:       int(raw[2]),
		NHeads:        int(raw[3]),
		NKVHeads:      int(raw[4]),
		VocabSize:     int(vocabSize),
		SeqLen:        int(raw[6]),
		SharedWeights: vocabRaw > 0,
	}
	if cfg.Dim <= 0 || cfg.HiddenDim <= 0 || cfg.NLayers <= 0 || cfg.NHeads <= 0 ||
		cfg.NKVHeads <= 0 || cfg.VocabSize <= 0 || cfg.SeqLen <= 0 {
		return nil, fmt.Errorf("loader: invalid config header (all fields must be positive, vocab_size sign aside): %+v", cfg)
	}
	if cfg.Dim%cfg.NHeads != 0 {
		return nil, fmt.Errorf("loader: dim %d not evenly divisible by n_heads %d", cfg.Dim, cfg.NHeads)
	}
	if cfg.NHeads%cfg.NKVHeads != 0 {
		return nil, fmt.Errorf("loader: n_heads %d not evenly divisible by n_kv_heads %d", cfg.NHeads, cfg.NKVHeads)
	}

	body := data[configHeaderBytes:]
	off := 0
	readF := func(n int) ([]float32, error) {
		need := n * 4
		if off+need > len(body) {
			return nil, fmt.Errorf("loader: checkpoint truncated: need %d more bytes at body offset %d, have %d remaining", need, off, len(body)-off)
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[off+i*4:]))
		}
		off += need
		return out, nil
	}

	headSize := cfg.HeadSize()
	var w Weights
	steps := []struct {
		dst *[]float32
		n   int
	}{
		{&w.TokenEmbedding, cfg.VocabSize * cfg.Dim},
		{&w.RMSAttWeight, cfg.NLayers * cfg.Dim},
		{&w.WQ, cfg.NLayers * cfg.Dim * cfg.NHeads * headSize},
		{&w.WK, cfg.NLayers * cfg.Dim * cfg.NKVHeads * headSize},
		{&w.WV, cfg.NLayers * cfg.Dim * cfg.NKVHeads * headSize},
		{&w.WO, cfg.NLayers * cfg.NHeads * headSize * cfg.Dim},
		{&w.RMSFFNWeight, cfg.NLayers * cfg.Dim},
		{&w.W1, cfg.NLayers * cfg.HiddenDim * cfg.Dim},
		{&w.W2, cfg.NLayers * cfg.Dim * cfg.HiddenDim},
		{&w.W3, cfg.NLayers * cfg.HiddenDim * cfg.Dim},
		{&w.RMSFinalWeight, cfg.Dim},
	}
	for _, s := range steps {
		v, err := readF(s.n)
		if err != nil {
			return nil, err
		}
		*s.dst = v
	}

	// Skip the legacy freq_cis_real/freq_cis_imag RoPE tables the exporter
	// still writes for backward compatibility — internal/model computes
	// RoPE frequencies itself rather than reading them from the checkpoint.
	if _, err := readF(cfg.SeqLen * headSize / 2); err != nil {
		return nil, err
	}
	if _, err := readF(cfg.SeqLen * headSize / 2); err != nil {
		return nil, err
	}

	if cfg.SharedWeights {
		w.WCLS = w.TokenEmbedding
	} else {
		v, err := readF(cfg.VocabSize * cfg.Dim)
		if err != nil {
			return nil, err
		}
		w.WCLS = v
	}

	return &Checkpoint{Config: cfg, Weights: w}, nil
}
