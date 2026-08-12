package model

import (
	"testing"

	"inferlab/internal/loader"
)

// fixtureDims deliberately sets n_heads=2, n_kv_heads=1 to exercise grouped-
// query attention's head-sharing math, while staying small enough that the
// forward pass can be independently cross-checked against an external
// reference implementation (see the golden-value test below).
type fixtureDims struct {
	dim, hiddenDim, nLayers, nHeads, nKVHeads, vocabSize, seqLen int
}

var dims = fixtureDims{dim: 4, hiddenDim: 4, nLayers: 1, nHeads: 2, nKVHeads: 1, vocabSize: 3, seqLen: 4}

// gen fills a tensor with a small deterministic sequence, restarting at
// index 0 for every tensor. Matches, term for term, the weight-generation
// formula used in the independent NumPy reference script
// (model_reference.py) that produced this test's golden logits — so the two
// implementations are computing over identical weights.
func gen(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i%7)-3) * 0.1
	}
	return out
}

func fixtureCheckpoint() *loader.Checkpoint {
	d := dims
	kvDim := d.dim * d.nKVHeads / d.nHeads
	cfg := loader.Config{
		Dim: d.dim, HiddenDim: d.hiddenDim, NLayers: d.nLayers,
		NHeads: d.nHeads, NKVHeads: d.nKVHeads, VocabSize: d.vocabSize, SeqLen: d.seqLen,
		SharedWeights: true,
	}
	tokenEmbedding := gen(d.vocabSize * d.dim)
	w := loader.Weights{
		TokenEmbedding: tokenEmbedding,
		RMSAttWeight:   gen(d.nLayers * d.dim),
		WQ:             gen(d.nLayers * d.dim * d.dim),
		WK:             gen(d.nLayers * d.dim * kvDim),
		WV:             gen(d.nLayers * d.dim * kvDim),
		WO:             gen(d.nLayers * d.dim * d.dim),
		RMSFFNWeight:   gen(d.nLayers * d.dim),
		W1:             gen(d.nLayers * d.hiddenDim * d.dim),
		W2:             gen(d.nLayers * d.dim * d.hiddenDim),
		W3:             gen(d.nLayers * d.hiddenDim * d.dim),
		RMSFinalWeight: gen(d.dim),
		WCLS:           tokenEmbedding,
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

// TestGoldenLogitsMatchIndependentReference cross-checks Step's output
// against model_reference.py, a from-scratch NumPy reimplementation of the
// same RMSNorm/RoPE/GQA-attention/SwiGLU math over identical weights (see
// gen's doc comment) run independently of this codebase. This is the
// project's highest-value correctness gate: a bug in the transformer math
// would otherwise only show up as "the real model's output looks like
// garbage," with no way to localize which stage broke it.
func TestGoldenLogitsMatchIndependentReference(t *testing.T) {
	ck := fixtureCheckpoint()
	m := New(ck)

	logits := m.ForwardSequence([]int32{1, 0})

	want := []float32{-0.1902301013469696, 0.10815493762493134, -0.11563384532928467}
	if len(logits) != len(want) {
		t.Fatalf("logits len = %d, want %d", len(logits), len(want))
	}
	for i := range want {
		if !approxEqual(logits[i], want[i], 1e-4) {
			t.Fatalf("logits[%d] = %v, want %v (independent reference)", i, logits[i], want[i])
		}
	}
}

func TestRoPEIdentityAtPositionZero(t *testing.T) {
	m := &Model{}
	m.ropeCos, m.ropeSin = precomputeRoPE(4, 4) // seqLen=4, headSize=4 -> 1 head-chunk, 2 pairs
	vec := []float32{1, 2, 3, 4}
	orig := append([]float32(nil), vec...)
	m.applyRoPE(vec, 4, 0)
	for i := range vec {
		if !approxEqual(vec[i], orig[i], 1e-6) {
			t.Fatalf("applyRoPE at pos=0 changed vec[%d]: %v -> %v, want identity", i, orig[i], vec[i])
		}
	}
}

func TestRoPENonIdentityAtLaterPosition(t *testing.T) {
	m := &Model{}
	m.ropeCos, m.ropeSin = precomputeRoPE(4, 4)
	vec := []float32{1, 2, 3, 4}
	orig := append([]float32(nil), vec...)
	m.applyRoPE(vec, 4, 1)
	same := true
	for i := range vec {
		if !approxEqual(vec[i], orig[i], 1e-6) {
			same = false
		}
	}
	if same {
		t.Fatalf("applyRoPE at pos=1 left vec unchanged, expected a real rotation")
	}
}

// TestCausalMaskingIgnoresFutureCache is the cheapest-to-break, most
// important transformer correctness property: logits at position t must
// depend only on cache entries at positions <= t. It poisons every cache
// slot beyond the position under test with extreme values before Step ever
// legitimately writes them, and checks the computed logits are unaffected —
// a regression here (e.g. an off-by-one in the attention loop's upper
// bound) would otherwise silently leak future tokens into past predictions.
func TestCausalMaskingIgnoresFutureCache(t *testing.T) {
	ck := fixtureCheckpoint()
	m := New(ck)
	kvDim := ck.Config.KVDim()
	target := 1
	tokens := []int32{2, 1, 0}

	kc1, vc1 := m.NewCacheBuffers()
	var clean []float32
	for pos := 0; pos <= target; pos++ {
		clean = m.Step(tokens[pos], pos, kc1, vc1)
	}

	kc2, vc2 := m.NewCacheBuffers()
	for l := 0; l < ck.Config.NLayers; l++ {
		for pos := target + 1; pos < ck.Config.SeqLen; pos++ {
			for i := 0; i < kvDim; i++ {
				kc2[l][pos*kvDim+i] = 1e6
				vc2[l][pos*kvDim+i] = 1e6
			}
		}
	}
	var poisoned []float32
	for pos := 0; pos <= target; pos++ {
		poisoned = m.Step(tokens[pos], pos, kc2, vc2)
	}

	for i := range clean {
		if clean[i] != poisoned[i] {
			t.Fatalf("logits at position %d differ when future cache slots are poisoned (index %d: %v vs %v) — attention is reading beyond the causal boundary",
				target, i, clean[i], poisoned[i])
		}
	}
}

func TestForwardSequenceDeterministic(t *testing.T) {
	ck := fixtureCheckpoint()
	m := New(ck)
	tokens := []int32{1, 2, 0}
	a := m.ForwardSequence(tokens)
	b := m.ForwardSequence(tokens)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("ForwardSequence not deterministic at index %d: %v vs %v", i, a[i], b[i])
		}
	}
}

// TestStepBatchMatchesSequentialStep is internal/batch's foundational
// correctness property, tested here since it's really a property of
// StepBatch's math: batching several independent sequences' single-token
// decode steps into one tick must produce exactly the same per-sequence
// logits as stepping each sequence individually — batching changes only
// performance, never results. Since BatchMatMul is itself just N
// independent per-row dot products (see tensor.BatchMatMul's doc comment),
// and attention is an explicit per-request loop in both StepBatch and Step,
// there is no floating-point reassociation between the two paths, so the
// two must agree bit-for-bit, not just within tolerance.
func TestStepBatchMatchesSequentialStep(t *testing.T) {
	ck := fixtureCheckpoint()
	m := New(ck)

	seqs := [][]int32{
		{1, 2, 0},
		{0, 0, 1},
		{2, 1, 2},
	}

	sequentialCaches := make([][][]float32, len(seqs))
	sequentialValCaches := make([][][]float32, len(seqs))
	batchCaches := make([][][]float32, len(seqs))
	batchValCaches := make([][][]float32, len(seqs))
	for i := range seqs {
		sequentialCaches[i], sequentialValCaches[i] = m.NewCacheBuffers()
		batchCaches[i], batchValCaches[i] = m.NewCacheBuffers()
	}

	for step := 0; step < 3; step++ {
		var sequential [][]float32
		for i, s := range seqs {
			sequential = append(sequential, m.Step(s[step], step, sequentialCaches[i], sequentialValCaches[i]))
		}

		reqs := make([]BatchRequest, len(seqs))
		for i, s := range seqs {
			reqs[i] = BatchRequest{Token: s[step], Pos: step, KeyCache: batchCaches[i], ValCache: batchValCaches[i]}
		}
		batched := m.StepBatch(reqs)

		for i := range seqs {
			for j := range sequential[i] {
				if sequential[i][j] != batched[i][j] {
					t.Fatalf("step %d, sequence %d: StepBatch diverges from sequential Step at logit %d: %v vs %v",
						step, i, j, batched[i][j], sequential[i][j])
				}
			}
		}
	}
}

func TestStepBatchEmptyReturnsNil(t *testing.T) {
	ck := fixtureCheckpoint()
	m := New(ck)
	if got := m.StepBatch(nil); got != nil {
		t.Fatalf("StepBatch(nil) = %v, want nil", got)
	}
}

func TestForwardSequenceSensitiveToInputToken(t *testing.T) {
	ck := fixtureCheckpoint()
	m := New(ck)
	a := m.ForwardSequence([]int32{1})
	b := m.ForwardSequence([]int32{2})
	same := true
	for i := range a {
		if !approxEqual(a[i], b[i], 1e-6) {
			same = false
		}
	}
	if same {
		t.Fatalf("different first tokens produced identical logits: %v", a)
	}
}
