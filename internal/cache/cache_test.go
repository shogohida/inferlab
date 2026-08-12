package cache

import (
	"testing"

	"inferlab/internal/loader"
	"inferlab/internal/model"
)

// fixtureCheckpoint mirrors internal/model's own test fixture (n_heads=2,
// n_kv_heads=1 to exercise GQA) — duplicated rather than shared across
// package boundaries, which is the simpler trade-off for one small helper.
func fixtureCheckpoint() *loader.Checkpoint {
	dim, hiddenDim, nLayers, nHeads, nKVHeads, vocabSize, seqLen := 4, 4, 2, 2, 1, 3, 6
	kvDim := dim * nKVHeads / nHeads
	gen := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32((i%7)-3) * 0.1
		}
		return out
	}
	cfg := loader.Config{
		Dim: dim, HiddenDim: hiddenDim, NLayers: nLayers,
		NHeads: nHeads, NKVHeads: nKVHeads, VocabSize: vocabSize, SeqLen: seqLen,
		SharedWeights: true,
	}
	tokenEmbedding := gen(vocabSize * dim)
	w := loader.Weights{
		TokenEmbedding: tokenEmbedding,
		RMSAttWeight:   gen(nLayers * dim),
		WQ:             gen(nLayers * dim * dim),
		WK:             gen(nLayers * dim * kvDim),
		WV:             gen(nLayers * dim * kvDim),
		WO:             gen(nLayers * dim * dim),
		RMSFFNWeight:   gen(nLayers * dim),
		W1:             gen(nLayers * hiddenDim * dim),
		W2:             gen(nLayers * dim * hiddenDim),
		W3:             gen(nLayers * hiddenDim * dim),
		RMSFinalWeight: gen(dim),
		WCLS:           tokenEmbedding,
	}
	return &loader.Checkpoint{Config: cfg, Weights: w}
}

// TestCachedDecodeMatchesUncachedRecompute is the project's "money" test:
// generating token-by-token with a persistent KVCache must produce exactly
// the logits a full from-scratch recompute of the same growing prefix
// would — the entire point of caching is that it changes performance, not
// results.
func TestCachedDecodeMatchesUncachedRecompute(t *testing.T) {
	ck := fixtureCheckpoint()
	m := model.New(ck)
	tokens := []int32{1, 2, 0, 1, 2}

	kv := New(ck.Config.NLayers, ck.Config.KVDim(), ck.Config.SeqLen)
	for i, tok := range tokens {
		pos, err := kv.Reserve()
		if err != nil {
			t.Fatalf("Reserve() at step %d: %v", i, err)
		}
		if pos != i {
			t.Fatalf("Reserve() at step %d returned pos=%d, want %d", i, pos, i)
		}
		cached := m.Step(tok, pos, kv.Keys, kv.Values)
		uncached := m.ForwardSequence(tokens[:i+1])

		if len(cached) != len(uncached) {
			t.Fatalf("step %d: logits length mismatch: cached=%d uncached=%d", i, len(cached), len(uncached))
		}
		for j := range cached {
			if cached[j] != uncached[j] {
				t.Fatalf("step %d: cached decode diverges from uncached recompute at logit %d: %v vs %v",
					i, j, cached[j], uncached[j])
			}
		}
	}
}

func TestTwoSequencesDoNotCrossContaminate(t *testing.T) {
	ck := fixtureCheckpoint()
	m := model.New(ck)

	kvA := New(ck.Config.NLayers, ck.Config.KVDim(), ck.Config.SeqLen)
	kvB := New(ck.Config.NLayers, ck.Config.KVDim(), ck.Config.SeqLen)

	tokensA := []int32{1, 1, 1}
	tokensB := []int32{2, 0, 2}

	var lastA, lastB []float32
	for i := 0; i < 3; i++ {
		posA, _ := kvA.Reserve()
		lastA = m.Step(tokensA[i], posA, kvA.Keys, kvA.Values)
		posB, _ := kvB.Reserve()
		lastB = m.Step(tokensB[i], posB, kvB.Keys, kvB.Values)
	}

	wantA := m.ForwardSequence(tokensA)
	wantB := m.ForwardSequence(tokensB)
	for i := range wantA {
		if lastA[i] != wantA[i] {
			t.Fatalf("sequence A contaminated by sequence B's cache at logit %d: %v vs %v", i, lastA[i], wantA[i])
		}
		if lastB[i] != wantB[i] {
			t.Fatalf("sequence B contaminated by sequence A's cache at logit %d: %v vs %v", i, lastB[i], wantB[i])
		}
	}
}

func TestReserveErrorsAtMaxSeqLen(t *testing.T) {
	kv := New(1, 4, 2) // seqLen=2
	if _, err := kv.Reserve(); err != nil {
		t.Fatalf("Reserve() 1st call: unexpected error: %v", err)
	}
	if _, err := kv.Reserve(); err != nil {
		t.Fatalf("Reserve() 2nd call: unexpected error: %v", err)
	}
	if _, err := kv.Reserve(); err == nil {
		t.Fatalf("Reserve() 3rd call (beyond max_seq_len=2): got nil error, want error")
	}
}

func TestEvictResetsLenNotBuffers(t *testing.T) {
	kv := New(1, 4, 4)
	pos, _ := kv.Reserve()
	kv.Keys[0][pos*4] = 42
	kv.Evict()
	if kv.Len != 0 {
		t.Fatalf("Len after Evict() = %d, want 0", kv.Len)
	}
	newPos, err := kv.Reserve()
	if err != nil {
		t.Fatalf("Reserve() after Evict(): unexpected error: %v", err)
	}
	if newPos != 0 {
		t.Fatalf("Reserve() after Evict() returned pos=%d, want 0", newPos)
	}
}
