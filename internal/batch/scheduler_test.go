package batch

import (
	"testing"

	"inferlab/internal/cache"
	"inferlab/internal/loader"
	"inferlab/internal/model"
)

func fixtureCheckpoint() *loader.Checkpoint {
	dim, hiddenDim, nLayers, nHeads, nKVHeads, vocabSize, seqLen := 4, 4, 2, 2, 1, 3, 8
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

func newKV(ck *loader.Checkpoint) *cache.KVCache {
	return cache.New(ck.Config.NLayers, ck.Config.KVDim(), ck.Config.SeqLen)
}

func TestAdmitCapacityAndDuplicateID(t *testing.T) {
	ck := fixtureCheckpoint()
	m := model.New(ck)
	s := NewScheduler(m, 2)

	if err := s.Admit(1, newKV(ck)); err != nil {
		t.Fatalf("Admit(1): unexpected error: %v", err)
	}
	if err := s.Admit(2, newKV(ck)); err != nil {
		t.Fatalf("Admit(2): unexpected error: %v", err)
	}
	if err := s.Admit(3, newKV(ck)); err == nil {
		t.Fatalf("Admit(3) at capacity: got nil error, want error")
	}
	if err := s.Admit(1, newKV(ck)); err == nil {
		t.Fatalf("Admit(1) duplicate id: got nil error, want error")
	}
	if got := s.Active(); got != 2 {
		t.Fatalf("Active() = %d, want 2", got)
	}
}

func TestTickErrorsOnMissingToken(t *testing.T) {
	ck := fixtureCheckpoint()
	m := model.New(ck)
	s := NewScheduler(m, 2)
	if err := s.Admit(1, newKV(ck)); err != nil {
		t.Fatalf("Admit: %v", err)
	}
	if _, err := s.Tick(map[int]int32{}); err == nil {
		t.Fatalf("Tick() with no token for active id 1: got nil error, want error")
	}
}

func TestTickOnEmptySchedulerReturnsNil(t *testing.T) {
	ck := fixtureCheckpoint()
	m := model.New(ck)
	s := NewScheduler(m, 2)
	got, err := s.Tick(map[int]int32{})
	if err != nil {
		t.Fatalf("Tick() on empty scheduler: unexpected error: %v", err)
	}
	if got != nil {
		t.Fatalf("Tick() on empty scheduler = %v, want nil", got)
	}
}

// TestTickMatchesSequentialStep exercises the equivalence property through
// the Scheduler's own API (Admit/Tick, using cache.KVCache.Reserve() under
// the hood) rather than calling model.StepBatch directly, since Tick's own
// position bookkeeping is exactly the part a scheduling bug would break.
func TestTickMatchesSequentialStep(t *testing.T) {
	ck := fixtureCheckpoint()
	m := model.New(ck)
	s := NewScheduler(m, 3)

	seqs := map[int][]int32{
		1: {1, 2, 0, 1},
		2: {0, 0, 1, 2},
		3: {2, 1, 2, 0},
	}
	kvs := map[int]*cache.KVCache{}
	for id := range seqs {
		kvs[id] = newKV(ck)
		if err := s.Admit(id, kvs[id]); err != nil {
			t.Fatalf("Admit(%d): %v", id, err)
		}
	}

	refCaches := map[int][][]float32{}
	refValCaches := map[int][][]float32{}
	for id := range seqs {
		k, v := m.NewCacheBuffers()
		refCaches[id] = k
		refValCaches[id] = v
	}

	for step := 0; step < 4; step++ {
		tokens := map[int]int32{}
		for id, s := range seqs {
			tokens[id] = s[step]
		}
		got, err := s.Tick(tokens)
		if err != nil {
			t.Fatalf("Tick() at step %d: %v", step, err)
		}
		for id := range seqs {
			want := m.Step(seqs[id][step], step, refCaches[id], refValCaches[id])
			for j := range want {
				if got[id][j] != want[j] {
					t.Fatalf("step %d id %d: Tick() diverges from sequential Step at logit %d: %v vs %v",
						step, id, j, got[id][j], want[j])
				}
			}
		}
	}
}

// TestEvictAdmitMidStreamDoesNotDisturbOthers is the "continuous" property:
// evicting a finished sequence and admitting a new one into its freed slot
// must not change any other still-active sequence's results relative to
// running it alone.
func TestEvictAdmitMidStreamDoesNotDisturbOthers(t *testing.T) {
	ck := fixtureCheckpoint()
	m := model.New(ck)
	s := NewScheduler(m, 2)

	kvA, kvB := newKV(ck), newKV(ck)
	if err := s.Admit(100, kvA); err != nil { // "A": finishes early
		t.Fatalf("Admit(A): %v", err)
	}
	if err := s.Admit(200, kvB); err != nil { // "B": runs the whole time
		t.Fatalf("Admit(B): %v", err)
	}

	tokensA := []int32{1, 2}
	tokensB := []int32{0, 1, 2, 0}

	if _, err := s.Tick(map[int]int32{100: tokensA[0], 200: tokensB[0]}); err != nil {
		t.Fatalf("Tick 0: %v", err)
	}
	if _, err := s.Tick(map[int]int32{100: tokensA[1], 200: tokensB[1]}); err != nil {
		t.Fatalf("Tick 1: %v", err)
	}

	// A is "done" after 2 tokens: evict it and admit C into its slot.
	s.Evict(100)
	kvC := newKV(ck)
	if err := s.Admit(300, kvC); err != nil {
		t.Fatalf("Admit(C) into freed slot: %v", err)
	}
	tokensC := []int32{2, 0}

	var lastB, lastC []float32
	var err error
	res, err := s.Tick(map[int]int32{200: tokensB[2], 300: tokensC[0]})
	if err != nil {
		t.Fatalf("Tick 2: %v", err)
	}
	lastB, lastC = res[200], res[300]

	res, err = s.Tick(map[int]int32{200: tokensB[3], 300: tokensC[1]})
	if err != nil {
		t.Fatalf("Tick 3: %v", err)
	}
	lastB, lastC = res[200], res[300]
	_ = lastC

	wantB := m.ForwardSequence(tokensB)
	for j := range wantB {
		if lastB[j] != wantB[j] {
			t.Fatalf("B contaminated by A's eviction/C's admission at logit %d: %v vs %v", j, lastB[j], wantB[j])
		}
	}

	wantC := m.ForwardSequence(tokensC)
	for j := range wantC {
		if lastC[j] != wantC[j] {
			t.Fatalf("C (admitted into a freed slot) diverges from running it alone at logit %d: %v vs %v", j, lastC[j], wantC[j])
		}
	}
}
