// Package batch implements continuous, decode-only request batching: a
// Scheduler that groups multiple concurrently in-flight sequences' single-
// token decode steps into one model.StepBatch call per tick, and lets
// finished sequences be evicted mid-stream with a newly admitted sequence
// taking their slot on the very next tick — the "continuous" in continuous
// batching, as opposed to a static batch that waits for every member to
// finish before accepting new work.
//
// This package owns scheduling policy only (which sequences share a tick,
// admission, capacity, eviction). The batched forward-pass math itself
// lives in internal/model's StepBatch, which needs the same unexported
// weight-access plumbing Step does.
package batch

import (
	"fmt"
	"sort"

	"inferlab/internal/cache"
	"inferlab/internal/model"
)

// Scheduler batches the decode-phase forward pass of up to capacity
// concurrently active sequences. It has no notion of prefill (processing a
// prompt) or of how a next token is chosen from logits — a caller prefills
// a sequence's KVCache with ordinary sequential Step calls first, then
// Admits it here once it's ready to generate token-by-token; token
// selection (greedy, sampling, ...) stays the caller's responsibility.
type Scheduler struct {
	model    *model.Model
	capacity int
	active   map[int]*cache.KVCache
}

// NewScheduler creates a Scheduler that batches up to capacity concurrently
// active sequences per tick.
func NewScheduler(m *model.Model, capacity int) *Scheduler {
	return &Scheduler{model: m, capacity: capacity, active: make(map[int]*cache.KVCache)}
}

// Admit registers a new sequence under id, whose decode steps will be
// included starting from the next Tick call. Returns an error if id is
// already admitted or the scheduler is already at capacity — the same
// server-side resource-bounding discipline applied elsewhere in this
// project's public endpoints (see routelab's request clamping), since an
// unbounded batch size directly drives free-tier CPU/memory cost.
func (s *Scheduler) Admit(id int, kv *cache.KVCache) error {
	if _, exists := s.active[id]; exists {
		return fmt.Errorf("batch: id %d is already admitted", id)
	}
	if len(s.active) >= s.capacity {
		return fmt.Errorf("batch: scheduler at capacity (%d)", s.capacity)
	}
	s.active[id] = kv
	return nil
}

// Evict removes id from the scheduler, freeing its slot for a future Admit
// call. Per-sequence state lives entirely in the caller-owned *cache.KVCache
// passed to Admit, so evicting one id never touches any other active
// sequence's cache — this is what makes it safe to evict a finished
// sequence and admit a new one in between ticks without disturbing whatever
// else is mid-generation.
func (s *Scheduler) Evict(id int) {
	delete(s.active, id)
}

// Active reports how many sequences are currently admitted.
func (s *Scheduler) Active() int { return len(s.active) }

// Tick advances every currently active sequence by exactly one token,
// combining their single-token forward passes into one batched
// model.StepBatch call. tokens must supply an input token for every active
// id (the token generated on that id's previous Tick, or its last prompt
// token on the first tick after prefill); Tick errors if any active id is
// missing from tokens, or if any sequence's cache has reached its
// max_seq_len. Reserving each sequence's next cache position is a side
// effect of this call — callers must not also call kv.Reserve() directly
// for ids passed here.
func (s *Scheduler) Tick(tokens map[int]int32) (map[int][]float32, error) {
	if len(s.active) == 0 {
		return nil, nil
	}

	ids := make([]int, 0, len(s.active))
	for id := range s.active {
		ids = append(ids, id)
	}
	sort.Ints(ids) // deterministic request ordering, for reproducible tests/debugging

	reqs := make([]model.BatchRequest, 0, len(ids))
	for _, id := range ids {
		tok, ok := tokens[id]
		if !ok {
			return nil, fmt.Errorf("batch: no input token supplied for active id %d", id)
		}
		kv := s.active[id]
		pos, err := kv.Reserve()
		if err != nil {
			return nil, fmt.Errorf("batch: id %d: %w", id, err)
		}
		reqs = append(reqs, model.BatchRequest{Token: tok, Pos: pos, KeyCache: kv.Keys, ValCache: kv.Values})
	}

	logits := s.model.StepBatch(reqs)

	out := make(map[int][]float32, len(ids))
	for i, id := range ids {
		out[id] = logits[i]
	}
	return out, nil
}
