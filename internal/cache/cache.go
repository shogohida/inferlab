// Package cache owns the persistent key/value buffers that turn
// internal/model's per-position Step primitive into real incremental
// decoding: call Step with the same KVCache's buffers across increasing
// positions and each call only pays for the one new token, instead of
// recomputing the whole growing prefix from scratch every time.
package cache

import "fmt"

// KVCache holds per-layer key/value buffers for one sequence, each sized
// [seqLen * kvDim], plus how many positions have been reserved so far.
// Keys and Values are exported because internal/model.Step operates on them
// directly as plain [][]float32 — Step has no notion of a "cache" at all,
// it just reads and writes whichever buffers it's given.
type KVCache struct {
	Keys, Values [][]float32

	kvDim  int
	seqLen int
	Len    int // number of positions reserved so far (0..seqLen)
}

// New allocates a zeroed KVCache for nLayers layers, each able to hold up to
// seqLen positions of a kvDim-wide key/value vector.
func New(nLayers, kvDim, seqLen int) *KVCache {
	c := &KVCache{kvDim: kvDim, seqLen: seqLen}
	c.Keys = make([][]float32, nLayers)
	c.Values = make([][]float32, nLayers)
	for l := 0; l < nLayers; l++ {
		c.Keys[l] = make([]float32, seqLen*kvDim)
		c.Values[l] = make([]float32, seqLen*kvDim)
	}
	return c
}

// Reserve claims the next position for a new token, returning its index for
// use as model.Step's pos argument. It errors instead of silently wrapping
// or overwriting once the cache is full, so a caller that keeps generating
// past seqLen gets a clear failure rather than quietly corrupting an
// earlier position's data.
func (c *KVCache) Reserve() (pos int, err error) {
	if c.Len >= c.seqLen {
		return 0, fmt.Errorf("cache: sequence length exceeded: max_seq_len=%d", c.seqLen)
	}
	pos = c.Len
	c.Len++
	return pos, nil
}

// Evict resets the cache to empty (Len=0) so its buffers can be reused by a
// new sequence without reallocating — the operation internal/batch's
// continuous-batching scheduler performs when a finished sequence's slot is
// handed to a newly admitted one. The underlying buffers are not zeroed:
// the next sequence's first Reserve()+Step() call will overwrite position 0
// before anything ever reads it, for the same reason Step never reads a
// position beyond what it has itself just written.
func (c *KVCache) Evict() {
	c.Len = 0
}
