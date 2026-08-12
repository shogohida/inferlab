# System-design interview reference (inferlab deep-dive)

This document collects the deep-dive questions an interviewer is likely to
ask about `inferlab`'s implementation, with model answers grounded in the
actual code (`internal/model/`, `internal/cache/`, `internal/quant/`,
`internal/batch/`). Each answer is sized to say out loud in roughly
30-60 seconds.

Each entry follows: **Follow-up question** an interviewer might ask →
**Model answer** → **If they push further** → **Code reference**.

---

## 1. Why does a KV cache turn per-step attention from O(n) work into O(1) new work?

### Follow-up question
"Walk me through why you need a KV cache at all — why can't you just call
the model fresh on the growing sequence each time?"

### Model answer
Self-attention at position `t` needs the key and value vectors of every
position `0..t`. Those key/value vectors are a *deterministic function of
that position's token and the model's weights* — they never change once
computed. Without a cache, generating token `t+1` means recomputing K/V for
positions `0..t` all over again just to get the one new position's worth of
attention context: `internal/model.ForwardSequence` does exactly this,
rebuilding fresh cache buffers and replaying `Step` from position 0 every
call. Over a full generation of length `n`, that's `1+2+...+n = O(n²)` total
Step-equivalent work. A persistent `cache.KVCache` (`internal/cache/cache.go`)
just keeps those already-computed K/V vectors around: each new token calls
`Step` exactly once, writing only its own position's K/V and reading the
existing `0..t-1` entries — `O(n)` total work for the same `n`-token
generation.

### If they push further
"Doesn't that mean caching *changes* what attention computes?" — No, and
that's the property that actually matters: `internal/cache/cache_test.go`'s
`TestCachedDecodeMatchesUncachedRecompute` asserts the cached path and a
full from-scratch recompute produce *bit-identical* logits at every step.
Caching is purely a performance optimization; if it changed results, it
would be a bug, not a feature.

### Code reference
`internal/model/model.go`'s `Step` (writes/reads cache), `internal/cache/cache.go`'s `Reserve`, `internal/cache/cache_test.go`'s `TestCachedDecodeMatchesUncachedRecompute`.

---

## 2. Why can't attention batch across sequences the same way the feed-forward layer does?

### Follow-up question
"You batched the linear layers but not attention — why is that split
necessary rather than just convenient?"

### Model answer
The linear projections (QKVO, FFN gate/up/down) are *position-independent*:
`out[i] = sum_j W[i][j] * x[j]` for one sequence's current-token activation,
with no dependency on any other sequence. Stacking N sequences' activations
into one matrix and running one `Linear.BatchMatMul` call is exactly
equivalent to N separate calls — see `internal/model/weights.go`'s
`denseLinear.BatchMatMul`, which literally delegates to `tensor.BatchMatMul`,
itself just N independent per-row dot products. Attention is different: each
sequence attends over *its own* KV-cache history, and under continuous
batching, different sequences in the same batch tick have been generating
for different numbers of steps — their cache lengths genuinely differ. There
is no shape-uniform matrix you can build across sequences without padding
every sequence out to the longest one's length and masking the padding, and
this project deliberately doesn't implement that. So `StepBatch`
(`internal/model/batch_step.go`) batches the linear layers as one call per
layer, then falls back to a plain per-sequence, per-head loop for the
attention score/softmax/weighted-sum step.

### If they push further
"How does a real system like vLLM handle this?" — vLLM's PagedAttention is
the production answer to exactly this problem: it splits a batched
linear-algebra path from a separate, purpose-built ragged/paged attention
kernel that handles variable-length KV histories directly, without padding
waste. `StepBatch`'s split (batch what's position-independent, loop over
what's ragged) is a small, honestly-scoped version of the same idea, not a
coincidence.

### Code reference
`internal/model/batch_step.go`'s `StepBatch`; `internal/model/weights.go`'s `Linear`/`BatchMatMul`; `internal/model/model_test.go`'s `TestStepBatchMatchesSequentialStep`.

---

## 3. Why does batching improve throughput at all, if the arithmetic is identical either way?

### Follow-up question
"If `BatchMatMul` is literally N independent dot products with the same
total multiply-adds as N separate calls, where does the speedup come from?"

### Model answer
It comes from memory bandwidth, not arithmetic reduction. A weight matrix
like `WQ` for one layer has to be read from RAM to compute a dot product
against it. Run N sequences through `MatMul` separately and — depending on
whether the matrix survives in cache between calls — you may stream that
same weight matrix from memory N times. Run them through `BatchMatMul` and
the matrix is read once per layer and reused across all N sequences' dot
products while it's hot. This project's own benchmark panel measures this
directly: `internal/api/handlers.go`'s `runSequential` (N independent cached
generations, run back to back) versus `runBatched` (the same N sequences
processed through `batch.Scheduler` one tick at a time) report *aggregate*
tokens/sec, and batching measurably wins on real hardware — see the
`Running the tests`/live-demo numbers in `README.md`.

### If they push further
"Is this always a win?" — No: for very small batch sizes or very large
per-sequence hidden dimensions, the weight matrix may already be cache-
resident regardless of batching, and the win shrinks. This project's own
quantization result (Q6 below) is the sharper version of this same lesson:
a memory-bandwidth argument only pays off when the workload is actually
bandwidth-bound.

### Code reference
`internal/api/handlers.go`'s `runSequential`/`runBatched`; `internal/batch/scheduler.go`'s `Tick`.

---

## 4. What does "continuous" mean in continuous batching, and why is admit/evict the hard part?

### Follow-up question
"What's the actual difference between continuous batching and just batching
N requests together?"

### Model answer
Static batching waits for every sequence in a batch to finish before
accepting new work — a batch of N runs at the speed of its slowest/longest
member, and the GPU/CPU sits partially idle once shorter sequences finish
early. Continuous batching instead lets a finished sequence's slot be handed
to a newly queued sequence on the very next tick, without disturbing anyone
else mid-generation. The correctness bar for that is: evicting sequence A
and admitting sequence C into its slot must not change sequence B's results
at all. `batch.Scheduler.Evict` (`internal/batch/scheduler.go`) just deletes
an entry from a map — the surviving sequences' `*cache.KVCache` objects are
untouched, so there's no shared mutable state to corrupt.
`TestEvictAdmitMidStreamDoesNotDisturbOthers` proves this directly: it
evicts one sequence, admits a new one into its freed slot, and checks the
surviving sequence's output still matches running it completely alone.

### If they push further
"What would break this in a naive implementation?" — Using a slice with
fixed positional indices instead of a map keyed by sequence ID, and shifting
elements on eviction, would silently reassign which physical cache buffer
belongs to which logical sequence — a correctness bug that a slice-based
scheduler without exactly this test would likely ship undetected.

### Code reference
`internal/batch/scheduler.go`'s `Admit`/`Evict`/`Tick`; `internal/batch/scheduler_test.go`'s `TestEvictAdmitMidStreamDoesNotDisturbOthers`.

---

## 5. Why per-row quantization instead of one scale for the whole matrix?

### Follow-up question
"You quantize each row of a weight matrix with its own scale instead of one
scale for the whole thing — why does that matter?"

### Model answer
Symmetric int8 quantization maps a row's values to `[-127, 127]` using
`scale = max(|row|) / 127`. If one row happens to contain a much larger
weight than the rest of the matrix, a single matrix-wide scale would be
dominated by that outlier, crushing every other row's precision down toward
zero. Scoping the scale to each row means one row's outlier only affects
that row. `internal/tensor/quant_test.go`'s
`TestQuantizePerRowBeatsPerTensorOnOutlierRow` makes this concrete: a
two-row matrix where row 0 has huge outlier values and row 1 has small,
uniform ones — per-row quantization keeps row 1 accurate; a single shared
scale wouldn't.

### If they push further
"Why not go further and use blockwise quantization, like GGUF's
group-size-64 scheme?" — Blockwise quantization scopes the scale even
tighter (per N-element block within a row instead of per whole row),
recovering more accuracy at a given bit width at the cost of more
bookkeeping (a scale per block instead of per row). It's a real, deliberate
scope cut for this project — documented in `README.md`'s Known Limitations
— chosen because per-row quantization is a smaller, easier-to-verify
correctness surface while still being a genuine, defensible design decision
rather than the simplest possible option (per-tensor).

### Code reference
`internal/tensor/quant.go`'s `Quantize`/`QuantMatMul`; `internal/tensor/quant_test.go`'s `TestQuantizePerRowBeatsPerTensorOnOutlierRow`.

---

## 6. Why did quantization save memory but not improve speed in this project's own benchmark?

### Follow-up question
"Your README says int8 quantization was actually *slower* than fp32 on your
hardware. Isn't quantization supposed to make things faster?"

### Model answer
Quantization's speed win, when it happens, comes from moving less data
through memory — 1 byte per weight instead of 4. But `QuantMatMul`
(`internal/tensor/quant.go`) still has to convert each int8 value back to a
`float32` and multiply it by that row's scale, per element, in software.
Real hardware speedups for int8 come from dedicated SIMD dot-product
instructions (AVX-VNNI on x86, the dot-product extensions in ARM NEON) that
do the int8 multiply-accumulate natively, without a per-element float
conversion. This project's Go implementation has no such instruction path —
it's a portable scalar loop. So each quantized multiply-add costs *more*
compute than the fp32 version, while the memory-bandwidth savings only pay
for themselves if the workload is actually bandwidth-bound. A ~60MB model
largely resident in a modern CPU's cache during a benchmark run may not be
bandwidth-bound at all, so the compute overhead wins and quantization comes
out slower — measured, not assumed. The real win quantization delivers here
is static memory footprint (`~15MB int8` vs `~60MB fp32`, both measured by
`internal/quant.Weights.ByteSize()`), which matters directly for fitting
inside Render free tier's 512MB RAM budget — an honest, narrower claim than
"quantization is always faster."

### If they push further
"How would you actually get a speedup from int8 on this hardware?" — Either
call into a real SIMD int8 GEMM kernel (which would mean stepping outside
"pure Go, no ML library," a deliberate constraint of this project) or batch
enough sequences together that the *aggregate* bandwidth savings across a
larger batch outweigh the fixed per-element conversion cost — an
experiment `internal/batch`'s scheduler makes possible to actually run.

### Code reference
`internal/tensor/quant.go`'s `QuantMatMul`; `internal/quant/quant.go`'s `ByteSize`; the benchmark numbers in `README.md`.

---

## 7. Why is the causal-masking test built the way it is, instead of just checking output looks reasonable?

### Follow-up question
"How do you actually verify attention respects the causal boundary, instead
of just eyeballing generated text?"

### Model answer
A subtle off-by-one in the attention loop's upper bound (`t <= pos` versus,
say, `t < len(cache)`) would let a token "see" a future position's key/value
— and depending on the bug, the output might still look like plausible
text, just subtly wrong in a way that's very hard to notice by inspection.
`TestCausalMaskingIgnoresFutureCache`
(`internal/model/model_test.go`) makes this observable directly: it fills
every cache slot *beyond* the position under test with extreme values
(`1e6`) before those slots would ever legitimately be written, computes
logits at the target position, and asserts they're identical to a run where
those future slots were never touched. If attention ever read past its
causal boundary, the extreme poisoned values would visibly blow up the
softmax and change the output — this test would fail loudly instead of
silently producing "slightly worse" text.

### Code reference
`internal/model/model.go`'s `Step` (the `for t := 0; t <= pos; t++` attention loop); `internal/model/model_test.go`'s `TestCausalMaskingIgnoresFutureCache`.
