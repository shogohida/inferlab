# inferlab

A transformer inference engine built from scratch in pure Go — no PyTorch,
no ONNX Runtime, no cgo bindings to anything. The forward pass (RMSNorm,
rotary positional embeddings, grouped-query causal attention, SwiGLU), a
KV cache, int8 weight-only quantization, and a decode-only continuous
batching scheduler are all implemented here, over
[Andrej Karpathy's llama2.c](https://github.com/karpathy/llama2.c) file
format and its `stories15M` TinyStories checkpoint (a small Llama-2-
architecture model, small enough to run token-by-token on a free-tier CPU).

```
inferlab/
├── internal/tensor/      MatMul, RMSNorm, Softmax, SiLU; int8 QuantizedTensor
├── internal/loader/      llama2.c checkpoint format parser
├── internal/tokenizer/   llama2.c's tokenizer.bin format + BPE-merge encode/decode
├── internal/model/       the transformer forward pass (Step, StepBatch)
├── internal/cache/       per-sequence KV cache
├── internal/quant/       int8 weight-only quantization
├── internal/batch/       decode-only continuous batching scheduler
├── internal/api/         /api/generate (SSE) and /api/benchmark
├── web/                  embedded frontend: live generation + benchmark bars
└── cmd/server/           entrypoint: serves the API and frontend on one port
```

## Why this project

Production experience building and shipping backend systems doesn't
automatically demonstrate systems-level fundamentals — the kind of thing a
Google/Amazon/Meta-tier infra interview, or a serious ML-infra role,
actually probes: what a KV cache buys you and why, why attention doesn't
batch the same way a feed-forward layer does, why quantization is a
memory/compute trade-off and not a free win. This project exists to make
those fundamentals runnable and testable rather than something to describe
secondhand. Unlike this portfolio's other projects, it's deliberately *not*
tied to a specific résumé claim — see `INTERVIEW_QA.md` for the deep-dive
version of what each piece is actually demonstrating.

## Live demo

Deployed for free on [Render](https://render.com) via `render.yaml` — the
build step downloads `stories15M.bin` and `tokenizer.bin` directly from
their canonical public sources (see "Model weights" below), verifies their
checksums, and builds. No Docker, no credit card. Free-tier services sleep
after 15 minutes of inactivity; the first request after a sleep takes a few
seconds to wake up.

```bash
go run ./cmd/server   # reads weights/stories15M.bin + weights/tokenizer.bin
# open http://localhost:8080
```

For local development, `dev-weights/` (gitignored) holds a local copy;
point the server at it with `CHECKPOINT_PATH`/`TOKENIZER_PATH`:

```bash
CHECKPOINT_PATH=dev-weights/stories15M.bin TOKENIZER_PATH=dev-weights/tokenizer.bin go run ./cmd/server
```

## Model weights

Weights are **never committed to this repository**. `render.yaml`'s
`buildCommand` fetches them at build time from
[`karpathy/tinyllamas`](https://huggingface.co/karpathy/tinyllamas) on
Hugging Face (`stories15M.bin`) and the
[llama2.c](https://github.com/karpathy/llama2.c) repo (`tokenizer.bin`),
verifying each against a pinned SHA-256 checksum before building. This
sidesteps any checkpoint-redistribution licensing nuance entirely — this
project points to Karpathy's own public release rather than redistributing
it — and keeps the git repo free of binary blobs. Model weights were
trained by Andrej Karpathy on the TinyStories dataset
([Eldan & Li, Microsoft Research](https://arxiv.org/abs/2305.07759)); this
project reimplements only the inference engine from scratch.

`go test ./...` never needs network access: every package's tests run
against small, hand-built synthetic fixtures, not the real checkpoint.

## Design decisions & trade-offs

**Decode-only batching, no padding or attention masking.** The position-
independent linear layers (QKVO projections, FFN gate/up/down) are batched
as one real `Linear.BatchMatMul` call across every active sequence's
current-token activation — the shared weight matrix is streamed from memory
once per layer and amortized over N dot products instead of once per
sequence, which is the actual memory-bandwidth win batching exists to
deliver. Attention stays a plain per-sequence, per-head loop: each sequence
has its own KV-cache length (sequences enter a batch at different points in
their generation under continuous batching), and there's no padding/masking
scheme here to make that shape-uniform. This is a faithful, if smaller-
scale, version of how systems like vLLM split a batched linear-algebra path
from a separate ragged/paged attention kernel — not a simplification that
hides a correctness gap. Prefill (processing the initial prompt) runs
sequentially per request rather than batched, since TinyStories prompts are
short and decode is where continuous-batching systems get most of their
throughput win anyway.

**`BatchMatMul` starts as N independent per-row dot products**, not a
tiled/blocked GEMM — bit-identical to calling the single-sequence `MatMul`
once per row (see `internal/model`'s batched-vs-sequential equivalence
test), which made it possible to assert *exact* equality rather than a
tolerance in that test. A cache-blocked version is a real roadmap item (see
Known Limitations) but would reassociate floating-point summation order,
trading that exact-equality guarantee for a bounded-tolerance one.

**Per-row (per-output-channel) symmetric int8 weight-only quantization**,
not blockwise/groupwise (GGUF-style). Simpler to reason about and implement
than a blocked scheme, and meaningfully more accurate than a single scale
for the whole matrix — see `internal/tensor`'s outlier-row test, which
demonstrates *why*. Activations stay fp32 throughout (weight-only
quantization): `QuantMatMul` dequantizes each int8 row on the fly rather
than requiring a separate int8 activation path.

**The token embedding table is quantized too, not just attention/FFN
weights.** At TinyStories scale the embedding table is a much larger share
of total parameters than in a full-size LLM — for `stories15M`'s 32,000-word
vocabulary and `dim=288`, the embedding table alone is roughly 9.2M of the
model's ~15M parameters. Quantizing only the matmul weights would miss most
of the actual memory win.

**Quantization measurably shrinks memory but was *not* faster in this
project's own local benchmark — and that's an honest, expected result, not
a bug.** `QuantMatMul` does a scalar `int8 → float32` conversion and
multiply per element, with no SIMD int8 dot-product instructions (the kind
real hardware support via AVX-VNNI or ARM NEON's dot-product extensions).
Without that hardware path, each quantized multiply-add costs *more* than
the plain fp32 version, and the memory-bandwidth savings only pay for
themselves when the workload is actually bandwidth-bound rather than
compute-bound — which a ~60MB model largely resident in a modern CPU's
cache may not be. The benchmark panel reports whatever the real hardware
measures rather than a hand-waved "quantization is always faster" claim;
see `INTERVIEW_QA.md` for the deeper version of this trade-off.

**Model weights are fetched at build time, never committed** — see "Model
weights" above.

**Every unit test runs against small, hand-built fixtures — never the real
60MB checkpoint.** Loader/tokenizer tests hand-serialize tiny synthetic
files to test the byte format directly; model/cache/quant/batch tests build
a small in-memory `loader.Checkpoint` (a handful of layers, `dim` in the
single digits) so the whole suite runs in well under a second with zero
network dependency. `internal/model`'s golden-value test additionally cross-
checks its output against an independent NumPy reimplementation of the same
math, run once during development — the single highest-value correctness
gate in the project, since a subtle transformer-math bug would otherwise
only surface as "the real model's output looks like garbage," with no way
to localize which stage broke it.

## Running the tests

```bash
go test ./... -v
```

- `internal/tensor` — MatMul/RMSNorm/Softmax/SiLU against hand-computed and
  known reference values; int8 quantize/dequantize round-trip error bounded
  by half a quantization step; a per-row-beats-per-tensor outlier test.
- `internal/loader` — byte-format parsing against a hand-serialized
  synthetic checkpoint; both signs of the shared-weights convention;
  truncated/corrupt input errors instead of panicking.
- `internal/tokenizer` — byte-format parsing; encode→decode round-trip
  through the full greedy BPE-merge algorithm; the raw-byte fallback path;
  the BOS-leading-space-stripping quirk.
- `internal/model` — a golden-value test cross-checked against an
  independent NumPy reference; a causal-masking property test that poisons
  future cache slots with extreme values and confirms they're never read;
  RoPE identity at position 0; batched-vs-sequential decode equivalence
  (exact, not tolerance-based).
- `internal/cache` — cached decode matches a full uncached recompute at
  every step, exactly; two independent sequences never cross-contaminate;
  reserving past `max_seq_len` errors clearly.
- `internal/quant` — a full quantized checkpoint's forward pass stays close
  to its fp32 reference; embedding-row dequantization accuracy; quantized
  weight size is meaningfully smaller than fp32.
- `internal/batch` — the Scheduler's own `Admit`/`Tick`/`Evict` API produces
  results identical to sequential `Step` calls; mid-stream eviction and
  admission never disturbs other in-flight sequences; capacity is enforced.
- `internal/api` — server-side clamping on prompt length, `maxTokens`, and
  the no-cache benchmark leg's token budget; SSE streaming produces a
  `done` event; the benchmark endpoint returns all five configurations with
  a positive, well-formed result.

## Known limitations / roadmap

- **No prefill batching, no padding/masking.** Only the decode phase
  batches; see "Design decisions" above for why this is a deliberate scope
  cut rather than an oversight.
- **`BatchMatMul` is not a blocked/tiled GEMM.** It's N independent
  per-row dot products under a batched-shaped API. A real cache-blocked
  kernel would improve throughput further but reassociates floating-point
  summation order, so the batched-vs-sequential test would need to become
  tolerance-based rather than exact.
- **Quantization is per-row, not blockwise/groupwise.** GGUF-style grouped
  quantization (e.g. group size 64) would recover more accuracy at a given
  bit width at the cost of more bookkeeping per row.
- **No SIMD int8 dot-product path.** This is the direct cause of
  quantization not being faster in this project's own CPU benchmark — see
  "Design decisions" above.
- **`stories110M` is out of scope.** At roughly 440MB fp32 alone, it doesn't
  comfortably fit Render free tier's 512MB total-system RAM budget once the
  Go runtime and request buffers are added. `stories42M` (~165MB fp32,
  ~165MB more headroom to work with) is a documented "try a bigger model"
  build-time parameter, since `internal/loader`/`internal/model` are
  generic over the checkpoint's own Config header.

## What each package proves

| Package | Demonstrates |
|---|---|
| `internal/tensor` | Correct, tested numeric kernels (matmul, normalization, activations) and per-row symmetric int8 quantization with a real accuracy argument for the "per-row" choice |
| `internal/loader` | Parsing a real, non-self-describing binary model format correctly, including its shared-weights sign convention |
| `internal/tokenizer` | Reimplementing a production BPE-merge tokenizer's exact algorithm — including its byte-fallback and dummy-prefix quirks — from a format spec, not a library |
| `internal/model` | A from-scratch transformer forward pass (RoPE, GQA attention, SwiGLU) verified against an independent reference, plus the causal-masking and batching-equivalence properties that actually matter for correctness |
| `internal/cache` | Understanding *why* a KV cache is correct (identical results, not just faster) — the property most people can state but few can prove |
| `internal/quant` | Weight-only quantization applied to a full checkpoint, including the non-obvious fact that a small model's embedding table dominates its memory footprint |
| `internal/batch` | Continuous batching's real shape — batch what's batchable (linear layers), don't fake what isn't (ragged attention) — plus admit/evict scheduling that doesn't disturb unrelated in-flight sequences |
| `internal/api` | Composing a full inference engine into a resource-bounded public HTTP surface, with the same request-clamping discipline this portfolio's other public demos use |
