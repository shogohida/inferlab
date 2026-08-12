// Package api wires the loaded model into two public HTTP endpoints:
// /api/generate (SSE token streaming for the live text-generation demo) and
// /api/benchmark (runs generation under toggled configurations and reports
// measured tokens/sec, making the effect of caching, batching, and
// quantization watchable rather than a claimed number). Both fp32 and int8
// weight-only quantized models are built once at startup and reused across
// requests — quantizing on every request would defeat the point of a memory
// win.
package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"inferlab/internal/batch"
	"inferlab/internal/cache"
	"inferlab/internal/loader"
	"inferlab/internal/model"
	"inferlab/internal/quant"
	"inferlab/internal/tokenizer"
)

// Public-endpoint resource bounds — this project's CPU is a public,
// unauthenticated, free-tier server (see routelab's internal/api for the
// same discipline), so every knob a request body could turn up is clamped
// server-side regardless of what it claims.
const (
	minMaxTokens = 1
	maxMaxTokens = 60
	defMaxTokens = 30

	maxPromptChars = 200

	// benchmarkUncachedTokens bounds only the no-cache leg, independent of
	// the caller's maxTokens: recomputing the whole prefix from scratch
	// every step is O(tokens^2), so this stays small even when the other
	// legs run the full requested length.
	benchmarkUncachedTokens = 8
	benchmarkBatchSize      = 3
)

type Handler struct {
	checkpoint *loader.Checkpoint
	tokenizer  *tokenizer.Tokenizer
	fp32Model  *model.Model
	quantModel *model.Model

	fp32Bytes  int64
	quantBytes int64
}

// New builds a Handler with both precisions of the model preloaded.
func New(ck *loader.Checkpoint, tok *tokenizer.Tokenizer) *Handler {
	qw := quant.Quantize(ck)
	return &Handler{
		checkpoint: ck,
		tokenizer:  tok,
		fp32Model:  model.New(ck),
		quantModel: model.NewFromWeights(ck.Config, qw),
		fp32Bytes:  fp32WeightBytes(ck.Config),
		quantBytes: int64(qw.ByteSize()),
	}
}

// fp32WeightBytes computes the same set of tensors quant.Weights.ByteSize()
// covers (embedding/classifier + every per-layer matmul weight, excluding
// the small fp32 RMSNorm vectors) at 4 bytes/element, so the two numbers are
// a fair apples-to-apples static memory comparison.
func fp32WeightBytes(cfg loader.Config) int64 {
	dim, hiddenDim, kvDim := int64(cfg.Dim), int64(cfg.HiddenDim), int64(cfg.KVDim())
	nLayers, vocab := int64(cfg.NLayers), int64(cfg.VocabSize)
	params := vocab*dim + // embedding (classifier shared for the checkpoint this project targets)
		nLayers*dim*dim + // wq
		nLayers*dim*kvDim + // wk
		nLayers*dim*kvDim + // wv
		nLayers*dim*dim + // wo
		nLayers*hiddenDim*dim + // w1
		nLayers*dim*hiddenDim + // w2
		nLayers*hiddenDim*dim // w3
	return params * 4
}

func (h *Handler) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/generate", h.handleGenerate)
	mux.HandleFunc("POST /api/benchmark", h.handleBenchmark)
	return mux
}

func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func argmax(x []float32) int32 {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return int32(best)
}

// --- /api/generate ---------------------------------------------------

type generateRequest struct {
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"maxTokens"`
	Quantize  bool   `json:"quantize"`
}

func (h *Handler) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req generateRequest
	if r.Body != nil {
		defer r.Body.Close()
		_ = json.NewDecoder(r.Body).Decode(&req)
	}

	prompt := req.Prompt
	if len(prompt) > maxPromptChars {
		prompt = prompt[:maxPromptChars]
	}
	if prompt == "" {
		prompt = "Once upon a time"
	}
	maxTokens := defMaxTokens
	if req.MaxTokens != 0 {
		maxTokens = clampInt(req.MaxTokens, minMaxTokens, maxMaxTokens)
	}

	m := h.fp32Model
	if req.Quantize {
		m = h.quantModel
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	cfg := h.checkpoint.Config
	promptTokens := h.tokenizer.Encode(prompt, true, false)
	if len(promptTokens) >= cfg.SeqLen {
		promptTokens = promptTokens[:cfg.SeqLen-1]
	}

	keyCache, valCache := m.NewCacheBuffers()
	var logits []float32
	pos := 0
	for _, t := range promptTokens {
		logits = m.Step(t, pos, keyCache, valCache)
		pos++
	}

	prev := promptTokens[len(promptTokens)-1]
	generated := 0
	for pos < cfg.SeqLen && generated < maxTokens {
		next := argmax(logits)
		if next == tokenizer.EOSID {
			break
		}
		piece := h.tokenizer.Decode(prev, next)
		writeSSEEvent(w, "token", piece)
		flusher.Flush()

		prev = next
		generated++
		logits = m.Step(next, pos, keyCache, valCache)
		pos++
	}
	writeSSEEvent(w, "done", "")
	flusher.Flush()
}

func writeSSEEvent(w http.ResponseWriter, event, data string) {
	fmt.Fprintf(w, "event: %s\n", event)
	fmt.Fprintf(w, "data: %s\n\n", data)
}

// --- /api/benchmark ----------------------------------------------------

type benchmarkRequest struct {
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"maxTokens"`
}

type benchmarkResult struct {
	Label        string  `json:"label"`
	Tokens       int     `json:"tokens"`
	ElapsedMS    float64 `json:"elapsedMs"`
	TokensPerSec float64 `json:"tokensPerSec"`
	MemoryBytes  int64   `json:"memoryBytes"`
}

type benchmarkResponse struct {
	Results []benchmarkResult `json:"results"`
}

func (h *Handler) handleBenchmark(w http.ResponseWriter, r *http.Request) {
	var req benchmarkRequest
	if r.Body != nil {
		defer r.Body.Close()
		_ = json.NewDecoder(r.Body).Decode(&req)
	}

	prompt := req.Prompt
	if len(prompt) > maxPromptChars {
		prompt = prompt[:maxPromptChars]
	}
	if prompt == "" {
		prompt = "Once upon a time"
	}
	maxTokens := defMaxTokens
	if req.MaxTokens != 0 {
		maxTokens = clampInt(req.MaxTokens, minMaxTokens, maxMaxTokens)
	}

	promptTokens := h.tokenizer.Encode(prompt, true, false)
	cfg := h.checkpoint.Config
	if len(promptTokens) >= cfg.SeqLen {
		promptTokens = promptTokens[:cfg.SeqLen-1]
	}

	cachedFP32 := h.runCached(h.fp32Model, "cached_fp32", promptTokens, maxTokens)
	cachedFP32.MemoryBytes = h.fp32Bytes
	cachedInt8 := h.runCached(h.quantModel, "cached_int8", promptTokens, maxTokens)
	cachedInt8.MemoryBytes = h.quantBytes

	results := []benchmarkResult{
		h.runUncached(h.fp32Model, "no_cache_fp32", promptTokens, clampInt(maxTokens, 1, benchmarkUncachedTokens)),
		cachedFP32,
		cachedInt8,
		h.runSequential(h.fp32Model, "sequential_x3_fp32", promptTokens, maxTokens, benchmarkBatchSize),
		h.runBatched(h.fp32Model, "batched_x3_fp32", promptTokens, maxTokens, benchmarkBatchSize),
	}

	writeJSON(w, http.StatusOK, benchmarkResponse{Results: results})
}

// runUncached recomputes the entire growing prefix from scratch at every
// step (model.ForwardSequence, no cache reuse across steps) — the O(n^2)
// baseline the KV cache exists to avoid. tokens is capped independently of
// the other legs since this one is quadratic in it.
func (h *Handler) runUncached(m *model.Model, label string, promptTokens []int32, maxTokens int) benchmarkResult {
	cfg := h.checkpoint.Config
	start := time.Now()
	tokens := append([]int32(nil), promptTokens...)
	generated := 0
	for generated < maxTokens && len(tokens) < cfg.SeqLen {
		logits := m.ForwardSequence(tokens)
		next := argmax(logits)
		if next == tokenizer.EOSID {
			break
		}
		tokens = append(tokens, next)
		generated++
	}
	return toResult(label, generated, time.Since(start))
}

// runCached generates with a persistent KV cache — one Step call per new
// token, each paying only for that token.
func (h *Handler) runCached(m *model.Model, label string, promptTokens []int32, maxTokens int) benchmarkResult {
	cfg := h.checkpoint.Config
	start := time.Now()
	keyCache, valCache := m.NewCacheBuffers()
	var logits []float32
	pos := 0
	for _, t := range promptTokens {
		logits = m.Step(t, pos, keyCache, valCache)
		pos++
	}
	generated := 0
	for generated < maxTokens && pos < cfg.SeqLen {
		next := argmax(logits)
		if next == tokenizer.EOSID {
			break
		}
		generated++
		logits = m.Step(next, pos, keyCache, valCache)
		pos++
	}
	return toResult(label, generated, time.Since(start))
}

// runSequential is the batching baseline: n independent cached generations
// of the same prompt, run one after another. Reported tok/s is aggregate
// (total tokens across all n / total wall time), directly comparable to
// runBatched's aggregate figure.
func (h *Handler) runSequential(m *model.Model, label string, promptTokens []int32, maxTokens, n int) benchmarkResult {
	start := time.Now()
	totalGenerated := 0
	for i := 0; i < n; i++ {
		r := h.runCached(m, "", promptTokens, maxTokens)
		totalGenerated += r.Tokens
	}
	return toResult(label, totalGenerated, time.Since(start))
}

// runBatched generates n concurrent copies of the same prompt together,
// batching every decode step's linear layers into one call via
// batch.Scheduler. Each sequence is prefilled sequentially first (this
// project's batching is decode-only, see internal/batch's doc comment) and
// then admitted; tok/s is aggregate across all n sequences.
func (h *Handler) runBatched(m *model.Model, label string, promptTokens []int32, maxTokens, n int) benchmarkResult {
	cfg := h.checkpoint.Config
	sched := batch.NewScheduler(m, n)
	lastToken := make(map[int]int32, n)
	active := n

	for i := 0; i < n; i++ {
		kv := cache.New(cfg.NLayers, cfg.KVDim(), cfg.SeqLen)
		var logits []float32
		pos := 0
		for _, t := range promptTokens {
			logits = m.Step(t, pos, kv.Keys, kv.Values)
			pos++
		}
		if err := sched.Admit(i, kv); err != nil {
			active--
			continue
		}
		lastToken[i] = argmax(logits)
	}

	start := time.Now()
	totalGenerated := 0
	for step := 0; step < maxTokens && active > 0; step++ {
		results, err := sched.Tick(lastToken)
		if err != nil {
			break
		}
		for id, logits := range results {
			next := argmax(logits)
			if next == tokenizer.EOSID {
				sched.Evict(id)
				delete(lastToken, id)
				active--
				continue
			}
			lastToken[id] = next
			totalGenerated++
		}
	}
	return toResult(label, totalGenerated, time.Since(start))
}

func toResult(label string, tokens int, elapsed time.Duration) benchmarkResult {
	secs := elapsed.Seconds()
	tps := 0.0
	if secs > 0 {
		tps = float64(tokens) / secs
	}
	return benchmarkResult{
		Label:        label,
		Tokens:       tokens,
		ElapsedMS:    float64(elapsed.Microseconds()) / 1000,
		TokensPerSec: tps,
	}
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
