package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"inferlab/internal/loader"
	"inferlab/internal/tokenizer"
)

// fixtureCheckpoint and fixtureTokenizer build a small but *format-valid*
// checkpoint/tokenizer pair: vocab_size must be at least 259 (id 0 = <unk>,
// 1 = BOS, 2 = EOS, 3-258 = raw-byte fallback tokens) for the real
// tokenizer.Encode/Decode algorithm to function on arbitrary text, even
// though the "vocabulary" here has no real words and generation output is
// gibberish — these tests exercise HTTP/streaming/clamping behavior, not
// text quality (that's covered by cmd/generate against the real checkpoint).
func fixtureCheckpoint() *loader.Checkpoint {
	dim, hiddenDim, nLayers, nHeads, nKVHeads, vocabSize, seqLen := 8, 8, 1, 2, 1, 259, 16
	kvDim := dim * nKVHeads / nHeads
	gen := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32((i%7)-3) * 0.05
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

func fixtureTokenizer() *tokenizer.Tokenizer {
	vocab := make([]string, 259)
	scores := make([]float32, 259)
	vocab[0], vocab[1], vocab[2] = "<unk>", "\n<s>\n", "\n</s>\n"
	for b := 0; b < 256; b++ {
		const hex = "0123456789ABCDEF"
		vocab[3+b] = string([]byte{'<', '0', 'x', hex[b>>4], hex[b&0xF], '>'})
		scores[3+b] = -1e9
	}
	return tokenizer.New(vocab, scores, 6)
}

func newTestHandler() *Handler {
	return New(fixtureCheckpoint(), fixtureTokenizer())
}

func countSSEEvents(body, event string) int {
	return strings.Count(body, "event: "+event)
}

func TestHandleGenerateStreamsSSEWithDoneEvent(t *testing.T) {
	h := newTestHandler()
	body := bytes.NewBufferString(`{"prompt": "hello there", "maxTokens": 5}`)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", body)
	w := httptest.NewRecorder()
	h.Routes().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if ct := w.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}
	got := w.Body.String()
	if countSSEEvents(got, "done") != 1 {
		t.Fatalf("expected exactly one 'done' event, got body: %q", got)
	}
	if countSSEEvents(got, "token") > 5 {
		t.Fatalf("expected at most 5 token events (maxTokens=5), got %d in body: %q", countSSEEvents(got, "token"), got)
	}
}

func TestHandleGenerateDefaultsOnEmptyBody(t *testing.T) {
	h := newTestHandler()
	req := httptest.NewRequest(http.MethodPost, "/api/generate", nil)
	w := httptest.NewRecorder()
	h.Routes().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 with empty body, got %d: %s", w.Code, w.Body.String())
	}
	if countSSEEvents(w.Body.String(), "done") != 1 {
		t.Fatalf("expected a 'done' event even with defaults, got: %q", w.Body.String())
	}
}

func TestHandleGenerateMalformedBodyStillWorks(t *testing.T) {
	h := newTestHandler()
	body := bytes.NewBufferString(`not json`)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", body)
	w := httptest.NewRecorder()
	h.Routes().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 with malformed body, got %d", w.Code)
	}
}

func TestHandleGenerateClampsOversizedMaxTokens(t *testing.T) {
	h := newTestHandler()
	body := bytes.NewBufferString(`{"prompt": "x", "maxTokens": 999999}`)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", body)
	w := httptest.NewRecorder()
	h.Routes().ServeHTTP(w, req)

	if got := countSSEEvents(w.Body.String(), "token"); got > maxMaxTokens {
		t.Fatalf("expected at most %d token events after clamping, got %d", maxMaxTokens, got)
	}
}

func TestHandleGenerateOversizedPromptIsTruncated(t *testing.T) {
	h := newTestHandler()
	huge := strings.Repeat("a", maxPromptChars*3)
	reqBody, _ := json.Marshal(generateRequest{Prompt: huge, MaxTokens: 1})
	req := httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewReader(reqBody))
	w := httptest.NewRecorder()
	h.Routes().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 with oversized prompt, got %d: %s", w.Code, w.Body.String())
	}
}

func TestHandleGenerateQuantizedFlagRuns(t *testing.T) {
	h := newTestHandler()
	body := bytes.NewBufferString(`{"prompt": "hi", "maxTokens": 3, "quantize": true}`)
	req := httptest.NewRequest(http.MethodPost, "/api/generate", body)
	w := httptest.NewRecorder()
	h.Routes().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 with quantize=true, got %d: %s", w.Code, w.Body.String())
	}
	if countSSEEvents(w.Body.String(), "done") != 1 {
		t.Fatalf("expected a 'done' event, got: %q", w.Body.String())
	}
}

func TestHandleBenchmarkReturnsAllLegsWithPositiveThroughput(t *testing.T) {
	h := newTestHandler()
	body := bytes.NewBufferString(`{"prompt": "hi", "maxTokens": 6}`)
	req := httptest.NewRequest(http.MethodPost, "/api/benchmark", body)
	w := httptest.NewRecorder()
	h.Routes().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp benchmarkResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON response: %v (%s)", err, w.Body.String())
	}

	wantLabels := []string{"no_cache_fp32", "cached_fp32", "cached_int8", "sequential_x3_fp32", "batched_x3_fp32"}
	if len(resp.Results) != len(wantLabels) {
		t.Fatalf("got %d results, want %d", len(resp.Results), len(wantLabels))
	}
	byLabel := map[string]benchmarkResult{}
	for i, r := range resp.Results {
		if r.Label != wantLabels[i] {
			t.Fatalf("result %d label = %q, want %q", i, r.Label, wantLabels[i])
		}
		if r.Tokens <= 0 {
			t.Fatalf("result %q: Tokens = %d, want > 0", r.Label, r.Tokens)
		}
		byLabel[r.Label] = r
	}

	fp32 := byLabel["cached_fp32"]
	int8 := byLabel["cached_int8"]
	if fp32.MemoryBytes <= 0 || int8.MemoryBytes <= 0 {
		t.Fatalf("expected positive MemoryBytes for both cached legs, got fp32=%d int8=%d", fp32.MemoryBytes, int8.MemoryBytes)
	}
	if int8.MemoryBytes >= fp32.MemoryBytes {
		t.Fatalf("expected cached_int8 MemoryBytes (%d) < cached_fp32 (%d)", int8.MemoryBytes, fp32.MemoryBytes)
	}
}

func TestHandleBenchmarkMalformedBodyStillDefaults(t *testing.T) {
	h := newTestHandler()
	body := bytes.NewBufferString(`not json`)
	req := httptest.NewRequest(http.MethodPost, "/api/benchmark", body)
	w := httptest.NewRecorder()
	h.Routes().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 with malformed body, got %d", w.Code)
	}
}

func TestHandleBenchmarkUncachedLegBoundedRegardlessOfMaxTokens(t *testing.T) {
	h := newTestHandler()
	body := bytes.NewBufferString(fmt.Sprintf(`{"prompt": "hi", "maxTokens": %d}`, maxMaxTokens))
	req := httptest.NewRequest(http.MethodPost, "/api/benchmark", body)
	w := httptest.NewRecorder()
	h.Routes().ServeHTTP(w, req)

	var resp benchmarkResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	for _, r := range resp.Results {
		if r.Label == "no_cache_fp32" && r.Tokens > benchmarkUncachedTokens {
			t.Fatalf("no_cache_fp32 generated %d tokens, want <= %d (independent of maxTokens=%d)",
				r.Tokens, benchmarkUncachedTokens, maxMaxTokens)
		}
	}
}

func TestClampInt(t *testing.T) {
	cases := []struct{ v, min, max, want int }{
		{5, 1, 10, 5},
		{-5, 1, 10, 1},
		{50, 1, 10, 10},
	}
	for _, c := range cases {
		if got := clampInt(c.v, c.min, c.max); got != c.want {
			t.Fatalf("clampInt(%d, %d, %d) = %d, want %d", c.v, c.min, c.max, got, c.want)
		}
	}
}

func TestArgmax(t *testing.T) {
	got := argmax([]float32{0.1, 0.9, -0.2, 0.5})
	if got != 1 {
		t.Fatalf("argmax = %d, want 1", got)
	}
}
