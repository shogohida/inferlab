// Package tokenizer implements llama2.c's tokenizer.bin format and its
// encode/decode algorithm from scratch: a greedy, highest-score pairwise
// byte-pair merge over a fixed vocabulary, with a raw-byte fallback path for
// bytes that never appear in the vocabulary directly. This is not
// SentencePiece or any general-purpose tokenizer library — it's the specific
// minimal format llama2.c exports, tied to the Llama vocabulary convention
// where token id 0 is <unk>, id 1 is BOS, id 2 is EOS, and ids 3-258 are
// reserved for raw-byte fallback tokens (stored in the vocab as literal
// "<0xXX>" hex-escape strings, one per byte value).
package tokenizer

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

const (
	// BOSID and EOSID are the fixed beginning/end-of-sequence token ids in
	// the Llama vocabulary convention this tokenizer implements — exported
	// so callers generating text know which token id means "stop."
	BOSID = 1
	EOSID = 2
)

// Tokenizer holds a fixed vocabulary (token id -> piece string) and the
// per-token merge scores used to greedily rank candidate merges during
// encoding.
type Tokenizer struct {
	Vocab          []string
	Scores         []float32
	MaxTokenLength int

	byPiece map[string]int32
}

// New builds a Tokenizer directly from an in-memory vocabulary, bypassing
// the tokenizer.bin byte format — used by Load and by test/demo fixtures
// that construct a small vocabulary programmatically.
func New(vocab []string, scores []float32, maxTokenLength int) *Tokenizer {
	byPiece := make(map[string]int32, len(vocab))
	for i, s := range vocab {
		byPiece[s] = int32(i)
	}
	return &Tokenizer{Vocab: vocab, Scores: scores, MaxTokenLength: maxTokenLength, byPiece: byPiece}
}

// Load parses llama2.c's tokenizer.bin format:
//
//	int32   max_token_length
//	repeated vocabSize times:
//	  float32  score
//	  int32    len
//	  byte[len] token bytes
//
// vocabSize is not stored in the file itself — it comes from the paired
// model checkpoint's Config.VocabSize.
func Load(r io.Reader, vocabSize int) (*Tokenizer, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read: %w", err)
	}
	if len(data) < 4 {
		return nil, fmt.Errorf("tokenizer: file too short for max_token_length header: got %d bytes", len(data))
	}
	maxTokenLength := int(binary.LittleEndian.Uint32(data))
	off := 4

	vocab := make([]string, vocabSize)
	scores := make([]float32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		if off+8 > len(data) {
			return nil, fmt.Errorf("tokenizer: truncated before entry %d/%d (offset %d, %d bytes remain)", i, vocabSize, off, len(data)-off)
		}
		scores[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		off += 4
		length := int(int32(binary.LittleEndian.Uint32(data[off:])))
		off += 4
		if length < 0 || off+length > len(data) {
			return nil, fmt.Errorf("tokenizer: truncated token %d/%d body (want %d bytes at offset %d, have %d)", i, vocabSize, length, off, len(data)-off)
		}
		vocab[i] = string(data[off : off+length])
		off += length
	}

	return New(vocab, scores, maxTokenLength), nil
}

func (t *Tokenizer) lookup(s string) (int32, bool) {
	id, ok := t.byPiece[s]
	return id, ok
}

// Encode converts text into a token sequence via llama2.c's algorithm:
// seed one token per raw byte/UTF-8-codepoint via direct vocabulary lookup
// (falling back to id = byte+3 for any byte that isn't itself a vocab
// entry), then greedily merge the highest-scoring adjacent pair whose
// concatenation is a vocabulary entry, repeating until no merge applies.
// A leading-space "dummy prefix" token is inserted before non-empty text,
// matching SentencePiece's implicit-leading-space convention.
func (t *Tokenizer) Encode(text string, bos, eos bool) []int32 {
	tokens := make([]int32, 0, len(text)+3)
	if bos {
		tokens = append(tokens, BOSID)
	}
	if text != "" {
		if id, ok := t.lookup(" "); ok {
			tokens = append(tokens, id)
		}
	}

	var buf []byte
	for i := 0; i < len(text); i++ {
		c := text[i]
		if c&0xC0 != 0x80 { // not a UTF-8 continuation byte: start a new codepoint
			buf = buf[:0]
		}
		buf = append(buf, c)
		if i+1 < len(text) && text[i+1]&0xC0 == 0x80 && len(buf) < 4 {
			continue // keep buffering continuation bytes of this codepoint
		}
		if id, ok := t.lookup(string(buf)); ok {
			tokens = append(tokens, id)
		} else {
			for _, b := range buf {
				tokens = append(tokens, int32(b)+3)
			}
		}
		buf = buf[:0]
	}

	for {
		bestScore := float32(math.Inf(-1))
		bestID := int32(-1)
		bestIdx := -1
		for i := 0; i < len(tokens)-1; i++ {
			merged := t.Vocab[tokens[i]] + t.Vocab[tokens[i+1]]
			if id, ok := t.lookup(merged); ok && t.Scores[id] > bestScore {
				bestScore = t.Scores[id]
				bestID = id
				bestIdx = i
			}
		}
		if bestIdx == -1 {
			break
		}
		tokens[bestIdx] = bestID
		tokens = append(tokens[:bestIdx+1], tokens[bestIdx+2:]...)
	}

	if eos {
		tokens = append(tokens, EOSID)
	}
	return tokens
}

// Decode returns the display text for token, given the token that preceded
// it (needed to strip SentencePiece's leading space immediately after BOS,
// and to correctly resolve raw-byte fallback tokens rendered as "<0xXX>").
func (t *Tokenizer) Decode(prevToken, token int32) string {
	piece := t.Vocab[token]
	if prevToken == BOSID && strings.HasPrefix(piece, " ") {
		piece = piece[1:]
	}
	if b, ok := parseBytePiece(piece); ok {
		return string([]byte{b})
	}
	return piece
}

// DecodeSequence decodes a full token sequence (as produced by Encode) back
// into text, skipping BOS and stopping at the first EOS.
func (t *Tokenizer) DecodeSequence(tokens []int32) string {
	var sb strings.Builder
	prev := int32(-1)
	for _, tok := range tokens {
		if tok == BOSID {
			prev = tok
			continue
		}
		if tok == EOSID {
			break
		}
		sb.WriteString(t.Decode(prev, tok))
		prev = tok
	}
	return sb.String()
}

func parseBytePiece(piece string) (byte, bool) {
	if len(piece) != 6 || !strings.HasPrefix(piece, "<0x") || piece[5] != '>' {
		return 0, false
	}
	v, err := strconv.ParseUint(piece[3:5], 16, 8)
	if err != nil {
		return 0, false
	}
	return byte(v), true
}
