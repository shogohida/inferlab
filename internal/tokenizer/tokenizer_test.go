package tokenizer

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// buildFixture constructs a small Tokenizer that follows the real Llama
// vocabulary convention closely enough to exercise Encode/Decode correctly:
// id 0 = <unk>, id 1 = BOS, id 2 = EOS, ids 3-258 = raw-byte fallback tokens
// (one "<0xXX>" hex-escape entry per byte value, low score so they never
// win a merge over a "real" vocabulary entry), plus a handful of ordinary
// tokens layered on top to exercise the greedy merge algorithm: a space,
// the individual letters of "hello", and every intermediate merge up to the
// full word, each with a strictly increasing score so the merge order is
// deterministic and easy to hand-verify.
func buildFixture() *Tokenizer {
	vocab := make([]string, 259, 259+16)
	scores := make([]float32, 259, 259+16)
	vocab[0], scores[0] = "<unk>", 0
	vocab[1], scores[1] = "\n<s>\n", 0
	vocab[2], scores[2] = "\n</s>\n", 0
	for b := 0; b < 256; b++ {
		vocab[3+b] = byteHex(byte(b))
		scores[3+b] = -1e9
	}

	add := func(piece string, score float32) {
		vocab = append(vocab, piece)
		scores = append(scores, score)
	}
	add(" ", -1)    // id 259: dummy-prefix space
	add("h", -1)    // 260
	add("e", -1)    // 261
	add("l", -1)    // 262
	add("o", -1)    // 263
	add("he", 1)    // 264
	add("hel", 2)   // 265
	add("hell", 3)  // 266
	add("hello", 4) // 267

	return New(vocab, scores, 5)
}

func byteHex(b byte) string {
	const hex = "0123456789ABCDEF"
	return string([]byte{'<', '0', 'x', hex[b>>4], hex[b&0xF], '>'})
}

func TestEncodeDecodeRoundTripMergesToWholeWord(t *testing.T) {
	tok := buildFixture()
	ids := tok.Encode("hello", true, true)

	wantIDs := []int32{BOSID, 259, 267, EOSID} // bos, space, "hello" (fully merged), eos
	if len(ids) != len(wantIDs) {
		t.Fatalf("Encode(%q) = %v, want %v", "hello", ids, wantIDs)
	}
	for i := range wantIDs {
		if ids[i] != wantIDs[i] {
			t.Fatalf("Encode(%q) = %v, want %v", "hello", ids, wantIDs)
		}
	}

	got := tok.DecodeSequence(ids)
	if got != "hello" {
		t.Fatalf("DecodeSequence(%v) = %q, want %q", ids, got, "hello")
	}
}

func TestEncodeDecodeByteFallback(t *testing.T) {
	tok := buildFixture()
	// '!' (0x21) has no direct vocab entry, only "h" does — exercises the
	// byte+3 fallback path adjacent to a normal token.
	ids := tok.Encode("h!", true, true)

	wantFallbackID := int32('!') + 3
	found := false
	for _, id := range ids {
		if id == wantFallbackID {
			found = true
		}
	}
	if !found {
		t.Fatalf("Encode(%q) = %v, expected byte-fallback id %d for '!'", "h!", ids, wantFallbackID)
	}

	got := tok.DecodeSequence(ids)
	if got != "h!" {
		t.Fatalf("DecodeSequence(%v) = %q, want %q", ids, got, "h!")
	}
}

func TestDecodeStripsLeadingSpaceAfterBOS(t *testing.T) {
	tok := buildFixture()
	// Directly after BOS, a piece beginning with a space has that space
	// stripped (SentencePiece's implicit-leading-space convention).
	got := tok.Decode(BOSID, 259) // 259 = " "
	if got != "" {
		t.Fatalf("Decode(bos, space-token) = %q, want empty string", got)
	}
	// The same token elsewhere keeps its space.
	got2 := tok.Decode(260, 259) // prev = "h", not bos
	if got2 != " " {
		t.Fatalf("Decode(non-bos, space-token) = %q, want %q", got2, " ")
	}
}

func appendU32(buf *bytes.Buffer, v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	buf.Write(b[:])
}

func TestLoadParsesTokenizerBinFormat(t *testing.T) {
	buf := new(bytes.Buffer)
	appendU32(buf, 5) // max_token_length

	entries := []struct {
		score float32
		piece string
	}{
		{0, "<unk>"},
		{0, "\n<s>\n"},
		{0, "\n</s>\n"},
		{1.5, "hi"},
	}
	for _, e := range entries {
		appendU32(buf, math.Float32bits(e.score))
		appendU32(buf, uint32(len(e.piece)))
		buf.WriteString(e.piece)
	}

	tok, err := Load(bytes.NewReader(buf.Bytes()), len(entries))
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if tok.MaxTokenLength != 5 {
		t.Fatalf("MaxTokenLength = %d, want 5", tok.MaxTokenLength)
	}
	if tok.Vocab[3] != "hi" || tok.Scores[3] != 1.5 {
		t.Fatalf("Vocab[3]=%q Scores[3]=%v, want %q / %v", tok.Vocab[3], tok.Scores[3], "hi", 1.5)
	}
	if id, ok := tok.lookup("hi"); !ok || id != 3 {
		t.Fatalf("lookup(%q) = (%d, %v), want (3, true)", "hi", id, ok)
	}
}

func TestLoadTruncatedInputErrors(t *testing.T) {
	buf := new(bytes.Buffer)
	appendU32(buf, 5)
	appendU32(buf, math.Float32bits(0))
	appendU32(buf, 100) // claims 100-byte token but provides none
	if _, err := Load(bytes.NewReader(buf.Bytes()), 1); err == nil {
		t.Fatalf("Load() on truncated token body: got nil error, want error")
	}
}
