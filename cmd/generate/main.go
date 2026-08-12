// Command generate is a local CLI for greedy text generation, mainly useful
// for sanity-checking a checkpoint end to end during development: if the
// transformer math has a bug, the output here is visibly garbage long
// before any of the cache/quant/batch layers get built on top of it.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"inferlab/internal/loader"
	"inferlab/internal/model"
	"inferlab/internal/quant"
	"inferlab/internal/tokenizer"
)

func main() {
	checkpointPath := flag.String("checkpoint", "dev-weights/stories15M.bin", "path to a llama2.c-format checkpoint")
	tokenizerPath := flag.String("tokenizer", "dev-weights/tokenizer.bin", "path to tokenizer.bin")
	prompt := flag.String("prompt", "Once upon a time", "generation prompt")
	maxTokens := flag.Int("max-tokens", 60, "maximum tokens to generate")
	quantize := flag.Bool("quantize", false, "run generation over int8 weight-only quantized weights instead of fp32")
	flag.Parse()

	ckFile, err := os.Open(*checkpointPath)
	if err != nil {
		log.Fatalf("open checkpoint: %v", err)
	}
	defer ckFile.Close()
	ck, err := loader.Load(ckFile)
	if err != nil {
		log.Fatalf("load checkpoint: %v", err)
	}
	fmt.Fprintf(os.Stderr, "config: %+v\n", ck.Config)

	tokFile, err := os.Open(*tokenizerPath)
	if err != nil {
		log.Fatalf("open tokenizer: %v", err)
	}
	defer tokFile.Close()
	tok, err := tokenizer.Load(tokFile, ck.Config.VocabSize)
	if err != nil {
		log.Fatalf("load tokenizer: %v", err)
	}

	var m *model.Model
	if *quantize {
		qw := quant.Quantize(ck)
		fmt.Fprintf(os.Stderr, "quantized weights: %.1f MB (fp32 checkpoint was larger on disk)\n", float64(qw.ByteSize())/1e6)
		m = model.NewFromWeights(ck.Config, qw)
	} else {
		m = model.New(ck)
	}
	keyCache, valCache := m.NewCacheBuffers()

	promptTokens := tok.Encode(*prompt, true, false)
	var generated []int32
	var logits []float32
	pos := 0
	for _, t := range promptTokens {
		if pos >= ck.Config.SeqLen {
			break
		}
		logits = m.Step(t, pos, keyCache, valCache)
		generated = append(generated, t)
		pos++
	}

	for pos < ck.Config.SeqLen && len(generated)-len(promptTokens) < *maxTokens {
		next := argmax(logits)
		generated = append(generated, next)
		if next == 2 { // EOS
			break
		}
		logits = m.Step(next, pos, keyCache, valCache)
		pos++
	}

	fmt.Println(tok.DecodeSequence(generated))
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
