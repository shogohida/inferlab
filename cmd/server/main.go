// Command server runs the inferlab demo: it serves the embedded frontend
// and the /api/generate and /api/benchmark endpoints on a single port — the
// same free-tier-friendly shape used by the sibling raftkv/sqllab/routelab
// demos. The checkpoint and tokenizer are loaded once at startup from local
// files (fetched into place by render.yaml's buildCommand at deploy time —
// see its comment for why the weights aren't committed to this repo).
package main

import (
	"log"
	"net/http"
	"os"

	"inferlab/internal/api"
	"inferlab/internal/loader"
	"inferlab/internal/tokenizer"
	"inferlab/web"
)

func main() {
	checkpointPath := getenv("CHECKPOINT_PATH", "weights/stories15M.bin")
	tokenizerPath := getenv("TOKENIZER_PATH", "weights/tokenizer.bin")

	ck := mustLoadCheckpoint(checkpointPath)
	log.Printf("inferlab: loaded checkpoint %s: %+v", checkpointPath, ck.Config)

	tok := mustLoadTokenizer(tokenizerPath, ck.Config.VocabSize)
	log.Printf("inferlab: loaded tokenizer %s (%d tokens)", tokenizerPath, len(tok.Vocab))

	handler := api.New(ck, tok)

	mux := http.NewServeMux()
	mux.Handle("/api/", handler.Routes())
	mux.Handle("/", http.FileServer(http.FS(web.Assets)))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	addr := ":" + port
	log.Printf("inferlab: serving on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("HTTP server died: %v", err)
	}
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func mustLoadCheckpoint(path string) *loader.Checkpoint {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("inferlab: open checkpoint %s: %v", path, err)
	}
	defer f.Close()
	ck, err := loader.Load(f)
	if err != nil {
		log.Fatalf("inferlab: load checkpoint %s: %v", path, err)
	}
	return ck
}

func mustLoadTokenizer(path string, vocabSize int) *tokenizer.Tokenizer {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("inferlab: open tokenizer %s: %v", path, err)
	}
	defer f.Close()
	tok, err := tokenizer.Load(f, vocabSize)
	if err != nil {
		log.Fatalf("inferlab: load tokenizer %s: %v", path, err)
	}
	return tok
}
