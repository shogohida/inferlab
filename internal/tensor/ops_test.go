package tensor

import (
	"math"
	"testing"
)

func approxEqual(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

func TestMatMul(t *testing.T) {
	// W = [[1,2,3],[4,5,6]] (d=2, n=3), x = [1,1,1]
	w := []float32{1, 2, 3, 4, 5, 6}
	x := []float32{1, 1, 1}
	out := make([]float32, 2)
	MatMul(out, x, w, 3, 2)
	want := []float32{6, 15}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("MatMul()[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestBatchMatMulMatchesPerRowMatMul(t *testing.T) {
	w := []float32{1, 0, -1, 2, 0.5, 0.5, -2, 3}
	n, d, nSeqs := 4, 2, 3
	x := []float32{
		1, 2, 3, 4,
		-1, 0, 1, 0.5,
		0, 0, 0, 1,
	}
	batched := make([]float32, nSeqs*d)
	BatchMatMul(batched, x, w, nSeqs, n, d)

	for s := 0; s < nSeqs; s++ {
		want := make([]float32, d)
		MatMul(want, x[s*n:s*n+n], w, n, d)
		for i := 0; i < d; i++ {
			if batched[s*d+i] != want[i] {
				t.Fatalf("seq %d: BatchMatMul = %v, want %v (from per-row MatMul)", s, batched[s*d+i], want[i])
			}
		}
	}
}

func TestAdd(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{10, 20, 30}
	out := make([]float32, 3)
	Add(out, a, b)
	want := []float32{11, 22, 33}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("Add()[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestRMSNorm(t *testing.T) {
	x := []float32{3, 4}
	weight := []float32{1, 1}
	out := make([]float32, 2)
	RMSNorm(out, x, weight)
	// mean(x^2) = (9+16)/2 = 12.5, scale = 1/sqrt(12.5+eps)
	scale := float32(1.0 / math.Sqrt(12.5+1e-5))
	want := []float32{3 * scale, 4 * scale}
	for i := range want {
		if !approxEqual(out[i], want[i], 1e-5) {
			t.Fatalf("RMSNorm()[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestSoftmaxSumsToOne(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	Softmax(x)
	var sum float32
	for _, v := range x {
		sum += v
	}
	if !approxEqual(sum, 1, 1e-5) {
		t.Fatalf("Softmax sums to %v, want 1", sum)
	}
	// monotonic: larger input -> larger probability
	for i := 1; i < len(x); i++ {
		if x[i] <= x[i-1] {
			t.Fatalf("Softmax not monotonic at %d: %v", i, x)
		}
	}
}

func TestSoftmaxShiftInvariant(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{101, 102, 103}
	Softmax(a)
	Softmax(b)
	for i := range a {
		if !approxEqual(a[i], b[i], 1e-4) {
			t.Fatalf("Softmax not shift-invariant at %d: %v vs %v", i, a[i], b[i])
		}
	}
}

func TestSoftmaxAllEqual(t *testing.T) {
	x := []float32{5, 5, 5, 5}
	Softmax(x)
	for _, v := range x {
		if !approxEqual(v, 0.25, 1e-5) {
			t.Fatalf("Softmax of equal inputs = %v, want uniform 0.25", x)
		}
	}
}

func TestSiLUKnownValues(t *testing.T) {
	x := []float32{0}
	SiLU(x)
	if !approxEqual(x[0], 0, 1e-6) {
		t.Fatalf("SiLU(0) = %v, want 0", x[0])
	}

	// SiLU(x) -> x as x -> +inf (sigmoid -> 1)
	big := []float32{20}
	SiLU(big)
	if !approxEqual(big[0], 20, 1e-3) {
		t.Fatalf("SiLU(20) = %v, want ~20", big[0])
	}

	// SiLU(x) -> 0 as x -> -inf
	neg := []float32{-20}
	SiLU(neg)
	if !approxEqual(neg[0], 0, 1e-3) {
		t.Fatalf("SiLU(-20) = %v, want ~0", neg[0])
	}
}
