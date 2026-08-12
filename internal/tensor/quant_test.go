package tensor

import (
	"math/rand"
	"testing"
)

func TestQuantizeDequantizeRoundTripBounded(t *testing.T) {
	rows, cols := 4, 8
	w := make([]float32, rows*cols)
	r := rand.New(rand.NewSource(1))
	for i := range w {
		w[i] = (r.Float32()*2 - 1) * 10 // uniform in [-10, 10]
	}

	q := Quantize(w, rows, cols)
	got := q.Dequantize()

	for i := 0; i < rows; i++ {
		tol := q.Scales[i]/2 + 1e-6 // rounding error bounded by half a quantization step
		for j := 0; j < cols; j++ {
			idx := i*cols + j
			if !approxEqual(w[idx], got[idx], tol) {
				t.Fatalf("row %d col %d: quantize/dequantize error %v exceeds tolerance %v (orig=%v, got=%v)",
					i, j, w[idx]-got[idx], tol, w[idx], got[idx])
			}
		}
	}
}

func TestQuantizeAllZeroRowNoDivideByZero(t *testing.T) {
	w := []float32{0, 0, 0, 0}
	q := Quantize(w, 1, 4)
	if q.Scales[0] == 0 {
		t.Fatalf("scale for all-zero row is 0, would divide by zero on dequantize")
	}
	for _, v := range q.Data {
		if v != 0 {
			t.Fatalf("all-zero row quantized to nonzero value %v", v)
		}
	}
}

func TestQuantMatMulCloseToFP32(t *testing.T) {
	rows, cols := 6, 16
	r := rand.New(rand.NewSource(2))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = (r.Float32()*2 - 1) * 2
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = (r.Float32()*2 - 1) * 2
	}

	want := make([]float32, rows)
	MatMul(want, x, w, cols, rows)

	q := Quantize(w, rows, cols)
	got := make([]float32, rows)
	QuantMatMul(got, x, q, cols, rows)

	for i := range want {
		// relative tolerance: int8 quantization at this scale should stay
		// within a few percent of the fp32 reference for well-conditioned
		// random inputs.
		tol := abs32(want[i])*0.05 + 0.05
		if !approxEqual(want[i], got[i], tol) {
			t.Fatalf("row %d: QuantMatMul = %v, fp32 MatMul = %v, exceeds tolerance %v", i, got[i], want[i], tol)
		}
	}
}

func TestQuantizePerRowBeatsPerTensorOnOutlierRow(t *testing.T) {
	// Row 0 has a huge outlier; row 1 is small and uniform. A single
	// per-tensor scale derived from the outlier would crush row 1 to
	// near-zero precision. Per-row quantization keeps row 1 accurate.
	w := []float32{
		100, -100, 0.001, 0.002, // row 0: dominated by outliers
		0.01, -0.01, 0.02, -0.02, // row 1: small, uniform
	}
	q := Quantize(w, 2, 4)
	got := q.Dequantize()

	row1Tol := q.Scales[1]/2 + 1e-6
	for j := 0; j < 4; j++ {
		idx := 1*4 + j
		if !approxEqual(w[idx], got[idx], row1Tol) {
			t.Fatalf("row 1 (small row) col %d: error %v exceeds per-row tolerance %v — per-row scaling failed to protect it from row 0's outliers",
				j, w[idx]-got[idx], row1Tol)
		}
	}
	if q.Scales[1] >= q.Scales[0] {
		t.Fatalf("expected row 1's scale (%v) to be much smaller than row 0's (%v)", q.Scales[1], q.Scales[0])
	}
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
