package tensor

import "math"

// MatMul computes out = W @ x, where W is a (d, n) row-major matrix and x is
// a length-n vector: out[i] = sum_j W[i*n+j] * x[j] for i in [0, d).
// out must have length d, x length n, w length d*n.
func MatMul(out, x, w []float32, n, d int) {
	for i := 0; i < d; i++ {
		var val float32
		row := w[i*n : i*n+n]
		for j := 0; j < n; j++ {
			val += row[j] * x[j]
		}
		out[i] = val
	}
}

// BatchMatMul computes out = X @ W^T for nSeqs independent rows of X (each
// length n) against the shared (d, n) weight matrix W, writing nSeqs rows of
// length d into out. It is numerically equivalent to calling MatMul once per
// row of X — this is deliberate: batching here is about sharing one streamed
// pass over W across many activation rows (the actual memory-bandwidth win
// internal/batch relies on), not about reassociating the summation order, so
// the batched and per-row results match bit-for-bit.
func BatchMatMul(out, x, w []float32, nSeqs, n, d int) {
	for s := 0; s < nSeqs; s++ {
		MatMul(out[s*d:s*d+d], x[s*n:s*n+n], w, n, d)
	}
}

// Add computes dst[i] = a[i] + b[i] elementwise. dst may alias a or b.
func Add(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

// RMSNorm computes root-mean-square layer normalization: out[j] =
// weight[j] * x[j] / sqrt(mean(x^2) + eps). out may alias x.
func RMSNorm(out, x, weight []float32) {
	const eps = 1e-5
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	ss /= float32(len(x))
	scale := float32(1.0 / math.Sqrt(float64(ss+eps)))
	for j, v := range x {
		out[j] = weight[j] * (scale * v)
	}
}

// Softmax normalizes x in place into a probability distribution, subtracting
// the row max first for numerical stability.
func Softmax(x []float32) {
	if len(x) == 0 {
		return
	}
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range x {
		e := float32(math.Exp(float64(v - max)))
		x[i] = e
		sum += e
	}
	if sum == 0 {
		return
	}
	for i := range x {
		x[i] /= sum
	}
}

// SiLU applies the sigmoid linear unit x * sigmoid(x) elementwise, in place.
func SiLU(x []float32) {
	for i, v := range x {
		x[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
}
