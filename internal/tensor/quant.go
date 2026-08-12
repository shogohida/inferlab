package tensor

import "math"

// QuantizedTensor holds a (Rows, Cols) weight matrix as int8 values with one
// float32 scale per row (per-output-channel symmetric quantization). Per-row
// scaling is simpler than blockwise/groupwise schemes (as used by GGUF) but
// meaningfully more accurate than a single scale for the whole matrix: one
// row with an unusually large weight doesn't blow out the precision of every
// other row. Activations stay float32 — this is weight-only quantization,
// so QuantMatMul dequantizes on the fly rather than requiring a separate
// int8 activation path.
type QuantizedTensor struct {
	Data   []int8
	Scales []float32
	Rows   int
	Cols   int
}

// Quantize converts a (rows, cols) row-major float32 matrix into per-row
// symmetric int8. Each row's scale is max(|row|)/127; a row of all zeros
// gets scale 1 (arbitrary but safe — every quantized value in that row is 0
// regardless of scale, so no division by zero occurs).
func Quantize(w []float32, rows, cols int) *QuantizedTensor {
	q := &QuantizedTensor{
		Data:   make([]int8, rows*cols),
		Scales: make([]float32, rows),
		Rows:   rows,
		Cols:   cols,
	}
	for i := 0; i < rows; i++ {
		row := w[i*cols : i*cols+cols]
		var maxAbs float32
		for _, v := range row {
			a := v
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		scale := float32(1.0)
		if maxAbs > 0 {
			scale = maxAbs / 127
		}
		q.Scales[i] = scale
		out := q.Data[i*cols : i*cols+cols]
		for j, v := range row {
			iv := int32(math.Round(float64(v / scale)))
			if iv > 127 {
				iv = 127
			} else if iv < -127 {
				iv = -127
			}
			out[j] = int8(iv)
		}
	}
	return q
}

// DequantizeRow expands just row i back to a flat float32 slice of length
// Cols — used for embedding-table lookups, where only one row is ever
// needed per token rather than the whole table.
func (q *QuantizedTensor) DequantizeRow(i int) []float32 {
	out := make([]float32, q.Cols)
	scale := q.Scales[i]
	src := q.Data[i*q.Cols : i*q.Cols+q.Cols]
	for j, v := range src {
		out[j] = float32(v) * scale
	}
	return out
}

// Dequantize expands q back to a flat float32 (rows*cols) row-major matrix.
func (q *QuantizedTensor) Dequantize() []float32 {
	out := make([]float32, q.Rows*q.Cols)
	for i := 0; i < q.Rows; i++ {
		scale := q.Scales[i]
		src := q.Data[i*q.Cols : i*q.Cols+q.Cols]
		dst := out[i*q.Cols : i*q.Cols+q.Cols]
		for j, v := range src {
			dst[j] = float32(v) * scale
		}
	}
	return out
}

// QuantBatchMatMul is QuantMatMul applied independently to nSeqs rows of x
// against the shared quantized weight matrix q, the quantized counterpart
// to BatchMatMul — bit-identical to calling QuantMatMul once per row.
func QuantBatchMatMul(out, x []float32, q *QuantizedTensor, nSeqs, n, d int) {
	for s := 0; s < nSeqs; s++ {
		QuantMatMul(out[s*d:s*d+d], x[s*n:s*n+n], q, n, d)
	}
}

// QuantMatMul computes out = Q @ x the same way MatMul does for a dense
// weight matrix, dequantizing each row's int8 values on the fly:
// out[i] = scale[i] * sum_j Q.Data[i*n+j] * x[j].
func QuantMatMul(out, x []float32, q *QuantizedTensor, n, d int) {
	for i := 0; i < d; i++ {
		var val float32
		row := q.Data[i*n : i*n+n]
		for j := 0; j < n; j++ {
			val += float32(row[j]) * x[j]
		}
		out[i] = val * q.Scales[i]
	}
}
