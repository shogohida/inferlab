// Package tensor implements the numeric primitives the rest of inferlab is
// built on: matrix-vector and batched matrix-matrix products, normalization,
// activations, and int8 weight-only quantization. Everything here operates
// on flat []float32 slices with explicit dimensions rather than a general
// N-dimensional tensor abstraction — the shapes involved (a weight matrix
// times an activation vector) are fixed and few, so a generic tensor
// framework would be overhead this project doesn't need.
package tensor

// Tensor is a flat, row-major buffer with an explicit shape. It's used to
// carry loaded model weights around with their dimensions attached; the
// compute kernels below take raw []float32 slices plus explicit dimension
// arguments instead of a *Tensor, matching the reference C implementation's
// style and keeping the hot path free of shape-indexing overhead.
type Tensor struct {
	Data  []float32
	Shape []int
}

// New allocates a zeroed Tensor with the given shape.
func New(shape ...int) *Tensor {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return &Tensor{Data: make([]float32, n), Shape: append([]int(nil), shape...)}
}

// Len returns the total number of elements implied by Shape.
func (t *Tensor) Len() int {
	return len(t.Data)
}
