package nn

import (
	"testing"
	"github.com/stretchr/testify/assert"
)

func TestValue(t *testing.T) {
	x1 := MakeValue(4.0)
	x2 := MakeValue(2.0)
	x3 := Add(x1, x2)

	expected := x1.GetData() + x2.GetData()
	assert.Equalf(t, expected, x3.GetData(), "expected %f, got %f", expected, x3.GetData())
	assert.Equalf(t, 0.0, x3.GetGrad(), "expected %f, got %f", 0.0, x3.GetGrad())
	assert.Equalf(t, "+", x3.GetOp(), "expected %f, got %f", "+", x3.GetOp())

	x4 := MakeValue(3.0)
	x5 := Mul(x3, x4)

	expected = x3.GetData() * x4.GetData()
	assert.Equalf(t, expected, x5.GetData(), "expected %f, got %f", expected, x5.GetData())
	assert.Equalf(t, 0.0, x5.GetGrad(), "expected %f, got %f", 0.0, x5.GetGrad())
	assert.Equalf(t, "*", x5.GetOp(), "expected %f, got %f", "*", x5.GetOp())

	x6 := Relu(x5)

	expected = x5.GetData()
	if expected < 0.0 {
		expected = 0.0
	}
	assert.Equalf(t, expected, x6.GetData(), "expected %f, got %f", expected, x6.GetData())
	assert.Equalf(t, 0.0, x6.GetGrad(), "expected %f, got %f", 0.0, x6.GetGrad())
	assert.Equalf(t, "ReLU", x6.GetOp(), "expected %f, got %f", "ReLU", x6.GetOp())
	
	BackPropagate(x6)

	assert.Equalf(t, 3.0, x1.GetGrad(), "expected %f, got %f", 3.0, x1.GetGrad())
	assert.Equalf(t, 3.0, x2.GetGrad(), "expected %f, got %f", 3.0, x2.GetGrad())
	assert.Equalf(t, 3.0, x3.GetGrad(), "expected %f, got %f", 3.0, x3.GetGrad())
	assert.Equalf(t, 6.0, x4.GetGrad(), "expected %f, got %f", 6.0, x4.GetGrad())
	assert.Equalf(t, 1.0, x5.GetGrad(), "expected %f, got %f", 1.0, x5.GetGrad())
	assert.Equalf(t, 1.0, x6.GetGrad(), "expected %f, got %f", 1.0, x6.GetGrad())
}