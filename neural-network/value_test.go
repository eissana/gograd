package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestValue1(t *testing.T) {
	x, y := MakeValue(2.0), MakeValue(3.0)
	z := x.Add(x.Mul(y))
	z.BackPropagate()

	tol := 1e-4
	assert.InDelta(t, 2.0, x.GetData(), tol, "expected %f, got %f", 2.0, x.GetData())
	assert.InDelta(t, 3.0, y.GetData(), tol, "expected %f, got %f", 3.0, y.GetData())
	assert.InDelta(t, 8.0, z.GetData(), tol, "expected %f, got %f", 8.0, z.GetData())

	assert.InDelta(t, 4.0, x.GetGrad(), tol, "expected %f, got %f", 4.0, x.GetGrad())
	assert.InDelta(t, 2.0, y.GetGrad(), tol, "expected %f, got %f", 2.0, y.GetGrad())
	assert.InDelta(t, 1.0, z.GetGrad(), tol, "expected %f, got %f", 1.0, z.GetGrad())
}

func TestValue2(t *testing.T) {
	x1 := MakeValue(4.0)
	x2 := MakeValue(2.0)
	x3 := x1.Add(x2)

	expected := x1.GetData() + x2.GetData()
	assert.Equalf(t, expected, x3.GetData(), "expected %f, got %f", expected, x3.GetData())
	assert.Equalf(t, 0.0, x3.GetGrad(), "expected %f, got %f", 0.0, x3.GetGrad())
	assert.Equalf(t, "+", x3.GetOp(), "expected %f, got %f", "+", x3.GetOp())

	x4 := MakeValue(3.0)
	x5 := x3.Mul(x4)

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

	x6.BackPropagate()

	assert.Equalf(t, 3.0, x1.GetGrad(), "expected %f, got %f", 3.0, x1.GetGrad())
	assert.Equalf(t, 3.0, x2.GetGrad(), "expected %f, got %f", 3.0, x2.GetGrad())
	assert.Equalf(t, 3.0, x3.GetGrad(), "expected %f, got %f", 3.0, x3.GetGrad())
	assert.Equalf(t, 6.0, x4.GetGrad(), "expected %f, got %f", 6.0, x4.GetGrad())
	assert.Equalf(t, 1.0, x5.GetGrad(), "expected %f, got %f", 1.0, x5.GetGrad())
	assert.Equalf(t, 1.0, x6.GetGrad(), "expected %f, got %f", 1.0, x6.GetGrad())
}

func TestValue3(t *testing.T) {
	values := [5]*Value{}
	values[0] = MakeValue(-4.0)
	values[1] = values[0].Add(MakeValue(2).Add(MakeValue(2).Mul(values[0])))
	values[2] = Relu(values[1]).Add(values[1].Mul(values[0]))
	values[3] = Relu(values[1].Mul(values[1]))
	values[4] = values[3].Add(values[2]).Add(values[2].Mul(values[0]))
	values[4].BackPropagate()

	expectedValues := [5]float64{-4, -10, 40, 100, -20}
	expectedGrad0 := 46.0

	tol := 1e-4
	for i, value := range values {
		assert.InDelta(t, expectedValues[i], value.GetData(), tol, "expected %f, got %f", expectedValues[i], value.GetData())
	}

	assert.InDelta(t, expectedGrad0, values[0].GetGrad(), 0.001, "expected %f, got %f", expectedGrad0, values[0].GetGrad())
}

func TestValue4(t *testing.T) {
	values := [7]*Value{}
	values[0] = MakeValue(-4.0)
	values[1] = MakeValue(2.0)
	values[2] = values[0].Add(values[1])
	values[3] = values[0].Mul(values[1]).Add(values[1].Pow(3.0))
	values[2] = values[2].Add(values[2]).Add(MakeValue(1.0))
	values[2] = values[2].Add(MakeValue(1.0)).Add(values[2]).Sub(values[0])
	values[3] = values[3].Add(values[3].Mul(MakeValue(2.0))).Add(Relu(values[1].Add(values[0])))
	values[3] = values[3].Add(MakeValue(3.0).Mul(values[3])).Add(Relu(values[1].Sub(values[0])))
	values[4] = values[2].Sub(values[3])
	values[5] = values[4].Pow(2.0)
	values[6] = values[5].Div(MakeValue(2.0))
	values[6] = values[6].Add(MakeValue(10.0).Div(values[5]))
	values[6].BackPropagate()

	expectedValues := [7]float64{-4, 2.0, -1.0, 6.0, -7.0, 49.0, 24.7041}
	expectedGrad := [2]float64{138.8338, 645.5773}

	tol := 1e-4
	for i, value := range values {
		assert.InDelta(t, expectedValues[i], value.GetData(), tol, "expected %f, got %f", expectedValues[i], value.GetData())
	}

	for i, grad := range expectedGrad {
		assert.InDelta(t, grad, values[i].GetGrad(), 0.001, "expected %f, got %f", grad, values[i].GetGrad())
	}
}
