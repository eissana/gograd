package nn

import (
	"math"
)

// Given a function f and its derivative g, returns an activation function.
func MakeActivation(op string, f, g func(float64) float64) func(*Value) *Value {
	return func(value *Value) *Value {
		data := f(value.data)
		ans := &Value{
			data:     data,
			op:       op,
			children: []*Value{value},
		}
		ans.backward = func() {
			value.grad += g(value.data) * ans.grad
		}
		return ans
	}
}

// Rectified linear unit: y = max(0, x)
func Relu(value *Value) *Value {
	f := func(x float64) float64 {
		if x > 0.0 {
			return x
		}
		return 0.0
	}
	g := func(x float64) float64 {
		if x > 0.0 {
			return 1.0
		}
		return 0.0
	}
	return MakeActivation("ReLU", f, g)(value)
}

// Sigmoid function: y = 1/(1 + exp(-x))
func Sigmoid(value *Value) *Value {
	f := func(x float64) float64 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	g := func(x float64) float64 {
		y := f(x)
		return y * (1 - y)
	}
	return MakeActivation("Sigmoid", f, g)(value)
}

// Hyperbolic tangent (tanh): y = (exp(2x) - 1) / (exp(2x) + 1)
func Tanh(value *Value) *Value {
	f := func(x float64) float64 {
		y := math.Exp(2 * x)
		return (y - 1.0) / (y + 1.0)
	}
	g := func(x float64) float64 {
		y := f(x)
		return 1.0 - y*y
	}
	return MakeActivation("Tanh", f, g)(value)
}

// Exponent: y = exp(x)
func Exp(value *Value) *Value {
	f := func(x float64) float64 {
		return math.Exp(x)
	}
	g := func(x float64) float64 {
		return f(x)
	}
	return MakeActivation("Exp", f, g)(value)
}
