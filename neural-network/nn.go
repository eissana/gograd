package nn

import (
	"math/rand"
)

// Makes a neuron from a list of values and an activation function.
// Output of a neuron is a singkle value.
func MakeNeuron(input []*Value, activation func(*Value) *Value) *Value{
	// compute w_1 * x_1 + ... + w_n * x_n + b
	ans := MakeValue(/*intercept=*/0.0)
	for _, x := range input {
		// initialize with random weights in [-1, 1)
		w := MakeValue(rand.Float64()*2.0 - 1.0)
		ans = Add(ans, Mul(w, x))
	}
	// apply activation if given.
	if activation != nil {
		ans = activation(ans)
	}
	return ans
}

// Parameters of a layer including the number of neurons in the layer and
// the activation function.
type LayerParam struct {
	outputSize int
	activation func(*Value) *Value
}

// Makes LayerParam object.
func MakeLayerParam(numNeurons int, activation func(*Value) *Value) LayerParam {
	return LayerParam {
		outputSize: numNeurons,
		activation: activation,
	}
}

// Makes a layer consisting of multiple neurons.
func MakeLayer(input []*Value, layerParam LayerParam) []*Value {
	ans := make([]*Value, layerParam.outputSize)
	for i := range ans {
		ans[i] = MakeNeuron(input, layerParam.activation)
	}
	return ans
}

// Makes a neural network consisting of multiple layers.
func MakeNeuralNetwork(input []*Value, layerParams []LayerParam) []*Value {
	layer := input
	for _, layerParam := range layerParams {
		layer = MakeLayer(layer, layerParam)
	}
	return layer
}