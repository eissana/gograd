package nn

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

const (
	delta = 0.001
)

func init() {
	rand.Seed(123456)
}

func TestNeuron(t *testing.T) {
	model := MakeNeuron(2, Relu)
	input := []*Value{MakeValue(2), MakeValue(1)}
	output := model.Fit(input)

	assert.InDelta(t, 4.0723, output.GetData(), delta, "expected %f, got %f", 4.0723, output.GetData())

	output.BackPropagate()
}

func TestLayer(t *testing.T) {
	layerParam := MakeLayerParam(3, Tanh)
	model := MakeLayer(2, layerParam)

	input := []*Value{MakeValue(2), MakeValue(1)}
	output := model.Fit(input)

	assert.Equal(t, 3, len(output), "expected %d, got %d", 3, len(output))
	assert.InDelta(t, 0.2, output[0].GetData(), delta, "expected %f, got %f", 0.2, output[0].GetData())
	assert.InDelta(t, -0.729, output[1].GetData(), delta, "expected %f, got %f", -0.729, output[1].GetData())
	assert.InDelta(t, 0.099, output[2].GetData(), delta, "expected %f, got %f", 0.099, output[2].GetData())

	output[0].BackPropagate()
}

func TestNeuralNetwork(t *testing.T) {
	layerParams := []LayerParam{
		MakeLayerParam(3, Relu),
		MakeLayerParam(3, Relu),
		MakeLayerParam(1, Sigmoid),
	}
	model := MakeNeuralNetwork(2, layerParams)

	input := []*Value{MakeValue(3.1), MakeValue(1.2)}
	output := model.Fit(input)

	assert.Equal(t, 1, len(output), "expected %d, got %d", 1, len(output))
	assert.InDelta(t, 0.567, output[0].GetData(), delta, "expected %f, got %f", 0.567, output[0].GetData())

	output[0].BackPropagate()
}
