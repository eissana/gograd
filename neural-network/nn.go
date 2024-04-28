package nn

import (
	"math/rand"
)

// A neural network object consitsting of multiple layers.
type NeuralNetwork struct {
	layers []*Layer
}

// A neuron object with parameters w_1, ..., w_n, b.
type Neuron struct {
	intercept *Value
	weights   []*Value
}

// Makes a neuron with a given inputSize. A neuron has inputSize+1 parameters.
// The intercept is initialized to 0 and weights are initialized to random numbers
// in [-1, 1).
func MakeNeuron(inputSize int) *Neuron {
	weights := make([]*Value, inputSize)
	for i := range weights {
		weights[i] = MakeValue(rand.NormFloat64())
	}
	return &Neuron{
		intercept: MakeValue(rand.NormFloat64()),
		weights:   weights,
	}
}

// Computes output of a neuron as activation(w_1*x_1 + ... + w_n*x_n + b).
func (n *Neuron) Fit(input []*Value) *Value {
	// compute w_1 * x_1 + ... + w_n * x_n + b
	ans := n.intercept
	for i, x := range input {
		ans = ans.Add(x.Mul(n.weights[i]))
	}
	return ans
}

// Parameters of a layer: outputSize AKA the number of neurons in the layer.
// Each layer can have a different activation function.
type LayerParam struct {
	outputSize int
	// The number of activation functions must match the number of
	activation func(*Value) *Value
}

// Makes a LayerParam object with a given outputSize (number of neurons) and
// an activation function.
func MakeLayerParam(outputSize int, activation func(*Value) *Value) LayerParam {
	return LayerParam{
		outputSize: outputSize,
		activation: activation,
	}
}

// A layer object consisting of multiple neurons.
type Layer struct {
	neurons    []*Neuron
	activation func(*Value) *Value
}

// Makes a layer consisting of multiple neurons.
func MakeLayer(inputSize int, layerParam LayerParam) *Layer {
	neurons := make([]*Neuron, layerParam.outputSize)
	for i := range neurons {
		neurons[i] = MakeNeuron(inputSize)
	}
	return &Layer{
		neurons:    neurons,
		activation: layerParam.activation,
	}
}

// Computes all output values of the layer given the input values and an
// activation function.
func (l *Layer) Fit(input []*Value) []*Value {
	ans := make([]*Value, len(l.neurons))
	for i, neuron := range l.neurons {
		ans[i] = neuron.Fit(input)
	}
	// Fit activation if given.
	if l.activation != nil {
		for i := range ans {
			ans[i] = l.activation(ans[i])
		}
	}
	return ans
}

// Makes a neural network consisting of multiple layers.
func MakeNeuralNetwork(inputSize int, layerParams []LayerParam) *NeuralNetwork {
	layers := make([]*Layer, len(layerParams))
	for i, layerParam := range layerParams {
		layers[i] = MakeLayer(inputSize, layerParam)
		inputSize = layerParam.outputSize
	}
	return &NeuralNetwork{
		layers: layers,
	}
}

// Fits the model on input data and return the score.
func (n *NeuralNetwork) Fit(input []*Value) []*Value {
	ans := input
	for _, layer := range n.layers {
		ans = layer.Fit(ans)
	}
	return ans
}

// Computes scores of all input data.
func (n *NeuralNetwork) Forward(inputs [][]*Value) [][]*Value {
	scores := make([][]*Value, len(inputs))
	for i, input := range inputs {
		scores[i] = n.Fit(input)
	}
	return scores
}

// Computes the loss as a Value object which is minimized in the optimization
// process when traininng the model.
func (n *NeuralNetwork) Loss(labels, scores [][]*Value, trainingParam TrainingParam) *Value {
	floatNumRecords := float64(len(scores))
	// Initializing loss = 1/batchSize. Will update loss in the following loop.
	loss := MakeValue(0.0)

	for i := range scores {
		// Hinge loss: loss += Relu(1 - label * score) where label is in {-1, 1}
		// loss = loss.Add(Relu(MakeValue(1.0).Sub(labels[i].Mul(score[0]))))
		normalize(scores[i])
		for j, score := range scores[i] {
			// cross-entropy loss
			label := labels[i][j]
			pos := label.Mul(score.Log())
			neg := MakeValue(1).Sub(label).Mul(MakeValue(1).Sub(score).Log())
			loss = loss.Sub(pos.Add(neg))
		}
	}
	// accuracy /= floatNumRecords
	loss = loss.Div(MakeValue(floatNumRecords))

	regularizationParam := trainingParam.Regularization
	if regularizationParam > 0.0 {
		// Regularization term
		norm2Loss := MakeValue(0.0)
		for _, layer := range n.layers {
			for _, neuron := range layer.neurons {
				norm2Loss = norm2Loss.Add(neuron.intercept.Pow(2))
				for _, weight := range neuron.weights {
					norm2Loss = norm2Loss.Add(weight.Pow(2))
				}
			}
		}
		norm2Loss = norm2Loss.Mul(MakeValue(regularizationParam))
		loss = loss.Add(norm2Loss)
	}
	return loss
}

func normalize(score []*Value) {
	if len(score) < 2 {
		return
	}
	sum := MakeValue(0.0)
	for i := range score {
		score[i] = score[i].Exp()
		sum = sum.Add(score[i])
	}
	for i := range score {
		score[i] = score[i].Div(sum)
	}
}

// TrainingParam holds parameters required for training the network.
type TrainingParam struct {
	Epochs                  int
	Regularization          float64
	ClassificationThreshold float64
	LearningRate            float64
}

// Trains the network by minimizing the loss function
func (n *NeuralNetwork) Train(inputs, labels [][]*Value, trainingParam TrainingParam) ([]float64, [][]*Value) {
	scores := [][]*Value{}
	losses := make([]float64, trainingParam.Epochs)
	for i := 0; i < trainingParam.Epochs; i++ {
		scores = n.Forward(inputs)
		loss := n.Loss(labels, scores, trainingParam)
		losses[i] = loss.GetData()

		n.ResetGrad()
		loss.BackPropagate()

		n.NextData(trainingParam.LearningRate)
	}
	return losses, scores
}

// Resets grad values of the entire network recursively.
func (n *NeuralNetwork) ResetGrad() {
	for _, layer := range n.layers {
		for _, neuron := range layer.neurons {
			neuron.intercept.grad = 0.0
			for _, weight := range neuron.weights {
				weight.grad = 0.0
			}
		}
	}
}

// Moves in the direction of the gradient descent and updates model.
func (n *NeuralNetwork) NextData(learningRate float64) {
	for _, layer := range n.layers {
		for _, neuron := range layer.neurons {
			neuron.intercept.data -= learningRate * neuron.intercept.grad
			for _, weight := range neuron.weights {
				weight.data -= learningRate * weight.grad
			}
		}
	}
}

// Computes the accuracy of a model given scores and labels. It also requires a
// classification threshold.
func Accuracy(scores, labels [][]*Value, trainingParam TrainingParam) (accuracy float64) {
	threshold := trainingParam.ClassificationThreshold
	for i, score := range scores {
		label := labels[i]
		// label is 0.0 or 1.0, while score is in [0, 1] range.
		if (label[0].GetData() > threshold) == (score[0].GetData() > threshold) {
			accuracy++
		}
	}
	return accuracy / float64(len(scores))
}
