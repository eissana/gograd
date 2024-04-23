package main

import (
	"fmt"

	nn "github.com/eissana/gograd/neural-network"
	"gonum.org/v1/plot/vg"
)

const (
	epochs = 100
)

func main() {
	layerParams := []nn.LayerParam{
		// First hidden layer with 10 neurons.
		nn.MakeLayerParam(10, nn.Tanh),
		// Second hiden layer with 10 neurons.
		nn.MakeLayerParam(10, nn.Tanh),
		// Output layer with one neuron.
		nn.MakeLayerParam(1, nn.Sigmoid),
	}
	// Creates a neural network with input size of 2.
	model := nn.MakeNeuralNetwork( /*inputSize=*/ 2, layerParams)

	trainingParam := nn.TrainingParam{
		Epochs:                  epochs,
		Regularization:          0.0, // no regularization
		ClassificationThreshold: 0.5,
		LearningRate:            0.9,
	}
	// Reduce batchSize to speed up the process.
	batchSize := epochs

	// Reading make-moon dataset from a csv file.
	lines := nn.ReadCSV("data/make_moon.csv")
	inputs, labels := nn.GetRecords(lines, batchSize)

	plotter := nn.Plotter{
		Width:  6 * vg.Inch,
		Height: 4 * vg.Inch,
	}
	inputX, inputY := getXY(inputs)
	plotter.ScatterPlot(inputX, inputY, "data/make_moon.png")

	// Trains the model and returns losses and scores.
	losses, scores := model.Train(inputs, labels, trainingParam)

	// Computes the accuracy of the model.
	accuracy := nn.Accuracy(scores, labels, trainingParam)
	fmt.Printf("Loss: %3.4f, Accuracy: %3.0f%%\n", losses[len(losses)-1], 100*accuracy)

	// Plots the loss function.
	iterations := getX(len(losses))
	plotter.PlotLine(iterations, losses, "results/loss.png")
}

func getXY(inputs [][]*nn.Value) ([]float64, []float64) {
	n := len(inputs)
	x := make([]float64, n)
	y := make([]float64, n)
	for i, input := range inputs {
		x[i], y[i] = input[0].GetData(), input[1].GetData()
	}
	return x, y
}

func getX(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i)
	}
	return x
}
