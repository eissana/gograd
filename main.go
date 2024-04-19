package main

import (
	"github.com/eissana/gograd/neural-network"
)

func main() {
	input := []*nn.Value{nn.MakeValue(1.3), nn.MakeValue(4.1)}
	layers := []nn.LayerParam{
		nn.MakeLayerParam(2, nn.Tanh),
		nn.MakeLayerParam(1, nn.Sigmoid),
	}
	output := nn.MakeNeuralNetwork(input, layers)
	nn.BackPropagate(output[0])

	nn.DrawGraph(output[0], "graph.png")
}
