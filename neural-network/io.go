package nn

import (
	"encoding/csv"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// Reads data from a CSV file and returns each line as a slice of string.
func ReadCSV(filename string) [][]string {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("failed to open file: %v", err)
	}
	defer file.Close()
	lines, err := csv.NewReader(file).ReadAll()
	if err != nil {
		log.Fatalf("failed to read from the CSV file: %v", err)
	}
	return lines
}

// Returns the inputs and labels of the lines. If batchSize is smaller than
// the number of records, we randomly sample from it.
// The input and label sizes are equal to the batchSize.
func GetRecords(lines [][]string, batchSize int) ([][]*Value, [][]*Value) {
	numRecords := len(lines)
	batchIndices := getBatchIndices(batchSize, numRecords, time.Now().Unix())

	inputs := make([][]*Value, 0, batchSize)
	labels := make([][]*Value, 0, batchSize)

	for _, i := range batchIndices {
		input, label := getRecord(lines[i])
		inputs = append(inputs, input)
		labels = append(labels, []*Value{label})
	}
	return inputs, labels
}

// Plotter object containing parameters for plotting a graph.
type Plotter struct {
	Width, Height vg.Length
}

// Draws the loss function and saves it to a file.
func (p Plotter) PlotLine(x, y []float64, filename string) {
	xys := make(plotter.XYs, len(x))
	for i := range xys {
		if math.IsNaN(y[i]) {
			continue
		}
		xys[i].X = x[i]
		xys[i].Y = y[i]
	}

	plt := plot.New()
	s, _ := plotter.NewLine(xys)
	plt.Add(s)
	plt.Save(p.Width, p.Height, filename)
}

// Draws the scatter plot of given points.
func (p Plotter) ScatterPlot(x, y []float64, filename string) {
	xys := make(plotter.XYs, len(x))
	for i := range xys {
		xys[i].X = x[i]
		xys[i].Y = y[i]
	}

	plt := plot.New()
	s, _ := plotter.NewScatter(xys)
	plt.Add(s)
	plt.Save(p.Width, p.Height, filename)
}

func getRecord(line []string) ([]*Value, *Value) {
	n := len(line)
	label, _ := strconv.ParseFloat(line[n-1], 64)

	input := make([]*Value, n-1)
	for j := range input {
		value, _ := strconv.ParseFloat(line[j], 64)
		input[j] = MakeValue(value)
	}
	return input, MakeValue(label)
}

func getBatchIndices(batchSize, size int, seed int64) []int {
	r := rand.New(rand.NewSource(seed))
	return r.Perm(size)[:batchSize]
}
