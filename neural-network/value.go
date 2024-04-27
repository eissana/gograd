package nn

import (
	"fmt"
	"log"
	"math"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
)

// Value object.
// Leaf nodes represent input data with op="", len(children)=0 backward=nil.
// Other nodes represents data resulted from an operation (op) on children.
// backward() updates gradient of children.
type Value struct {
	data, grad float64
	op         string
	children   []*Value
	backward   func()
}

// Makes a new value from a float number.
func MakeValue(data float64) *Value {
	return &Value{
		data:     data,
		children: []*Value{},
	}
}

// Returns the data in this value object.
func (value Value) GetData() float64 {
	return value.data
}

func (value *Value) SetData(data float64) {
	value.data = data
}

// Returns the operation that is applied on the children resulted in
// this value.
func (value Value) GetOp() string {
	return value.op
}

// Returns the gradient of a given value.
func (value Value) GetGrad() float64 {
	return value.grad
}

func (value *Value) ResetGrad() {
	value.grad = 0.0
}

// Addition: a+b
func (value *Value) Add(other *Value) *Value {
	op := "+"
	if value == other {
		op = "*2"
	}
	ans := &Value{
		data:     value.data + other.data,
		op:       op,
		children: []*Value{value, other},
	}
	ans.backward = func() {
		value.grad += ans.grad
		other.grad += ans.grad
	}
	return ans
}

// Multiplication: a*b
func (value *Value) Mul(other *Value) *Value {
	op := "*"
	if value == other {
		op = "^2"
	}
	ans := &Value{
		data:     value.data * other.data,
		op:       op,
		children: []*Value{value, other},
	}
	ans.backward = func() {
		value.grad += other.data * ans.grad
		other.grad += value.data * ans.grad
	}
	return ans
}

func (value *Value) Pow(b float64) *Value {
	op := fmt.Sprintf("^%.2f", b)
	ans := &Value{
		data:     math.Pow(value.data, b),
		op:       op,
		children: []*Value{value},
	}
	ans.backward = func() {
		value.grad += (b * math.Pow(value.data, b-1.0)) * ans.grad
	}
	return ans
}

// Subtraction: a-b
func (value *Value) Sub(other *Value) *Value {
	op := "-"
	if value == other {
		op = "*0"
	}
	ans := &Value{
		data:     value.data - other.data,
		op:       op,
		children: []*Value{value, other},
	}
	ans.backward = func() {
		value.grad += ans.grad
		other.grad -= ans.grad
	}
	return ans
}

// Division: a/b
func (value *Value) Div(other *Value) *Value {
	return value.Mul(other.Pow(-1.0))
}

func (value *Value) Log() *Value {
	ans := &Value{
		data:     math.Log(value.data),
		op:       "Log",
		children: []*Value{value},
	}
	ans.backward = func() {
		value.grad += (1.0 / value.data) * ans.grad
	}
	return ans
}

func (value *Value) Exp() *Value {
	data := math.Exp(value.data)
	ans := &Value{
		data:     data,
		op:       "Exp",
		children: []*Value{value},
	}
	ans.backward = func() {
		value.grad += data * ans.grad
	}
	return ans
}

// Implements backward propagation the topologically sorted list of nodes.
// It's applied on the loss function value which needs to be minimized.
func (value *Value) BackPropagate() {
	sorted := []*Value{}
	topoSort(value, map[*Value]bool{}, &sorted)

	value.grad = 1.0
	for i := len(sorted) - 1; i >= 0; i-- {
		if sorted[i].backward != nil {
			sorted[i].backward()
		}
	}
}

func topoSort(value *Value, visited map[*Value]bool, ans *[]*Value) {
	if value == nil || visited[value] {
		return
	}
	visited[value] = true
	for _, next := range value.children {
		topoSort(next, visited, ans)
	}
	*ans = append(*ans, value)
}

// Builds the graph from the root value of the computation graph and assigns
// unique IDs to nodes. Leaf value nodes are represented by a single node in
// the graph. Other nodes are represented by two nodes as op->Value.
// Example: Consider the following computation graph:
//
//	  (4,+)
//	   / \
//	(1,) (3,)
//
// Its generated graph is as follows:
//
//	  4
//	  |
//	  +
//	 / \
//	1   3
func clone(value *Value, graph *cgraph.Graph, nodeId *int, visited map[*Value]*cgraph.Node) *cgraph.Node {
	if value == nil {
		return nil
	}
	if ans, ok := visited[value]; ok {
		return ans
	}
	(*nodeId)++
	label := fmt.Sprintf("ID: %d | data: %3.2f | grad: %3.2f", *nodeId, value.data, value.grad)
	valueNode, err := graph.CreateNode(label)
	if err != nil {
		log.Fatalf("failed to create data node: %v", err)
	}
	visited[value] = valueNode
	if value.op != "" {
		// create a node for the operation.
		label = fmt.Sprintf("ID: %d | %s", *nodeId, value.op)
		opNode, err := graph.CreateNode(label)
		if err != nil {
			log.Fatalf("failed to create op data node: %v", err)
		}
		// connect the operation node to the data node.
		_, err = graph.CreateEdge(value.op, opNode, valueNode)
		if err != nil {
			log.Fatalf("failed to create op->node edge: %v", err)
		}
		for _, child := range value.children {
			next := clone(child, graph, nodeId, visited)
			_, err := graph.CreateEdge("", next, opNode)
			if err != nil {
				log.Fatalf("failed to create child->parent edge: %v", err)
			}
		}
	}
	return valueNode
}

// Builds and draws a graph and writes it to a file.
func DrawGraph(values []*Value, filename string) {
	g := graphviz.New()
	graph, err := g.Graph()
	if err != nil {
		log.Fatalf("failed to initialize graph object: %v", err)
	}
	defer func() {
		if err := graph.Close(); err != nil {
			log.Fatalf("failed to close graph object: %v", err)
		}
		g.Close()
	}()

	nodeId := 1
	for _, value := range values {
		clone(value, graph, &nodeId, map[*Value]*cgraph.Node{})
	}

	if err := g.RenderFilename(graph, graphviz.PNG, filename); err != nil {
		log.Fatalf("failed to write graph to file: %v", err)
	}
}
