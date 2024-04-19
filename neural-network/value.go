package nn

import (
	"fmt"
	"log"
	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
)

// Value object.
// Leaf nodes represent input data with op="", left=nil, right=nil, backward=nil.
// Other nodes represents data resulted from an operation (op) on left and right.
// backward() updates gradient of children.
type Value struct {
	data, grad float64
	op string
	left, right *Value
	backward func()
}

func MakeValue(data float64) *Value {
	return &Value{
		data: data,
	}
}

func (value Value) GetData() float64 {
	return value.data
}

func (value Value) GetOp() string {
	return value.op
}

func (value Value) GetGrad() float64 {
	return value.grad
}

// Addition: a+b
func Add(a, b *Value) *Value {
	ans := &Value{
		data: a.data + b.data,
		op: "+", 
		left: a, 
		right: b,
	}
	ans.backward = func() {
		a.grad += ans.grad
		b.grad += ans.grad
	}
	return ans 
}

// Multiplication: a*b
func Mul(a, b *Value) *Value {
	ans := &Value{
		data: a.data * b.data,
		op: "*", 
		left: a, 
		right: b,
	}
	ans.backward = func() {
		a.grad += b.data * ans.grad
		b.grad += a.data * ans.grad
	}
	return ans 
}

// Subtraction: a-b
func Sub(a, b *Value) *Value {
	ans := &Value{
		data: a.data - b.data,
		op: "-", 
		left: a, 
		right: b,
	}
	ans.backward = func() {
		a.grad += ans.grad
		b.grad -= ans.grad
	}
	return ans 
}

// Division: a/b
func Div(a, b *Value) *Value {
	ans := &Value{
		data: a.data / b.data,
		op: "/", 
		left: a, 
		right: b,
	}
	ans.backward = func() {
		div := ans.grad / b.data
		a.grad += div
		b.grad -= a.data * div / b.data
	}
	return ans 
}

// Implements backward propagation by calling backward of each Value node
// on the topologically sorted list of nodes.
func BackPropagate(value *Value) {
	preorder := []*Value{}
	topoSort(value, &preorder)

	value.grad = 1.0
	for _, val := range preorder {
		if val.backward != nil {
			val.backward()
		}
	}
}

// In a tree, preorder yields topological sort.
func topoSort(value *Value, ans *[]*Value) {
	if value == nil {
		return
	}
	*ans = append(*ans, value)
	if value.left != nil {
		topoSort(value.left, ans)
	}
	if value.right != nil {
		topoSort(value.right, ans)
	}
} 

// Builds the graph from the root value of the computation graph and assigns
// unique IDs to nodes. Leaf value nodes are represented by a single node in
// the graph. Other nodes are represented by two nodes as op->Value.
// Example: Consider the following computation graph:
//        (4,+)
//         / \
//      (1,) (3,)
// Its generated graph is as follows:
//          4
//          |
//          +
//         / \
//        1   3
func buildGraph(value *Value, graph *cgraph.Graph, nodeId *int) *cgraph.Node{
	if value == nil {
		return nil
	}
	(*nodeId)++
	label := fmt.Sprintf("ID: %d | data: %3.2f | grad: %3.2f", *nodeId, value.data, value.grad)
	node, err := graph.CreateNode(label)
	if err != nil {
	  log.Fatalf("failed to create data node: %v", err)
	}
	if value.op == "" {
		// leaf node.
		return node
	}
	// both left and right nodes must exist.
	// create a node for the operation.
	(*nodeId)++
	label = fmt.Sprintf("ID: %d | %s", *nodeId, value.op)
	opNode, err := graph.CreateNode(label)
	if err != nil {
	  log.Fatalf("failed to create op data node: %v", err)
	}
	// connect the operation node to the data node.
	_, err = graph.CreateEdge(value.op, opNode, node)
	if err != nil {
	  log.Fatalf("failed to create op->node edge: %v", err)
	}
	if value.left != nil {
		// build left subtree.
		left := buildGraph(value.left, graph, nodeId)
		// connect left subtree to the operation node
		_, err = graph.CreateEdge(value.op, left, opNode)
		if err != nil {
		log.Fatalf("failed to create left->op edge: %v", err)
		}
	}
	if value.right != nil {
		// build right subtree.
		right := buildGraph(value.right, graph, nodeId)
		// connect right subtree to the operation node.
		_, err = graph.CreateEdge(value.op, right, opNode)
		if err != nil {
		log.Fatalf("failed to create right->op edge: %v", err)
		}
	}
	return node
}

// Builds and draws a graph and writes it to a file.
func DrawGraph(value *Value, filename string) {
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
	nodeId := 0
	buildGraph(value, graph, &nodeId)

	if err := g.RenderFilename(graph, graphviz.PNG, filename); err != nil {
		log.Fatalf("failed to write graph to file: %v", err)
	}
} 