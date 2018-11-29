package main

import (
	"fmt"
	"io/ioutil"
	"path/filepath"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	modeldir := "meta/model"
	modelpath := filepath.Join(modeldir, "frozen_inference_graph.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		panic(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		panic(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		panic(err)
	}
	defer session.Close()

	ops := graph.Operations()
	for _, op := range ops {
		fmt.Printf("%s\n", op.Name())
	}
}
