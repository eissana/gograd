# gograd

A simple go library to build a neural network model.

To run the main function, run:

```sh
go run main.go
```

To run all tests, run:

```sh
go test -count=1 -v ./...
```

Note that `-count=1` disables caching. To run a single test, run:

```sh
go test -count=1 -v -run TestValue1 ./...
```

For the `-run` option, one can use a regex expression too.
