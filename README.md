# TinyNN.jl

This is a simple Julia code that is setup to optimize a neural network that is fully connected (i.e., dense). The inner layers are all activated using the [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks) function and the output layer is activated using the [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function. Therefore this type of network is good for classification type problems that are simple, for example, the MNIST database (see example).

The code is very similar to that posted by [Xabier G. Andrade](https://medium.com/datadriveninvestor/how-to-build-a-deep-neural-network-from-scratch-with-julia-862116a194c) given thats where I got the idea to learn about Neural Networks by doing it myself.

## Backwards Propagation

In order to optimize the weights/parameters of the newtork gradient descent needs to be performed. For gradient descent we need the derivative of the cost/loss function with regards to the weights/parameters. This requires invoking the chain rule from derivative calculus and providing routines to do so. At first my approach was going to be to use the automatic differentation packages [ForwardDiff.jl]() or [ReverseDiff.jl](), but I couldn't quite figure out how to do this. I'm sure its a simple misunderstanding on my part so I will get around to doing this eventually. For now the backward propagation has been hardcoded in the module [BackProp.jl](src/BackProp.jl). 