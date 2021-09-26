using MLDatasets
include("../src/Train.jl")


train_X, train_Y = MNIST.traindata();
ninput,noutput = 784,10;
train_X = reshape(train_X,(60000,ninput));
train_X /= 255;

"""
Create a hot map
""" function numbermap(X::Array{Int,1},N::Int)
       maparray = zeros(Float64,length(X),N);
       for i=1:length(X)
           ii = X[i]+1;
           maparray[i,ii] = 1.00e0;
           end
       return maparray
end

train_Y = numbermap(train_Y,noutput);

neuralnetwork = [ ninput 64 64 noutput]
parameters, cost = Train.train(neuralnetwork, train_X', train_Y');
