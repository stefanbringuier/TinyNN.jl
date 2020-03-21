module InitParams

using Random,Distributions

export layerparams

""" layerparams(nlayers_dlength::Array{Int};disttype=:Normal)
Create a dictionary that corresponds to weights and biases
in for N fully connected layers with length D.

The dictionary consist of keys and values:

"weight_i", "bias_i" where i the connection index

the "weight_i" key produces an Array{Float64,2} and 
"bias_i" an Array{Float64,1}

Template:

Fully connected 
l-1  w_l-1  l  
 X     |    X
       | 
 X     |    X 
       |
 X     |    X 

      +b_l-1 
""" function layerparams(nlayers_dlength::Array{Int,1};disttype=:Normal)
    weights_and_bias = Dict();
    for l=2:length(nlayers_dlength)
        numneurons_layer1,numneurons_layer2 = nlayers_dlength[l-1:l]
        weightkey = string("weights_",string(l-1))
        biaskey = string("bias_",string(l-1))
        distribution = eval(Expr(:call,disttype));
        weights_and_bias[weightkey] = rand(distribution,
                                           numneurons_layer2,
                                           numneurons_layer1);
        weights_and_bias[biaskey] = zeros(numneurons_layer2,1);
    end
    return weights_and_bias
end

layerparams(nlayers_dlength::Array{Int,2};disttype=:Normal) = begin
    @assert size(nlayers_dlength)[1] == 1 ;
    layerparams(vec(nlayers_dlength),disttype=disttype);
end


end
