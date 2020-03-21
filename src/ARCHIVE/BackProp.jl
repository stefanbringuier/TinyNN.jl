module BackProp

using ReverseDiff: GradientTape,gradient,gradient!,compile

include("ForwdProp.jl")

"""
This function uses automatic differentiation to return the gradient.

NOTICE: Only allows for the same activation function in all hidden layers.

""" function autodifflayer(layerinput::Array{T,1},weights::Array{T,2},
                           bias::Array{T,2},
                           layermodel=:relu) where T<: Real

    layeractivate = ForwdProp.getactivate(model=layermodel) :: Function;
    l,w,b = layerinput,weights,bias  #Just for buffer/pre-recording
    #local gradfunc = GradientTape(layeractivate,(l,w,b));
    #local compiled_gradfunc = compile(gradfunc);
    gradients = Dict();
    gradinputs = (layerinput,weights,bias);
    gradresults = (similar(layerinput),similar(weights),similar(bias));
    gradient(layeractivate,gradinputs)
    #gradients["dinputs"] = gradresults[1];
    #gradients["dweights"] = gradresults[2];
    #gradients["dbias"] = gradresults[3];
    #return gradients
end
    
end
