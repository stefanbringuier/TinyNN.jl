module ForwardProp

export forwardprop

include("ActFunc.jl")


"""

""" function forwardprop(input,layerparams;
                           layermodel=:relu,
                           outmodel=:sigmoid)


    #layeractivate = getactivate(model=layermodel) :: Function;

    nlayers = Int(length(layerparams)/2);

    layerinput = copy(input);

    cache = Array{Tuple,1}(undef,nlayers);

    #Feed forward NN
    for l=1:nlayers-1
        layerinput_l = layerinput;
        weights = layerparams["weights_$l"];
        bias = layerparams["bias_$l"];
        pulse,pcache,backprop = getactivate(layerinput_l,weights,
                                           bias,model=layermodel);
        cache[l] = (backprop,pcache);
        layerinput = pulse;

    end

    #Output/Terminal layer
    weights = layerparams["weights_$(nlayers)"];
    bias = layerparams["bias_$(nlayers)"];
    #termactivate = getactivate(model=outmodel) ::Function;
    pulse,pcache,backprop = getactivate(layerinput,weights,
                                        bias,model=outmodel);
    cache[nlayers] = (backprop,pcache)

    return pulse,cache
end


"""

""" function getactivate(input::Array,weights::Array,
                         bias::Array;
                         model=:sigmoid)

    Z,backward_cache = forward(input,weights,bias);
    activation,cache = eval(Expr(:call,model,Z));
    return activation,cache,backward_cache
end

#"""
#Construct the composite function f(Z) = model( Z(X,w,b) );
#where Z(X,w,b) = wX + b
#returns an anonomyous function.
#""" function getactivate(;model=:sigmoid)
#     activationfunc = Expr(:call,∘,model,forward)
#     return eval(activationfunc)
# end


"""
Calculate the linear forward propagation eq. Z = w*X + b

we need to store the values for the equation for backpropagation.
""" function forward(input::Array,weights::Array,
                     bias::Array)
    #println(size(weights),size(input),size(bias))
    Z = weights * input .+ bias;
    cache = (input,weights,bias);
    return Z,cache
end



end
