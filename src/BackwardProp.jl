module BackwardProp

export backprop

include("ActFunc.jl")

"""

""" function getbackprop(dinput,cache;
                         model=:sigmoid)
    backmodel = Symbol(:back,model);
    dgdZ = eval(Expr(:call,backmodel,dinput,cache));
    return dgdZ
end

""" function backward(dZ::Array{Real},cache::Tuple)
Calculate the derivative
\\frac{\\partial J}{\\partial W} = \\frac{1}{m}\\frac{\\partial J}{\\partial Z} g(Z)^{T}

This is the back propagation step for the functio ForwardProp.forward()

""" function backward(dgdZ,cache)

    _input, _weights, _bias = cache;
    layersize = length(_input);
    dJdŶ = transpose(_weights) .* dgdZ;
    dweights = dgdZ .* (transpose(_input)) ./ layersize;
    dbias = sum(dgdZ)./layersize;

    return dinput, dweights, dbias
end


""" backprop(target::Array{Real}, backinput::Array{Real,1}, cache::Tuple;
            layermodel=:relu,outmodel=:sigmoid)

Main function call to perform backwards propagation for calculating derivatives.

""" function backprop(target,backinput,cache;
                      layermodel=:relu,
                      outmodel=:sigmoid)

    lencache = length(cache);
    layersize = length(backinput)
    gradients = Dict();

    Ŷ = copy(backinput);
    Y = reshape(target,size(backinput));

    dJdŶ = -1.00e0 * (Y./Ŷ) .+ (1.00e0 .- Y)./(1.00e0 .- Ŷ);

    cacheset = cache[lencache];
    dgdZ = getbackprop(dJdŶ,cacheset,model=outmodel);
    dJdŶ,dweights,dbias = backward(dgdZ,cacheset);
    gradients["dinput_$(lencache-1)"] = dJdŶ;
    gradients["dweights_$(lencache)"] = dweights;
    gradients["dbias_$(lencache)"] = dbias;

    for l=reverse(0:lencache-2)
        cacheset = cache[l+1];
        dgdZ = getbackprop(dJdŶ,cacheset,model=layermodel);
        dJdŶ,dweights,dbias = backward(dgdZ,cacheset);

        gradients["dinput_$(l)"] = dJdŶ;
        gradients["dweights_$(l+1)"] = dweights;
        gradients["dbias_$(l+1)"] = dbias;
    end

    return gradients


 end



end
