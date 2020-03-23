module GradDescent

export graddescent

"""graddescent(parameters::Dict,gradients::Dict;learnrate=1.0e0)

        Solve wᵢ = wᵢ- α ∂J/∂wᵢ

where J is the cost function and w are the parmeters.

""" function graddescent(parameters::Dict,
                         gradients::Dict;learnrate=0.1e0) where T<: Real

    nlayers = Int(length(layerparams)/2);
    for l=0:nlayers-1
        parameters["weights_$(l-1)"] -= learnrate .* gradients["dweights_$(l)"]
        parameters["bias_$(l-1)"] -= learnrate .* gradients["dbias_$(l)"]
    end

    return parameters
end


end
