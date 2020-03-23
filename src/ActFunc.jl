
#Forward Propagation
""" sigmoid(z::Number)
Returns the sigmoid function value of z.
""" function sigmoid(Z::T) where T<:Real
         σ = 1.00e0 / ( 1.00e0 + exp(-Z) );
         Z_cache = copy(Z);
    return σ,Z_cache
end

function sigmoid(Z⃗::Array{T}) where T<: Real
    σ = 1.00e0 ./ ( 1.00e0 .+ exp.(-Z⃗) );
    Z⃗_cache = copy(Z⃗)
    return σ,Z⃗_cache
end


#sigmoid(Z⃗::Array{T}) where T<:Real = map(sigmoid,Z⃗)
#sigmoid(Z,cache::T) where T<:Real = sigmoid(Z),cache


""" relu(z::Number)
returns the ReLu function value of z.
""" function relu(Z::T) where T<:Real
         a = max(0.00e0,Z);
         Z_cache = copy(Z);
    return a,Z_cache
end

function relu(Z⃗::Array{T}) where T<: Real
    a = max.(0.00e0,Z⃗);
    Z⃗_cache = copy(Z⃗);
    return a,Z⃗_cache
end

#relu(Z⃗::Array{T}) where T<:Real = map(relu,Z⃗)
#relu(Z,cache::T) where T<:Real = relu(Z),cache

#Backward Propagation

""" backrelu(dinput::Array{Real},cache::Tuple)
This provides the derivative of the activation function with regards to the
activation pulse, i.e., \\frac{\\partial g(z)}{\\partial z}

This function handles elementwise operations.
""" function backrelu(dinput::Array{T},cache::Tuple) where T<:Real
    dgdZ = input .* (cache .> 0)
    return dgdZ
end


"""backsigmoid(dinput::Array{Real},cache::Tuple)
This provides the derivative of the activation function with regards to the
activation pulse, i.e., \\frac{\\partial g(z)}{\\partial z}

This function handles elementwise operations.
""" function backsigmoid(dinput::Array{T},cache::Tuple) where T<:Real
    dgdZ = dinput .* (sigmoid(cache) .* (1.00e0 .- sigmoid(cache)));
    return dgdZ
end





   
