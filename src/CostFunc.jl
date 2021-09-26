module CostFunc

export costfunction

function meansamples(A::Array{S}) where S<: Number
    ndata,_ = size(A);
    μ = sum(A,dims=1) ./ ndata
    return μ
end

"""
Function to generate and evaluate the cost function expression

""" function costfunction(activation,target,lossfunc=:crossentropy)
        cost = Expr(:call,lossfunc,activation,target)
        return eval(cost)
end

""" crossentropy(activation,target)

""" function crossentropy(activation,target)
    entropy = map(calcentropy,target,activation);
    cost = -1.00e0 * sum(meansamples(entropy));
    return cost
end

calcentropy(y::Float64,ŷ::Float64) = y*log(ŷ) + (1.00e0-y)*log(1.00e0-ŷ);

"""
""" function meansquarederror(activation::AbstractArray,target::AbstractArray)
    squarederror = map(calcsquarederror,target,activation);
    cost = sum(meansamples(squarederror))
    return cost
end

calcsquareerror(y::Float64,ŷ::Float64) = (y-ŷ)^2
end
