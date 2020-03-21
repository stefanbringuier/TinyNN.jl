module CostFunc

export costfunction

"""
""" function costfunction(activation,target,lossfunc=:crossentropy)
        cost = Expr(:call,lossfunc,activation,target)
        return eval(cost)
end

""" crossentropy(activation,target)

""" function crossentropy(activation::Array{T,1},target::Array{T,1}) where T<:Real
    entropy = map(calcentropy,target,activation);
    cost = -1.00e0 * mean(entropy);
    return cost
end

calcentropy(y::Float64,ŷ::Float64) = y*log(ŷ) + (1.00e0-y)*log(1.00e0-ŷ);
"""
""" function meansquarederror(activation::Array{T,1},target::Array{T,1}) where T<:Real
    squarederror = map(calcsquarederror,target,activation);
    cost = mean(squarederror)
    return cost
end

calcsquareerror(y::Float64,ŷ::Float64) = (y-ŷ)^2
end
