module Train

export train

include("InitParams.jl")
include("ForwardProp.jl")
include("CostFunc.jl")
include("BackwardProp.jl")
include("GradDescent.jl")

"""

""" function train(nlayers_nsize::T,input,output;
          learnrate=0.1e0,n_iter=100) where {T <: Union{Array,Tuple}}

    parameters = InitParams.layerparams(nlayers_nsize);
    iterations = range(0,step=1,stop=n_iter);
    costloss = zeros(Float64,n_iter);
    accuracy = zeros(Float64,n_iter);


    for i=iterations
        activation, cache = ForwardProp.forwardprop(input,parameters);
        costfunc = CostFunc.costfunction(pulse,target);
        accurate = 0;
        gradients = BackwardProp.backprop(target,activation,cache);
        parameters = GradDescent.graddescent(parameters,gradients;learnrate=learnrate);
        println("$(i)     $(costfunc)     $(accurate)")
        costloss[i],accuracy[i] = costfunc,accurate;
    end

    return parameters,costloss
        
end


         
end
