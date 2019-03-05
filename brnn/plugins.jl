module plugins

############
#### Helpers
############

assertEqualLength(a::Array, b::Array) = length(a) != length(b) && error("arguments must be of equal length $(length(a)) != $(length(b))")

#########################
#### Activation Functions
#########################

sigmoid(nets::Array{Float64}) = 1 ./ (1 .+ exp.(-nets))

sigmoidPrime(outputs::Array{Float64}) = outputs .* (1 .- outputs)

function softmax(nets::Array{Float64})
    exps::Array{Float64} = exp.(nets)
    return exps ./ sum(exps)
end

ReLU(nets::Array{Float64}) = max.(0, nets)

ReLUPrime(output::Float64) = output < 0 ? 0 : 1

ReLUPrime(outputs::Array{Float64}) = ReLUPrime.(outputs)

linear(nets::Array{Float64}) = nets

linearPrime(outputs::Array{Float64}) = ones(size(outputs))

tanH(nets::Array{Float64}) = tanh.(nets)

tanHPrime(outputs::Array{Float64}) = 1 .- tanh.(outputs).^2

###################
#### Loss Functions
###################

function crossEntropy(t::Array{Float64}, z::Array{Float64})
    assertEqualLength(t, z)
    return -sum(t .* log.(z))
end

function SSE(t::Array{Float64}, z::Array{Float64})
    assertEqualLength(t, z)
    return sum((t - z).^2)
end

function MSE(t::Array{Float64}, z::Array{Float64})
    assertEqualLength(t, z)
    return sum((t - z).^2) / length(t)
end

##########
#### Other
##########

#=
Makes an array of arbitrary dimensions filled with normally-distributed values
following a distribution with mean `mean` and standard deviation `stddev`.
=#
function randGaussian(dims::Tuple{Vararg{Int64}}, mean::Float64, stddev::Float64)::Array{Float64}
    return (randn(dims) .* stddev) .- (stddev / 2) .+ mean
end



#############
#### Momentum
#############

function RPropMomentum(n_minus::Float64, n_plus::Float64)
    rprop = function (currentδWeights::Array{Float64,2}, prevδWeights::Array{Float64,2})
        c_lt = currentδWeights .< 0
        c_gt = currentδWeights .>= 0
        p_lt = prevδWeights .< 0
        p_gt = prevδWeights .>= 0
        didChangeSign = (c_lt .& p_gt) .| (c_gt .& p_lt)
        didNotChangeSign = (c_lt .& p_lt) .| (c_gt .& p_gt)
        return currentδWeights .* (n_minus .* didChangeSign + n_plus .* didNotChangeSign);
    end
    return rprop
end

function SimpleMomentum(momentum::Float64)
    momentumF = function (currentδWeights::Array{Float64,2}, prevδWeights::Array{Float64,2})
        return currentδWeights .+ (prevδWeights * momentum)
    end
    return momentumF
end

end # module plugins


