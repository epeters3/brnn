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

end # module plugins