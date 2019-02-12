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

end # module plugins