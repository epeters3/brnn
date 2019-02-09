module plugins
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

crossEntropy(t::Array{Float64}, z::Array{Float64}) = -sum(t .* log.(z))

end # module plugins