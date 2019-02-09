module brnn
using Random
using dataset: dataItem, dataSet
# Data Structures
mutable struct forwardLayer
    activations::Array{Float64}
    weights::Array{Float64,2}
    inputSize::Int
    outputSize::Int
end

mutable struct recurrentLayer
    activations::Array{Array{Float64}} #activations needed for each timestep, this is not a 2d-array since we don't know ahead of time one of the dimensions
    weights::Array{Float64,2} #weights only needed for one timestep
    inputSize::Int
    outputSize::Int
    forward::Bool
end

mutable struct brnnNetwork
    recurrentForwardsLayer::recurrentLayer
    recurrentBackwardsLayer::recurrentLayer
    outputLayer::forwardLayer
    inputSize::Int
    hiddenSize::Int
    outputSize::Int
end

struct learningParams
    activation::Function
    fprimeNet::Function
    learningRate::Float64
    τ::Int64
end

function sigmoidActivation(inputs::Array{Float64})
    return 1 ./ (1 .+ .^(-inputs))
end

function sigmoidFprimeNet(activations::Array{Float64})
    return activations .* (1 .- activations)
end

function learningParams(learningRate::Float)
    return learningParams(sigmoidActivation, sigmoidFprimeNet, learningRate)
end

# Initalize a brnn with i inputs, n hidden nodes, and o output nodes
function brnnNetwork(i::Int, n::Int, o::Int)
    forwardRecurrentLayer = recurrentLayer(i, n, true)
    backwardRecurrentLayer = recurrentLayer(i, n, false)
    outputLayer = forwardLayer(n * 2, o)
    return brnnNetwork(forwardRecurrentLayer, backwardRecurrentLayer, outputLayer, i, n, o)
end

# Initalize a recurrentLayer with i inputs, and o output nodes
function recurrentLayer(i::Int, o::Int, forward::Bool)
  # One timestep for now, we will add timesteps as we need to keep track of activations, and not before
    activations = [Array{Float64}(0, o)];
    weights = (rand(o, i + o + 1) .* 2) .- 1 #TODO: See how to get this to clip to [-1:1]
    return recurrentLayer(activations, weights, i, o, forward)
end

# Initalize a regular forward layer with i inputs, and o output nodes
function forwardLayer(i::Int, o::Int)
    activations = Array{Float64}(undef, o)
    weights = (rand(o, i + 1) .* 2) .- 1
    return forwardLayer(activations, weights, i, o)
end

function clearActivations(layer::recurrentLayer)
    layer.activations = [Array{Float64}(0, layer.outputSize)];
end

# Forward Propagation From Inputs to Outputs
function propagateForward(network::brnnNetwork, inputs::Array{dataItem}, params::learningParams)
    # TODO: Call propagate forward for the appropriate number of time steps and the number of inputs
    clearActivations(network.recurrentForwardsLayer)
    clearActivations(network.recurrentBackwardsLayer)
    propagateForward(network.recurrentForwardsLayer, inputs, params)
    propagateForward(network.recurrentBackwardsLayer, inputs, params)
    propagateForward(network.outputLayer, network.recurrentForwardsLayer, network.recurrentBackwardsLayer, params)
end

# Data items are the input to the recurrent layer
function propagateForward(layer::recurrentLayer, inputs::dataItem, params::learningParams)
    # The input to the recurrent layer is the input concatenated with the recurrent layer's previous activation and a bias
    if layer.forward 
        for i in inputs
            push!(layer.activations, params.activation(layer.weights * hcat(i.features, layer.activations[end], 1)))
        end
    else
        for i in reverse(inputs)
            push!(layer.activations, params.activation(layer.weights * hcat(i.features, layer.activations[end], 1)))
        end
    end
end

# The outputs of forward and backward recurrent layers are the inputs to the last layer
function propagateForward(layer::forwardLayer, forwardInputs::recurrentLayer, backwardInputs::recurrentLayer, params::learningParams)
    layer.activations = params.activation(layer.weights * hcat(forwardInputs.activations[end], backwardInputs.activations[end], 1))
end

function bptt(network::brnnNetwork, inputs::Array{dataItem}, params::learningParams)
    bptt(network.outputLayer, network.recurrentForwardsLayer, network.recurrentBackwardsLayer, inputs, params);
end

function bptt(layer::recurrentLayer, error::Array{Float64}, inputs::Array{dataItem}, params::learningParams)
    δweights = []
    layerError = error
    if layer.forward
        for i in layer.activations.length - 1:-1:1
            layerError = transpose(layer.weights) * layerError
            push!(δweights, layerError .* hcat(layer.activations[i], inputs[i], 1)) 
        end
    else
        for i in layer.activations.length - 1:-1:1
            layerError = transpose(layer.weights) * layerError
            push!(δweights, layerError .* hcat(layer.activations[i], inputs[inputs.length - i], 1)) 
        end
    end
    totalδweights = sum(δweights) ./ δweights.length
    layer.weights += totalδweights;
end

function bptt(layer::forwardLayer, forwardInputs::recurrentLayer, backwardInputs::recurrentLayer, inputs::Array{dataItem}, params::learningParams)
    error = (layer.activations .- inputs[end].labels) .* params.fprimenet(layer.activations)
    δweights = error .* hcat(forwardInputs.activations[end], backwardInputs.activations[end], 1) .* params.learningRate
    bptt(forwardInputs, error, params);
    bptt(backwardInputs, error, params);
    layer.weights .+= δweights;
end

# Train the network from a dataset
function learn(network::brnnNetwork, data::dataSet)
    for item in data.examples;
        println(item)
    end
end

# Test the network from a dataset
function test(network::brnnNetwork, data::dataSet)

end


printArgs() = println("The command line args are $(ARGS)")

end