module brnn
using Random
using dataset: dataItem, dataSet
# Data Structures
mutable struct forwardLayer
    net::Array{Float64}
    activations::Array{Float64}
    weights::Array{Float64,2}
    inputSize::Int
    outputSize::Int
end

mutable struct recurrentLayer
    net::Array{Array{Float64}} #outputs needed for each timestep, this is not a 2d-array since we don't know ahead of time one of the dimensions
    activations::Array{Array{Float64}} #activations needed for each timestep, this is not a 2d-array since we don't know ahead of time one of the dimensions
    weights::Array{Float64,2} #weights only needed for one timestep
    inputSize::Int
    outputSize::Int
end

mutable struct brnnNetwork
    recurrentForwardsLayer::recurrentLayer
    recurrentBackwardsLayer::recurrentLayer
    outputLayer::forwardLayer
    inputSize::Int
    hiddenSize::Int
    outputSize::Int
end

# Initalize a brnn with i inputs, n hidden nodes, and o output nodes
function brnnNetwork(i::Int, n::Int, o::Int)
    forwardRecurrentLayer = recurrentLayer(i, n)
    backwardRecurrentLayer = recurrentLayer(i, n)
    outputLayer = forwardLayer(n * 2, o)
    return brnnNetwork(forwardRecurrentLayer, backwardRecurrentLayer, outputLayer, i, n, o)
end

# Initalize a recurrentLayer with i inputs, and o output nodes
function recurrentLayer(i::Int, o::Int)
  # One timestep for now, we will add timesteps as we need to keep track of activations, and not before
    net = [Array{Float64}(undef, o)]
    activations = [Array{Float64}(undef, o)]
    weights = (rand(o, i + o + 1) .* 2) .- 1 #TODO: See how to get this to clip to [-1:1]
    return recurrentLayer(net, activations, weights, i, o)
end

# Initalize a regular forward layer with i inputs, and o output nodes
function forwardLayer(i::Int, o::Int)
    net = Array{Float64}(undef, o)
    activations = Array{Float64}(undef, o)
    weights = (rand(o, i + 1) .* 2) .- 1
    println(weights)
    println()
    return forwardLayer(net, activations, weights, i, o)
end


# Forward Propagation From Inputs to Outputs
function propagateForward(network::brnnNetwork, inputs::Array{dataItem}, activation::Function)
    # TODO: Call propagate forward for the appropriate number of time steps and the number of inputs 
    propagateForward(network.recurrentForwardsLayer, inputs[0], activation)
    propagateForward(network.recurrentBackwardsLayer, inputs[0], activation)
    propagateForward(network.outputLayer, network.recurrentForwardsLayer, network.recurrentBackwardsLayer, activation)
end

# Data items are the input to the recurrent layer
function propagateForward(layer::recurrentLayer, inputs::dataItem, activation::Function)
    # The input to the recurrent layer is the input concatenated with the recurrent layer's previous activation and a bias
    push!(layer.net, layer.weights * hcat(inputs.features, layer.activations[end], 1))
    push!(layer.activations, activation(layer.net[end]))
end

# The outputs of forward and backward recurrent layers are the inputs to the last layer
function propagateForward(layer::forwardLayer, forwardInputs::recurrentLayer, backwardInputs::recurrentLayer, activation::Function)
    layer.net = hcat(forwardInputs.activations[end], backwardInputs.activations[end], 1)
    layer.activation = activation(layer.net)
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