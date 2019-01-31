module brnn
using dataset: dataItem, dataSet

# Data Structures
mutable struct forwardLayer
    outputs::Array{Float64}
    activations::Array{Float64}
    weights::Array{Array{Float64}}
    inputSize::Int
    outputSize::Int
end

mutable struct recurrentLayer
    outputs::Array{Array{Float64}} #outputs needed for each timestep
    activations::Array{Array{Float64}} #activations needed for each timestep
    weights::Array{Array{Float64}} #weights only needed for one timestep
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
    outputs = [Array{Float64}(undef, o)]
    activations = [Array{Float64}(undef, o)]
    weights = Array{Array{Float64}}(undef, o)
    for index in 1:o;
        weights[o] = Array{Float64}(undef, i)
    end
    return recurrentLayer(outputs, activations, weights, i, o)
end

# Initalize a regular forward layer with i inputs, and o output nodes
function forwardLayer(i::Int, o::Int)
    outputs = Array{Float64}(undef, o)
    activations = Array{Float64}(undef, o)
    weights = Array{Array{Float64}}(undef, o)
    for index in 1:o;
        weights[o] = Array{Float64}(undef, i)
    end
    return forwardLayer(outputs, activations, weights, i, o)
end

# Forward Propagation From Inputs to Outputs
function propagateForward(network::brnnNetwork, inputs::dataItem)

end

# Data items are the input to the recurrent layer
function propagateForward(layer::recurrentLayer, inputs::dataItem)

end

# The outputs of forward and backward recurrent layers are the inputs to the last layer
function propagateForward(layer::forwardLayer, forwardInputs::recurrentLayer, backwardInputs::recurrentLayer)

end

# Resilient Propagation Learning
function rprop(network::brnnNetwork, outputs::dataItem)

end

# Data items are the input to the recurrent layer
function rprop(layer::recurrentLayer, inputs::dataItem)

end

# The outputs of forward and backward recurrent layers are the inputs to the last layer
function rprop(layer::forwardLayer, forwardInputs::recurrentLayer, backwardInputs::recurrentLayer)

end

# Train the network from a dataset
function learn(network::brnnNetwork, data::dataSet)
    for item in data.examples;
        print(item)
    end
end

# Test the network from a dataset
function test(network::brnnNetwork, data::dataSet)

end


printArgs() = println("The command line args are $(ARGS)")

end