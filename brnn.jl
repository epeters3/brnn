module brnn
using Random
using dataset: dataItem, dataSet
# Data Structures

struct learningParams
    activation::Function
    fPrimeNet::Function
    learningRate::Float64
    τ::Int64
end

struct learningStatistics
    averageWeightChange::Float64
    weightChangePerExample::Float64
end

mutable struct forwardLayer
    activations::Array{Float64}
    weights::Array{Float64,2}
    inputSize::Int
    outputSize::Int
    params::learningParams
    stats::learningStatistics
end

mutable struct recurrentLayer
    activations::Array{Array{Float64}} #activations needed for each timestep, this is not a 2d-array since we don't know ahead of time one of the dimensions
    weights::Array{Float64,2} #weights only needed for one timestep
    inputSize::Int
    outputSize::Int
    forward::Bool
    params::learningParams
    stats::learningStatistics
end

mutable struct brnnNetwork
    recurrentForwardsLayer::recurrentLayer
    recurrentBackwardsLayer::recurrentLayer
    outputLayer::forwardLayer
    inputSize::Int
    hiddenSize::Int
    outputSize::Int
    params::learningParams
end



function sigmoidActivation(inputs::Array{Float64,1})
    return 1 ./ (1 .+ ℯ.^(-inputs))
end

function sigmoidFprimeNet(activations::Array{Float64,1})
    return activations .* (1 .- activations)
end

function learningParams(learningRate::Float64, τ::Int)
    return learningParams(sigmoidActivation, sigmoidFprimeNet, learningRate, τ)
end

function learningStatistics()
    return learningStatistics(0, 0)
end

# Initalize a brnn with i inputs, n hidden nodes, and o output nodes
function brnnNetwork(i::Int, n::Int, o::Int, hiddenLearningParams::learningParams, outputLearningParams::learningParams, networkLearningParams::learningParams)
    forwardRecurrentLayer = recurrentLayer(i, n, hiddenLearningParams, true)
    backwardRecurrentLayer = recurrentLayer(i, n, hiddenLearningParams, false)
    outputLayer = forwardLayer(n * 2, o, outputLearningParams)
    return brnnNetwork(forwardRecurrentLayer, backwardRecurrentLayer, outputLayer, i, n, o, networkLearningParams)
end

# Initalize a recurrentLayer with i inputs, and o output nodes
function recurrentLayer(i::Int, o::Int, params::learningParams, forward::Bool)
  # One timestep for now, we will add timesteps as we need to keep track of activations, and not before
    activations = [zeros(o)];
    weights = (rand(o, i + o + 1) .* 2) .- 1 #There is a weight to every input output and 
    return recurrentLayer(activations, weights, i, o, forward, params, learningStatistics())
end

# Initalize a regular forward layer with i inputs, and o output nodes
function forwardLayer(i::Int, o::Int, params::learningParams)
    activations = Array{Float64}(undef, o)
    weights = (rand(o, i + 1) .* 2) .- 1
    return forwardLayer(activations, weights, i, o, params, learningStatistics())
end

function clearActivations(layer::recurrentLayer)
    layer.activations = [zeros(layer.outputSize)];
end

# Forward Propagation From Inputs to Outputs
function propagateForward(network::brnnNetwork, inputs::Array{dataItem})
    # TODO: Call propagate forward for the appropriate number of time steps and the number of inputs
    clearActivations(network.recurrentForwardsLayer)
    clearActivations(network.recurrentBackwardsLayer)
    propagateForward(network.recurrentForwardsLayer, inputs)
    propagateForward(network.recurrentBackwardsLayer, inputs)
    propagateForward(network.outputLayer, network.recurrentForwardsLayer, network.recurrentBackwardsLayer)
end

# Data items are the input to the recurrent layer
function propagateForward(layer::recurrentLayer, inputs::Array{dataItem})
    # The input to the recurrent layer is the input concatenated with the recurrent layer's previous activation and a bias
    if layer.forward 
        for i in inputs
            net = layer.weights * vcat(i.features..., layer.activations[end]..., 1)
            push!(layer.activations, layer.params.activation(net))
        end
    else
        for i in Iterators.reverse(inputs)
            net = layer.weights * vcat(i.features..., layer.activations[end]..., 1)
            push!(layer.activations, layer.params.activation(net))
        end
    end
end

# The outputs of forward and backward recurrent layers are the inputs to the last layer
function propagateForward(layer::forwardLayer, forwardInputs::recurrentLayer, backwardInputs::recurrentLayer)
    net = layer.weights * vcat(forwardInputs.activations[end], backwardInputs.activations[end], 1)
    layer.activations = layer.params.activation(net)
end

function bptt(network::brnnNetwork, inputs::Array{dataItem,1})
    bptt(network.outputLayer, network.recurrentForwardsLayer, network.recurrentBackwardsLayer, inputs);
end

function bptt(layer::recurrentLayer, error::Array{Float64,1}, inputs::Array{dataItem,1})
    δweights = Array{Array{Float64,2},1}(undef, 0)
    layerError = error
    if layer.forward
        for i in length(layer.activations) - 1:-1:2
            #print("Error $(size(layerError)), ")
            #println("Weights $(size(transpose(layer.weights)))")
            layerError = sum(transpose(layer.weights) * layerError) .* layer.params.fPrimeNet(layer.activations[i])
            push!(δweights, transpose(layerError .* vcat(layer.activations[i - 1]..., inputs[i].features..., 1) .* layer.params.learningRate)) 
        end
    else
        for i in length(layer.activations) - 1:-1:2
            layerError = sum(transpose(layer.weights) * layerError) .* layer.params.fPrimeNet(layer.activations[i])
            push!(δweights,  transpose(layerError .* vcat(layer.activations[i - 1]..., inputs[length(inputs) - i + 1].features..., 1) .* layer.params.learningRate))
           
        end
    end
    #print("dweights $(size(δweights[1])), ")
    #println("weights $(size(layer.weights))")
    totalδweights = sum(δweights) ./ length(δweights)
    layer.weights .+= totalδweights;
end

function bptt(layer::forwardLayer, forwardInputs::recurrentLayer, backwardInputs::recurrentLayer, inputs::Array{dataItem})
    error = (layer.activations .- inputs[end].labels) .* layer.params.fPrimeNet(layer.activations)
    δweights = error .* vcat(forwardInputs.activations[end]..., backwardInputs.activations[end]..., 1) .* layer.params.learningRate
    bptt(forwardInputs, error, inputs);
    bptt(backwardInputs, error, inputs);
    layer.weights .+= transpose(δweights);
end

# Train the network from a dataset
function learn(network::brnnNetwork, data::dataSet)
    window = Array{dataItem}(undef, 0)
    timesThrough = 0
    for item in data.examples
        push!(window, item) # Appends to the end
        if length(window) == network.params.τ
            propagateForward(network, window);
            bptt(network, window);
            popfirst!(window) # Pops from the first
            timesThrough += 1
        end
    end
    println("Learned from $(timesThrough) examples");
end 

# Test the network from a dataset
function test(network::brnnNetwork, data::dataSet)

end


printArgs() = println("The command line args are $(ARGS)")

end