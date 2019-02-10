module brnn
using Random
using dataset: dataItem, dataSet
using plugins: sigmoid, sigmoidPrime
# Data Structures

struct learningParams
    activation::Function
    fPrimeNet::Function
    learningRate::Float64
    τ::Int64
end

struct learningStatistics
    averageWeightChange::Array{Float64,1}
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

function learningParams(learningRate::Float64, τ::Int)
    return learningParams(sigmoid, sigmoidPrime, learningRate, τ)
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

function propagateForward(weights::Array{Float64,2}, inputs::Array{Float64,1}, activation::Function)
    return activation(weights * inputs)
end

# Data items are the input to the recurrent layer
function propagateForward(layer::recurrentLayer, inputs::Array{dataItem})
    # The input to the recurrent layer is the input concatenated with the recurrent layer's previous activation and a bias
    if layer.forward 
        for i in inputs
            push!(layer.activations, propagateForward(layer.weights, vcat(i.features..., layer.activations[end]..., 1), layer.params.activation))
        end
    else
        for i in Iterators.reverse(inputs)
            push!(layer.activations, propagateForward(layer.weights, vcat(i.features..., layer.activations[end]..., 1), layer.params.activation))
        end
    end
end

# The outputs of forward and backward recurrent layers are the inputs to the last layer
function propagateForward(layer::forwardLayer, forwardInputs::recurrentLayer, backwardInputs::recurrentLayer)
    layer.activations = propagateForward(layer.weights, vcat(forwardInputs.activations[end], backwardInputs.activations[end], 1), layer.params.activation)
end

function backpropLastLayer(targets_j::Array{Float64,1}, outputs_j::Array{Float64,1}, outputs_i::Array{Float64,1}, params_j::learningParams, stats::learningStatistics)
    error_j = (targets_j .- outputs_j) * params_j.fPrimeNet(outputs_j)
    δweights_ij = params_j.learningRate * error_j * transpose(outputs_i)
    return error_j, δweights_ij
end

function backprop(weights_jk::Array{Float64,2}, errors_k::Array{Float64,1}, outputs_j::Array{Float64,1}, outputs_i::Array{Float64,1}, params_j::learningParams, stats::learningStatistics)
    error_j = (errors_k * weights_jk) .* params_j.fPrimeNet(outputs_j)
    δweights_ij = params_j.learningRate * error_j * transpose(outputs_i)
    return error_j, δweights_ij
end

function bptt(network::brnnNetwork, inputs::Array{dataItem,1})
    bptt(network.outputLayer, network.recurrentForwardsLayer, network.recurrentBackwardsLayer, inputs);
end

function bptt(layer::recurrentLayer, errors_k::Array{Float64,1}, weights_jk::Array{Float64,2}, inputs::Array{dataItem,1})
    δweights = Array{Array{Float64,2},1}(undef, 0)
    layerError = errors_k
    if layer.forward
        for i in length(layer.activations) - 1:-1:2
            #print("Error $(size(layerError)), ")
            #println("Weights $(size(transpose(layer.weights)))")
            layerError, δweights_ij = backprop(weights_jk, layerError, layer.activations[i], vcat(layer.activations[i - 1]..., inputs[i].features..., 1), layer.params, layer.stats)
            push!(δweights, δweights_ij) 
        end
    else
        for i in length(layer.activations) - 1:-1:2
            layerError, δweights_ij = backprop(weights_jk, layerError, layer.activations[i], vcat(layer.activations[i - 1]..., inputs[length(layer.activations) - i + 1].features..., 1), layer.params, layer.stats)
            push!(δweights, δweights_ij) 
        end
    end
    #print("dweights $(size(δweights[1])), ")
    #println("weights $(size(layer.weights))")
    totalδweights = sum(δweights) ./ length(δweights)
    layer.weights .+= totalδweights;
end

function bptt(layer::forwardLayer, forwardInputs::recurrentLayer, backwardInputs::recurrentLayer, inputs::Array{dataItem})
    error, δweights = backpropLastLayer(inputs[end].labels, layer.activations, vcat(forwardInputs.activations[end]..., backwardInputs.activations[end]..., 1), layer.params, layer.stats)
    bptt(forwardInputs, error, layer.weights, inputs);
    bptt(backwardInputs, error, layer.weights, inputs);
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