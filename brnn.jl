module brnn
using Random
using dataset: dataItem, dataSet
using plugins: sigmoid, sigmoidPrime, SSE

####################
#### Data Structures
####################

struct learningParams
    activation::Function
    fPrimeNet::Function
    learningRate::Float64
    τ::Int64
end

struct layerStatistics
    averageWeightChange::Array{Float64,1}
end

# Stores data about each epoch
struct learningStatistics
    valErrors::Array{Float64,1}
    trainErrors::Array{Float64,1}
end

mutable struct forwardLayer
    activations::Array{Float64}
    weights::Array{Float64,2}
    inputSize::Int
    outputSize::Int
    params::learningParams
    stats::layerStatistics
end

mutable struct recurrentLayer
    activations::Array{Array{Float64}} #activations needed for each timestep, this is not a 2d-array since we don't know ahead of time one of the dimensions
    weights::Array{Float64,2} #weights only needed for one timestep
    inputSize::Int
    outputSize::Int
    forward::Bool
    params::learningParams
    stats::layerStatistics
end

mutable struct brnnNetwork
    recurrentForwardsLayer::recurrentLayer
    recurrentBackwardsLayer::recurrentLayer
    outputLayer::forwardLayer
    inputSize::Int
    hiddenSize::Int
    outputSize::Int
    params::learningParams
    stats::learningStatistics
end

#################
#### Constructors
#################

function learningParams(learningRate::Float64, τ::Int)
    return learningParams(sigmoid, sigmoidPrime, learningRate, τ)
end

function layerStatistics()
    return layerStatistics([0])
end

function learningStatistics()
    return learningStatistics([], [])
end

# Initalize a brnn with i inputs, n hidden nodes, and o output nodes
function brnnNetwork(i::Int, n::Int, o::Int, hiddenLearningParams::learningParams, outputLearningParams::learningParams, networkLearningParams::learningParams)
    forwardRecurrentLayer = recurrentLayer(i, n, hiddenLearningParams, true)
    backwardRecurrentLayer = recurrentLayer(i, n, hiddenLearningParams, false)
    outputLayer = forwardLayer(n * 2, o, outputLearningParams)
    stats = learningStatistics()
    return brnnNetwork(forwardRecurrentLayer, backwardRecurrentLayer, outputLayer, i, n, o, networkLearningParams, stats)
end

# Initalize a recurrentLayer with i inputs, and o output nodes
function recurrentLayer(i::Int, o::Int, params::learningParams, forward::Bool)
  # One timestep for now, we will add timesteps as we need to keep track of activations, and not before
    activations = [zeros(o)];
    weights = (rand(o, i + o + 1) .* 2) .- 1 #There is a weight to every input output and 
    return recurrentLayer(activations, weights, i, o, forward, params, layerStatistics())
end

# Initalize a regular forward layer with i inputs, and o output nodes
function forwardLayer(i::Int, o::Int, params::learningParams)
    activations = Array{Float64}(undef, o)
    weights = (rand(o, i + 1) .* 2) .- 1
    return forwardLayer(activations, weights, i, o, params, layerStatistics())
end

########################
#### Forward Propagation
########################

function clearActivations(layer::recurrentLayer)
    layer.activations = [zeros(layer.outputSize)];
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

# Forward Propagation From Inputs to Outputs
function propagateForward(network::brnnNetwork, inputs::Array{dataItem})
    # TODO: Call propagate forward for the appropriate number of time steps and the number of inputs
    clearActivations(network.recurrentForwardsLayer)
    clearActivations(network.recurrentBackwardsLayer)
    propagateForward(network.recurrentForwardsLayer, inputs)
    propagateForward(network.recurrentBackwardsLayer, inputs)
    propagateForward(network.outputLayer, network.recurrentForwardsLayer, network.recurrentBackwardsLayer)
end

####################
#### Backpropagation
####################

#j denotes current layer, i denotes input layer
function backpropLastLayer(targets_j::Array{Float64,1}, outputs_j::Array{Float64,1}, outputs_i::Array{Float64,1}, params_j::learningParams, stats::layerStatistics)
    #error, targets, outputs, are all j x 1 arrays
    error_j = (targets_j .- outputs_j) .* params_j.fPrimeNet(outputs_j)
    #error is j x 1, outputs is 1 x i (after transpose)
    #this leads to weights_ij being j x i (which is correct)
    δweights_ij::Array{Float64,2} = (params_j.learningRate * error_j) * transpose(outputs_i)
    return error_j, δweights_ij
end

#j denotes current layer, i denotes input layer, k denotes following layer
function backprop(weights_jk::Array{Float64,2}, errors_k::Array{Float64,1}, outputs_j::Array{Float64,1}, outputs_i::Array{Float64,1}, params_j::learningParams, stats::layerStatistics)
    #error_j is j x 1, weights_jk should be k x j, errors_k should be k x 1, outputs_j should be j x 1
    # println("outputs_j $(size(outputs_j))")
    # println("weights_jk $(size(weights_jk))")
    # println("errors_k $(size(errors_k))")
    error_j = (transpose(weights_jk) * errors_k)  .* params_j.fPrimeNet(outputs_j)
    #error is j x 1, outputs is 1 x i (after transpose)
    #this leads to weights_ij being j x i (which is correct)
    δweights_ij::Array{Float64,2} = (params_j.learningRate * error_j) *  transpose(outputs_i)
    return error_j, δweights_ij
end

function bptt(network::brnnNetwork, inputs::Array{dataItem,1})
    bptt(network.outputLayer, network.recurrentForwardsLayer, network.recurrentBackwardsLayer, inputs);
end

function bptt(layer::recurrentLayer, errors_k::Array{Float64,1}, weights_jk::Array{Float64,2}, inputs::Array{dataItem,1}, range::UnitRange{Int64})
    δweights = Array{Array{Float64,2},1}(undef, 0)
    layerError = errors_k
    # This is the half of the weights that take the forward layer activations as inputs
    forwardWeights = weights_jk[:,1:layer.outputSize]
    # This is the half of the weights that take the backwards layer activations as inputs (end - 1 excludes the bias)
    backwardWeights = weights_jk[:,layer.outputSize + 1:end - 1]
    if layer.forward
        for i in length(layer.activations) - 1:-1:2
            layerError, δweights_ij = backprop(forwardWeights, layerError, layer.activations[i], vcat(layer.activations[i - 1]..., inputs[i].features..., 1), layer.params, layer.stats)
            push!(δweights, reshape(δweights_ij[range], layer.outputSize, :)) 
            forwardWeights = layer.weights
            layerError = layerError[range]

        end
    else
        for i in length(layer.activations) - 1:-1:2
            layerError, δweights_ij = backprop(backwardWeights, layerError, layer.activations[i], vcat(layer.activations[i - 1]..., inputs[length(layer.activations) - i + 1].features..., 1), layer.params, layer.stats)
            push!(δweights, reshape(δweights_ij[range], layer.outputSize, :)) 
            backwardWeights = layer.weights
            layerError = layerError[range]
        end
    end
    #println("δweights $(size(δweights[1]))")
    #println("layer.weights $(size(layer.weights))")
    totalδweights = sum(δweights) ./ length(δweights)
    push!(layer.stats.averageWeightChange, sum(totalδweights) / length(totalδweights));
    layer.weights .+= totalδweights;
end

function bptt(layer::forwardLayer, forwardInputs::recurrentLayer, backwardInputs::recurrentLayer, inputs::Array{dataItem})
    error, δweights = backpropLastLayer(inputs[end].labels, layer.activations, vcat(forwardInputs.activations[end]..., backwardInputs.activations[end]..., 1), layer.params, layer.stats)
    bptt(forwardInputs, error, layer.weights, inputs, 1:forwardInputs.outputSize);
    bptt(backwardInputs, error, layer.weights, inputs, (forwardInputs.outputSize + 1):(forwardInputs.outputSize + backwardInputs.outputSize));
    push!(layer.stats.averageWeightChange, sum(δweights) / length(δweights));
    layer.weights .+= δweights;
end

#######################
#### Training Mechanics
#######################

#=
Train the network from a dataset
`patience`:     Number of epochs with no improvement after which training will be stopped.
`min_delta`:    Minimum change in the validation accuracy to qualify as an improvement,
                i.e. an absolute change of less than min_delta, will count as no improvement.
=#
function learn(network::brnnNetwork, data::dataSet, validation::dataSet, patience::Int, minDelta::Float64, maxEpochs::Int)
    window = Array{dataItem}(undef, 0)
    timesThrough = 0
    trainError = 0
    prevValError = Inf64
    valError = 0
    epoch = 1
    numNoImprovement = 0
    while numNoImprovement < patience && epoch <= maxEpochs
        # Train the model
        trainError = 0
        for item in data.examples
            push!(window, item) # Appends to the end
            if length(window) == network.params.τ
                propagateForward(network, window);
                trainError += SSE(item.labels, network.outputLayer.activations)
                # println("train: $(item.labels) - $(network.outputLayer.activations)")
                bptt(network, window);
                popfirst!(window) # Pops from the first
                timesThrough += 1
            end
        end
        
        # Validate the model
        window = Array{dataItem}(undef, 0)
        for item in validation.examples
            push!(window, item) # Appends to the end
            if length(window) == network.params.τ
                propagateForward(network, window);
                valError += SSE(item.labels, network.outputLayer.activations)
                # println("validate: $(item.labels) - $(network.outputLayer.activations)")
                popfirst!(window) # Pops from the first
            end
        end

        # Report
        println("Epoch $(epoch): Validation error: $(valError) Train error: $(trainError) numNoImprovement: $(numNoImprovement) ")

        # Housekeeping
        epoch += 1
        push!(network.stats.valErrors, valError / length(validation.examples)) # Mean Squared Error
        push!(network.stats.trainErrors, trainError / length(data.examples)) # Mean Squared Error
        valErrorDelta = prevValError - valError
        if (valErrorDelta < minDelta)
            numNoImprovement += 1
        else
            numNoImprovement = 0
        end
        prevValError = valError
        trainError = 0
        valError = 0
    end
    println("Learned from $(timesThrough) examples");
end 

# Test the network from a dataset
function test(network::brnnNetwork, data::dataSet)

end

end