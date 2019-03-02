module brnn
using dataset: dataItem, dataSet
using plugins: sigmoid, sigmoidPrime, SSE, randGaussian, RPropMomentum

####################
#### Data Structures
####################

struct learningParams
    activation::Function
    fPrimeNet::Function
    addMomentum::Function
    learningRate::Float64
    keepStats::Bool
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
    deltaWeightsPrev::Array{Float64,2}
    inputSize::Int
    outputSize::Int
    params::learningParams
    stats::layerStatistics
end

mutable struct recurrentLayer
    activations::Array{Float64,2}
    weights::Array{Float64,2} #weights only needed for one timestep
    deltaWeightsPrev::Array{Float64,2}
    inputSize::Int
    outputSize::Int
    forward::Bool
    params::learningParams
    τ::Int64
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
    τ::Int64 # This is the sum of recurrentForwardsLayer.τ and recurrentBackwardsLayer.τ
    stats::learningStatistics
end

#################
#### Constructors
#################

function learningParams(learningRate::Float64, keepStats::Bool=true)
    momentumF = RPropMomentum(.5, 1.2)
    return learningParams(sigmoid, sigmoidPrime, momentumF, learningRate, keepStats)
end

function layerStatistics()
    return layerStatistics([0])
end

function learningStatistics()
    return learningStatistics([], [])
end

# Initalize a brnn with i inputs, n hidden nodes, and o output nodes
function brnnNetwork(i::Int, n::Int, o::Int, hiddenLearningParams::learningParams, forwardτ::Int64, backwardτ::Int64, outputLearningParams::learningParams, networkLearningParams::learningParams)
    fullτ = forwardτ + backwardτ
    forwardRecurrentLayer = recurrentLayer(i, n, hiddenLearningParams, forwardτ, true, fullτ + 1)
    backwardRecurrentLayer = recurrentLayer(i, n, hiddenLearningParams, backwardτ, false, fullτ + 1)
    outputLayer = forwardLayer(n * 2, o, outputLearningParams)
    stats = learningStatistics()
    return brnnNetwork(forwardRecurrentLayer, backwardRecurrentLayer, outputLayer, i, n, o, networkLearningParams, fullτ , stats)
end

# Initalize a recurrentLayer with i inputs, and o output nodes, and a recurrent window of size τ.
function recurrentLayer(i::Int, o::Int, params::learningParams, τ::Int64, forward::Bool, fullτ::Int64)
    activations = zeros(fullτ , i + o + 1);
    weights = randGaussian((o, i + o + 1), 0.0, 0.1) #There is a weight to every input output and 
    deltaWeightsPrev = zeros((o, i + o + 1))
    return recurrentLayer(activations, weights, deltaWeightsPrev, i, o, forward, params, τ, layerStatistics())
end

# Initalize a regular forward layer with i inputs, and o output nodes
function forwardLayer(i::Int, o::Int, params::learningParams)
    activations = Array{Float64}(undef, o)
    weights = randGaussian((o, i + 1), 0.0, 0.1)
    deltaWeightsPrev = zeros((o, i + 1))
    return forwardLayer(activations, weights, deltaWeightsPrev, i, o, params, layerStatistics())
end

########################
#### Forward Propagation
########################

function propagateForward(weights::Array{Float64,2}, inputs::Array{Float64,1}, activation::Function)
    return activation(weights * inputs)
end

# Data items are the input to the recurrent layer
function propagateForward(layer::recurrentLayer, inputs::Array{dataItem})
    # The input to the recurrent layer is the input concatenated with the recurrent layer's previous activation and a bias
    iterable = undef
    if layer.forward
        iterable = inputs
    else
        iterable = Iterators.reverse(inputs)
    end
    i = 1
    for input in iterable
        # Persist the input and bias with the previous activations
        # since we'll use it during part of backpropagation.
        numFeatures = length(input.features)
        layer.activations[i, end-numFeatures:end] = vcat(input.features..., 1)
        nextActivations = propagateForward(layer.weights, layer.activations[i, :], layer.params.activation)
        i += 1
        # Pad activations with zeros for now to preserve dimensions of layer.activations.
        # On the next loop iteration, they will be replaced by the inputs and bias.
        layer.activations[i, :] = vcat(nextActivations, zeros(numFeatures+1))
    end
end

# The outputs of forward and backward recurrent layers are the inputs to the last layer
# We don't pass the inputs and bias to the last layer.
function propagateForward(layer::forwardLayer, forwardInputs::recurrentLayer, backwardInputs::recurrentLayer)
    forwardActivations = forwardInputs.activations[end, 1:forwardInputs.outputSize]
    backwardActivations = backwardInputs.activations[end, 1:backwardInputs.outputSize]
    layer.activations = propagateForward(layer.weights, vcat(forwardActivations, backwardActivations, 1), layer.params.activation)
end

# Forward Propagation From Inputs to Outputs
function propagateForward(network::brnnNetwork, inputs::Array{dataItem})
    propagateForward(network.recurrentForwardsLayer, inputs)
    propagateForward(network.recurrentBackwardsLayer, inputs)
    propagateForward(network.outputLayer, network.recurrentForwardsLayer, network.recurrentBackwardsLayer)
end

####################
#### Backpropagation
####################

function findOutputError(targets::Array{Float64,1}, outputs::Array{Float64,1}, params::learningParams)
    return (targets .- outputs) .* params.fPrimeNet(outputs)
end

function findHiddenError(weights_jk::Array{Float64,2}, errors_k::Array{Float64,1}, outputs_j::Array{Float64,1}, params_j::learningParams)
    return (transpose(weights_jk) * errors_k)  .* params_j.fPrimeNet(outputs_j)
end

function findWeightChanges(error_j::Array{Float64,1}, outputs_i::Array{Float64,1}, params_j::learningParams)::Array{Float64,2}
    return (params_j.learningRate * error_j) * transpose(outputs_i)
end

#j denotes current layer, i denotes input layer
function backpropLastLayer(targets_j::Array{Float64,1}, outputs_j::Array{Float64,1}, outputs_i::Array{Float64,1}, params_j::learningParams, stats::layerStatistics)
    #error, targets, outputs, are all j x 1 arrays
    error_j = findOutputError(targets_j, outputs_j, params_j)
    #error is j x 1, outputs is 1 x i (after transpose)
    #this leads to weights_ij being j x i (which is correct)
    δweights_ij = findWeightChanges(error_j, outputs_i, params_j)
    return error_j, δweights_ij
end

#j denotes current layer, i denotes input layer, k denotes following layer
function backprop(weights_jk::Array{Float64,2}, errors_k::Array{Float64,1}, outputs_j::Array{Float64,1}, outputs_i::Array{Float64,1}, params_j::learningParams, stats::layerStatistics, outputSize::Int64)
    #error_j is j x 1, weights_jk should be k x j, errors_k should be k x 1, outputs_j should be j x 1
    # println("outputs_j $(size(outputs_j[1:outputSize]))")
    # println("weights_jk $(size(weights_jk[:, 1:outputSize]))")
    # println("errors_k $(size(errors_k))")
    error_j = findHiddenError(weights_jk[:, 1:outputSize], errors_k, outputs_j[1:outputSize], params_j)
    #error is j x 1, outputs is 1 x i (after transpose)
    #this leads to weights_ij being j x i (which is correct)
    δweights_ij = findWeightChanges(error_j, outputs_i, params_j)
    return error_j, δweights_ij
end

function bptt(network::brnnNetwork, target::dataItem)
    bptt(network.outputLayer, network.recurrentForwardsLayer, network.recurrentBackwardsLayer, target);
end

function bptt(layer::recurrentLayer, errors_k::Array{Float64,1})
    δweights = Array{Array{Float64,2},1}(undef, 0)
    layerError = errors_k
    for i in size(layer.activations, 1) - 1:-1:2
        # The persisted activation vector already contains the
        # appropriate input vector and bias value so just pass it as is.
        layerError, δweights_ij = backprop(layer.weights, layerError, layer.activations[i, :], layer.activations[i - 1, :], layer.params, layer.stats, layer.outputSize)
        push!(δweights, δweights_ij) 
    end
    # Momentum takes into account the last weight change and the current weight change
    totalδweights = layer.params.addMomentum(sum(δweights) ./ length(δweights), layer.deltaWeightsPrev)
    if layer.params.keepStats
        push!(layer.stats.averageWeightChange, sum(totalδweights) / length(totalδweights));
    end

    layer.deltaWeightsPrev = totalδweights
    layer.weights .+= totalδweights;
end

function bptt(layer::forwardLayer, forwardInputs::recurrentLayer, backwardInputs::recurrentLayer, target::dataItem)
    # We don't pass the bias and inputs from the recurrent layer to the last layer.
    activations = vcat(forwardInputs.activations[end, 1:forwardInputs.outputSize]..., backwardInputs.activations[end, 1:backwardInputs.outputSize]..., 1)
    outputError, δweights = backpropLastLayer(target.labels, layer.activations, activations, layer.params, layer.stats)
    hiddenError = findHiddenError(layer.weights, outputError, activations, layer.params) #note that the layer params should actually be the layer params for each of the forward/backward inputs
    bptt(forwardInputs, hiddenError[1:forwardInputs.outputSize]);
    bptt(backwardInputs, hiddenError[forwardInputs.outputSize + 1:end - 1]);

    actualδWeights = layer.params.addMomentum(δweights, layer.deltaWeightsPrev)
    layer.deltaWeightsPrev = actualδWeights
    if layer.params.keepStats
        push!(layer.stats.averageWeightChange, sum(actualδWeights) / length(actualδWeights));
    end
    layer.weights .+= actualδWeights;
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
            if length(window) == network.τ
                propagateForward(network, window);
                target = window[network.recurrentForwardsLayer.τ+1]
                trainError += SSE(target.labels, network.outputLayer.activations)
                # println("train: $(target.labels) - $(network.outputLayer.activations)")
                bptt(network, target);
                popfirst!(window) # Pops from the first
                timesThrough += 1
            end
        end
        
        # Validate the model
        window = Array{dataItem}(undef, 0)
        for item in validation.examples
            push!(window, item) # Appends to the end
            if length(window) == network.τ
                propagateForward(network, window);
                target = window[network.recurrentForwardsLayer.τ+1]
                valError += SSE(target.labels, network.outputLayer.activations)
                # println("validate: $(target.labels) - $(network.outputLayer.activations)")
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