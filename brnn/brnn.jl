module brnn
using dataset: DataItem
using plugins: sigmoid, sigmoidPrime, randGaussian, RPropMomentum

struct LearningParams
    activation::Function
    fPrimeNet::Function
    addMomentum::Function
    learningRate::Float64
    keepStats::Bool
end

function LearningParams(learningRate::Float64; keepStats::Bool=true)
    momentumF = RPropMomentum(.5, 1.2)
    return LearningParams(sigmoid, sigmoidPrime, momentumF, learningRate, keepStats)
end


struct LayerStatistics
    averageWeightChange::Array{Float64,1}
end

function LayerStatistics()
    return LayerStatistics([0])
end


# Stores data about each epoch
struct LearningStatistics
    valErrors::Array{Float64,1}
    trainErrors::Array{Float64,1}
end

function LearningStatistics()
    return LearningStatistics([], [])
end


mutable struct ConnectedLayer
    activations::Array{Float64}
    weights::Array{Float64,2}
    deltaWeightsPrev::Array{Float64,2}
    inputSize::Int
    outputSize::Int
    params::LearningParams
    stats::LayerStatistics
end

# Initalize a regular connected layer with i inputs, and o output nodes
function ConnectedLayer(i::Int, o::Int, params::LearningParams)
    activations = Array{Float64}(undef, o)
    weights = randGaussian((o, i + 1), 0.0, 0.1)
    deltaWeightsPrev = zeros((o, i + 1))
    return ConnectedLayer(activations, weights, deltaWeightsPrev, i, o, params, LayerStatistics())
end


mutable struct RecurrentLayer
    activations::Array{Float64,2}
    weights::Array{Float64,2} #weights only needed for one timestep
    deltaWeightsPrev::Array{Float64,2}
    inputSize::Int
    outputSize::Int
    isForward::Bool
    params::LearningParams
    τ::Int64
    stats::LayerStatistics
end

# Initalize a RecurrentLayer with i inputs, and o output nodes, and a recurrent window of size τ.
function RecurrentLayer(i::Int, o::Int, params::LearningParams, τ::Int64, isForward::Bool)
    activations = zeros(τ+1, i + o + 1);
    weights = randGaussian((o, i + o + 1), 0.0, 0.1) #There is a weight to every input output and 
    deltaWeightsPrev = zeros((o, i + o + 1))
    return RecurrentLayer(activations, weights, deltaWeightsPrev, i, o, isForward, params, τ, LayerStatistics())
end


mutable struct RecurrentLayerLstm
    activations::Array{Float64,2}
    weights::Array{Float64,2} #weights only needed for one timestep
    deltaWeightsPrev::Array{Float64,2}
    inputSize::Int
    outputSize::Int
    isForward::Bool
    params::LearningParams
    τ::Int64
    stats::LayerStatistics
end

AnyRecurrentLayer = Union{RecurrentLayer, RecurrentLayerLstm}

mutable struct BrnnNetwork
    recurrentForwardsLayer::AnyRecurrentLayer
    recurrentBackwardsLayer::AnyRecurrentLayer
    outputLayer::ConnectedLayer
    inputSize::Int
    hiddenSize::Int
    outputSize::Int
    params::LearningParams
    τ::Int64 # This is the sum of recurrentForwardsLayer.τ and recurrentBackwardsLayer.τ
    stats::LearningStatistics
end

# Initalize a brnn with i inputs, n hidden nodes, and o output nodes
function BrnnNetwork(i::Int, n::Int, o::Int, hiddenLearningParams::LearningParams, forwardτ::Int64, backwardτ::Int64, outputLearningParams::LearningParams, networkLearningParams::LearningParams)
    forwardRecurrentLayer = RecurrentLayer(i, n, hiddenLearningParams, forwardτ, true)
    backwardRecurrentLayer = RecurrentLayer(i, n, hiddenLearningParams, backwardτ, false)
    outputLayer = ConnectedLayer(n * 2, o, outputLearningParams)
    stats = LearningStatistics()
    return BrnnNetwork(forwardRecurrentLayer, backwardRecurrentLayer, outputLayer, i, n, o, networkLearningParams, forwardτ+backwardτ, stats)
end

########################
#### Forward Propagation
########################

function _activate(weights::Array{Float64,2}, inputs::Array{Float64,1}, activation::Function)
    return activation(weights * inputs)
end

# Data items are the input to the recurrent layer
function _propForwardRecurrent(layer::RecurrentLayer, inputs::Array{DataItem})
    # The input to the recurrent layer is the input concatenated with the recurrent layer's previous activation and a bias
    iterable = undef
    if layer.isForward
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
        nextActivations = _activate(layer.weights, layer.activations[i, :], layer.params.activation)
        i += 1
        # Pad activations with zeros for now to preserve dimensions of layer.activations.
        # On the next loop iteration, they will be replaced by the inputs and bias.
        layer.activations[i, :] = vcat(nextActivations, zeros(numFeatures+1))
    end
end

# The outputs of forward and backward recurrent layers are the inputs to the last layer
# We don't pass the inputs and bias to the last layer.
function _propForwardConnected(layer::ConnectedLayer, forwardInputs::RecurrentLayer, backwardInputs::RecurrentLayer)
    forwardActivations = forwardInputs.activations[end, 1:forwardInputs.outputSize]
    backwardActivations = backwardInputs.activations[end, 1:backwardInputs.outputSize]
    layer.activations = _activate(layer.weights, vcat(forwardActivations, backwardActivations, 1), layer.params.activation)
end

# Forward Propagation From Inputs to Outputs
function propagateForward(network::BrnnNetwork, inputs::Array{DataItem})
    _propForwardRecurrent(network.recurrentForwardsLayer, inputs[1:network.recurrentForwardsLayer.τ])
    _propForwardRecurrent(network.recurrentBackwardsLayer, inputs[network.recurrentForwardsLayer.τ+1:end])
    _propForwardConnected(network.outputLayer, network.recurrentForwardsLayer, network.recurrentBackwardsLayer)
end

####################
#### Backpropagation
####################

function _findOutputError(targets::Array{Float64,1}, outputs::Array{Float64,1}, params::LearningParams)
    return (targets .- outputs) .* params.fPrimeNet(outputs)
end

function _findHiddenError(weights_jk::Array{Float64,2}, errors_k::Array{Float64,1}, outputs_j::Array{Float64,1}, params_j::LearningParams)
    return (transpose(weights_jk) * errors_k)  .* params_j.fPrimeNet(outputs_j)
end

function _findWeightChanges(error_j::Array{Float64,1}, outputs_i::Array{Float64,1}, params_j::LearningParams)::Array{Float64,2}
    return (params_j.learningRate * error_j) * transpose(outputs_i)
end

#j denotes current layer, i denotes input layer
function _backpropLastLayer(targets_j::Array{Float64,1}, outputs_j::Array{Float64,1}, outputs_i::Array{Float64,1}, params_j::LearningParams, stats::LayerStatistics)
    #error, targets, outputs, are all j x 1 arrays
    error_j = _findOutputError(targets_j, outputs_j, params_j)
    #error is j x 1, outputs is 1 x i (after transpose)
    #this leads to weights_ij being j x i (which is correct)
    δweights_ij = _findWeightChanges(error_j, outputs_i, params_j)
    return error_j, δweights_ij
end

#j denotes current layer, i denotes input layer, k denotes following layer
function _backprop(weights_jk::Array{Float64,2}, errors_k::Array{Float64,1}, outputs_j::Array{Float64,1}, outputs_i::Array{Float64,1}, params_j::LearningParams, stats::LayerStatistics, outputSize::Int64)
    #error_j is j x 1, weights_jk should be k x j, errors_k should be k x 1, outputs_j should be j x 1
    # println("outputs_j $(size(outputs_j[1:outputSize]))")
    # println("weights_jk $(size(weights_jk[:, 1:outputSize]))")
    # println("errors_k $(size(errors_k))")
    error_j = _findHiddenError(weights_jk[:, 1:outputSize], errors_k, outputs_j[1:outputSize], params_j)
    #error is j x 1, outputs is 1 x i (after transpose)
    #this leads to weights_ij being j x i (which is correct)
    δweights_ij = _findWeightChanges(error_j, outputs_i, params_j)
    return error_j, δweights_ij
end

function bptt(network::BrnnNetwork, target::DataItem)
    _bptt(network.outputLayer, network.recurrentForwardsLayer, network.recurrentBackwardsLayer, target);
end

function _bpttRecurrent(layer::RecurrentLayer, errors_k::Array{Float64,1})
    δweights = Array{Array{Float64,2},1}(undef, 0)
    layerError = errors_k
    for i in size(layer.activations, 1) - 1:-1:2
        # The persisted activation vector already contains the
        # appropriate input vector and bias value so just pass it as is.
        layerError, δweights_ij = _backprop(layer.weights, layerError, layer.activations[i, :], layer.activations[i - 1, :], layer.params, layer.stats, layer.outputSize)
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

function _bptt(layer::ConnectedLayer, forwardInputs::RecurrentLayer, backwardInputs::RecurrentLayer, target::DataItem)
    # We don't pass the bias and inputs from the recurrent layer to the last layer.
    activations = vcat(forwardInputs.activations[end, 1:forwardInputs.outputSize]..., backwardInputs.activations[end, 1:backwardInputs.outputSize]..., 1)
    outputError, δweights = _backpropLastLayer(target.labels, layer.activations, activations, layer.params, layer.stats)
    hiddenError = _findHiddenError(layer.weights, outputError, activations, layer.params)
    _bpttRecurrent(forwardInputs, hiddenError[1:forwardInputs.outputSize]);
    _bpttRecurrent(backwardInputs, hiddenError[forwardInputs.outputSize + 1:end - 1]);

    actualδWeights = layer.params.addMomentum(δweights, layer.deltaWeightsPrev)
    layer.deltaWeightsPrev = actualδWeights
    if layer.params.keepStats
        push!(layer.stats.averageWeightChange, sum(actualδWeights) / length(actualδWeights));
    end 
    layer.weights .+= actualδWeights;
end

end # module brnn