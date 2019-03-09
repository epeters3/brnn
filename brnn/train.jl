module train
using dataset: DataItem, DataSet
using brnn: bptt, propagateForward, BrnnNetwork, LearningParams
using plugins: SSE, crossEntropy

#######################
#### Training Mechanics
#######################

#=
Train the network from a dataset
`patience`:     Number of epochs with no improvement after which training will be stopped.
`min_delta`:    Minimum change in the validation accuracy to qualify as an improvement,
                i.e. an absolute change of less than min_delta, will count as no improvement.
=#
function learn(network::BrnnNetwork, data::DataSet, validation::DataSet, isClassification::Bool, patience::Int, minDelta::Float64, minEpochs::Int, maxEpochs::Int, targetOffset::Int)
    window = Array{DataItem}(undef, 0)
    timesThrough = 0
    trainError = 0
    valError = 0
    epoch = 1
    numNoImprovement = 0
    # We never get to test on the first left window items or the last right window - 1 items.
    trainLearnableSize = length(data.examples) - network.τ + 1
    validationLearnableSize = length(validation.examples) - network.τ + 1
    while epoch <= minEpochs || (numNoImprovement < patience && epoch <= maxEpochs)
        # Train the model
        trainError = 0
        for item in data.examples
            push!(window, item) # Appends to the end
            if length(window) == network.τ
                propagateForward(network, window);
                target = window[targetOffset]
                if isClassification
                    trainError += crossEntropy(target.labels, network.outputLayer.activations)
                else
                    trainError += SSE(target.labels, network.outputLayer.activations)
                end
                # println("train: $(target.features) -> ($(target.labels) == $(network.outputLayer.activations))?")
                bptt(network, target);
                popfirst!(window) # Pops from the first
                timesThrough += 1
            end
        end
        
        # Validate the model
        window = Array{DataItem}(undef, 0)
        valNumCorrect = 0
        for item in validation.examples
            push!(window, item) # Appends to the end
            if length(window) == network.τ
                propagateForward(network, window);
                target = window[targetOffset]
                if isClassification
                    valError += crossEntropy(target.labels, network.outputLayer.activations)
                    if findmax(target.labels)[2] == findmax(network.outputLayer.activations)[2]
                        # The model predicted the correct class
                        valNumCorrect += 1
                    end
                else
                    valError += SSE(target.labels, network.outputLayer.activations)
                end
                # println("validate: $(target.labels) - $(network.outputLayer.activations)")
                popfirst!(window) # Pops from the first
            end
        end

        # Track best val error and best val accuracy

        trainError /= trainLearnableSize # Scale by the size of the learnable dataset
        valError /= validationLearnableSize # Scale by the size of the learnable dataset
        valAccuracy = valNumCorrect / validationLearnableSize

        valErrorDelta = network.stats.bestValError - valError
        if valError < network.stats.bestValError
            network.stats.bestValError = valError
        end
        if valAccuracy > network.stats.bestValAccuracy
            network.stats.bestValAccuracy = valAccuracy
        end
        if valErrorDelta < minDelta
            numNoImprovement += 1
        else
            numNoImprovement = 0
        end
        
        # Report
        if isClassification
            # MCE is Mean Cross Entropy
            println("Epoch $(epoch): Val MCE: $(valError) Train MCE: $(trainError) Val accuracy: $(valAccuracy) numNoImprovement: $(numNoImprovement)")
        else
            println("Epoch $(epoch): Val MSE: $(valError) Train MSE: $(trainError) numNoImprovement: $(numNoImprovement)")
        end

        # Record more stats
        push!(network.stats.trainErrors, trainError)
        push!(network.stats.valErrors, valError)
        if isClassification
            push!(network.stats.valAccuracies, valAccuracy)
        end
        
        # Housekeeping
        epoch += 1
        trainError = 0
        valError = 0
    end
    println("Learned from $(timesThrough) examples");
end 

# Test the network from a dataset
function test(network::BrnnNetwork, data::DataSet)

end

end # module train