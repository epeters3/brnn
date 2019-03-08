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
    minError = Inf64
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
        for item in validation.examples
            push!(window, item) # Appends to the end
            if length(window) == network.τ
                propagateForward(network, window);
                target = window[targetOffset]
                if isClassification
                    valError += crossEntropy(target.labels, network.outputLayer.activations)
                else
                    valError += SSE(target.labels, network.outputLayer.activations)
                end
                # println("validate: $(target.labels) - $(network.outputLayer.activations)")
                popfirst!(window) # Pops from the first
            end
        end

        # Report
        valErrorDelta = minError - valError
        if valError < minError
            minError = valError
        end
        if (valErrorDelta < minDelta)
            numNoImprovement += 1
        else
            numNoImprovement = 0
        end
        println("Epoch $(epoch): Validation error: $(valError) Train error: $(trainError) numNoImprovement: $(numNoImprovement) ")
        
        # Housekeeping
        push!(network.stats.trainErrors, trainError / length(data.examples)) # Mean Squared Error
        push!(network.stats.valErrors, valError / length(validation.examples)) # Mean Squared Error
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