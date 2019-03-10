module main
#=
In order to find Modules that are not in the standard LOAD_PATH and
be able to import them, we need to update the LOAD_PATH variable for
the current folder explicitly. Then we will be able to import a
local module appropriately.
Source: https://stackoverflow.com/questions/51824403/import-modules-and-functions-from-a-file-in-a-specific-directory-in-julia-1-0
=#
push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./brnn")
# Import local project modules

using brnn: BrnnNetwork, LearningParams, LearningStatistics
using train: learn
using dataset: generateWeightedSumData, generateDparityData, DataSet, getGesturesDataSet
using ml_plots: displayLayerStatistics, displayLearningStatistics, displaySweepGraph
using plugins: sigmoid, sigmoidPrime, linear, linearPrime, softmax, softmaxPrime, ReLU, ReLUPrime, tanH, tanHPrime


function displayGraphs(network::BrnnNetwork, namePrefix::String, isClassification::Bool; layerGraphs::Bool = true)
    if layerGraphs
        displayLayerStatistics(network.outputLayer.stats, "$(namePrefix)output-layer-stats")
        displayLayerStatistics(network.recurrentBackwardsLayer.stats, "$(namePrefix)backward-layer-stats")
        displayLayerStatistics(network.recurrentForwardsLayer.stats, "$(namePrefix)forward-layer-stats")
    end
    displayLearningStatistics(network.stats, "$(namePrefix)brnn-learning-stats", isClassification)
end


function paramSweep(lrates::Array{Float64,1}, networkFcn::Function, data::DataSet, validation::DataSet, targetOffset::Int, name::String; test::DataSet = DataSet(0), isClassification::Bool = true, patience::Int = 10, minDelta::Float64 = .001, minEpochs::Int = 50, maxEpochs::Int = 1000, numTries::Int = 1)
    println("beginning learning rate sweep for LR=$(lrates)")
    mkpath(name)
    bestValidationErrorSoFar::Float64 = 100000000
    bestModelSoFar::BrnnNetwork = networkFcn(1.0)
    network::BrnnNetwork = bestModelSoFar
    allTrainingStats::LearningStatistics = LearningStatistics()
    bestLr::Float64 = 0.0
    for lr in lrates
        avgLearningStats::LearningStatistics = LearningStatistics()
        for n in 1:numTries
            println("Try $(n)/$(numTries) (LR=$(lr))")
            network = networkFcn(lr)
            learn(network, data, validation, isClassification, patience, minDelta, minEpochs, maxEpochs, targetOffset)
            push!(avgLearningStats.trainErrors, minimum(network.stats.trainErrors))
            push!(avgLearningStats.valErrors, network.stats.bestValError)
            push!(avgLearningStats.valAccuracies, network.stats.bestValAccuracy)
            displayGraphs(network, "$name/lr$(lr)-trial$n-", isClassification; layerGraphs = false)
        end
        avgTrainError = sum(avgLearningStats.trainErrors) / length(avgLearningStats.trainErrors)
        avgValidationError = sum(avgLearningStats.valErrors) / length(avgLearningStats.valErrors)
        avgValidationAccuracy = sum(avgLearningStats.valAccuracies) / length(avgLearningStats.valAccuracies)
        if (avgValidationError < bestValidationErrorSoFar)
            bestModelSoFar = network
            bestLr = lr
            bestValidationErrorSoFar = avgValidationError
        end

        push!(allTrainingStats.trainErrors, avgTrainError)
        push!(allTrainingStats.valErrors, avgValidationError)
        push!(allTrainingStats.valAccuracies, avgValidationAccuracy)
       
    end
    displayGraphs(bestModelSoFar, "$name/best-model-lr$(bestLr)-", isClassification; layerGraphs = false)
    displaySweepGraph(allTrainingStats, "$name/brnn-learning-stats-sweep", isClassification, lrates)
end

######################################
########## BRNN Experiments ##########
######################################

# `h` is the number of hidden nodes.
function runDparity(dparityWindow::Array{Int64}, targetOffset::Int, h::Int64, name::String, isLstm::Bool)
    trainDataSize = 10000
    function dParityFcn(lr::Float64)
        rParams::LearningParams = LearningParams(lr, sigmoid, sigmoidPrime, keepStats = false)
        oParams::LearningParams = LearningParams(lr, sigmoid, sigmoidPrime, keepStats = false)
        return BrnnNetwork(1, h, 1, rParams, length(dparityWindow), oParams, isLstm)
    end
    dataSet = generateDparityData(trainDataSize, dparityWindow)
    validation = generateDparityData(Int64(trainDataSize / 10), dparityWindow)
    lrSweep = [.001, .005, .01, .03,  .05, .1, .2, .3]
    paramSweep(lrSweep, dParityFcn, dataSet, validation, targetOffset, name; isClassification = false, minDelta = .0001, minEpochs = 200, maxEpochs = 200, numTries = 1)
end

function runWeightedSumClassification(lrates::Array{Float64,1}, window::Array{Int64}, innerActivation::Function, innerActivationPrime::Function, name::String, isLstm::Bool)
    trainDataSize = 10000
    function weightedSumClassificationFcn(lr::Float64)
        rParams::LearningParams = LearningParams(lr, innerActivation, innerActivationPrime, keepStats = false)
        oParams::LearningParams = LearningParams(lr, softmax, softmaxPrime, keepStats = false)
        return BrnnNetwork(1, sum(window), 2, rParams, sum(window), oParams, isLstm)
    end
    dataSet = generateWeightedSumData(trainDataSize, window[1], window[2], true)
    validation = generateWeightedSumData(Int64(trainDataSize / 10), window[1], window[2], true)
    << << << < HEAD
    paramSweep(lrates, weightedSumClassificationFcn, dataSet, validation, window[1] + 1, name; minDelta = .0001, minEpochs = 200, maxEpochs = 200, numTries = 2)
    === === =
    lrSweep = [.001]
    paramSweep(lrSweep, weightedSumClassificationFcn, dataSet, validation, window[1] + 1, name; minDelta = .0001, minEpochs = 200, maxEpochs = 200, numTries = 1)
    >>> >>> > ae399095fbee8fa3024db744288ce7faddccaf91
end

function runWeightedSumRegression(lrates::Array{Float64,1}, window::Array{Int64}, innerActivation::Function, innerActivationPrime::Function, name::String, isLstm::Bool)
    trainDataSize = 10000
    function weightedSumRegressionFcn(lr::Float64)
        rParams::LearningParams = LearningParams(lr, innerActivation, innerActivationPrime, keepStats = false)
        oParams::LearningParams = LearningParams(lr, linear, linearPrime, keepStats = false)
        return BrnnNetwork(1, sum(window), 1, rParams, sum(window), oParams, isLstm)
    end
    dataSet = generateWeightedSumData(trainDataSize, window[1], window[2], false)
    validation = generateWeightedSumData(Int64(trainDataSize / 10), window[1], window[2], false)
    paramSweep(lrates, weightedSumRegressionFcn, dataSet, validation, window[1] + 1, name; isClassification = false,minDelta = .0001, minEpochs = 200, maxEpochs = 200, numTries = 2)
end


function runGesturesClassification(innerActivation::Function, innerActivationPrime::Function, name::String, isLstm::Bool)
    dataSet = getGesturesDataSet(1:5)
    validationSet = getGesturesDataSet(6:8)
    function gesturesClassificationFcn(lr::Float64)
        rParams::LearningParams = LearningParams(lr, innerActivation, innerActivationPrime, keepStats = false)
        oParams::LearningParams = LearningParams(lr, softmax, softmaxPrime, keepStats = false)

        brnn::BrnnNetwork = BrnnNetwork(8, 10, 8, rParams, 15, oParams, false, isLstm)
    end
    lrSweep = [.005, .01, .05, .1, .2]
    paramSweep(lrSweep, gesturesClassificationFcn, dataSet, validationSet, 10, name; isClassification = false,minDelta = .0001, minEpochs = 50, maxEpochs = 100, numTries = 1)
end

#############################
#### Body of Main Module ####
#############################

function run()

    #runGesturesClassification(sigmoid, sigmoidPrime, "gesturesClassificationSigmoid/", false)
    #runDparity([1,0,-1], 2, "Dparity3", false)
    runDparity([2,1,0,-1,-2], 3, "Dparity5", false)
    #runDparity([4,3,2,1,0,-1,-2,-3,-4], 5, "Dparity9", false)   
    #lrSweep = [.001, .005, .01, .03,  .05, .1, .2]
    #smallWeightedSum = [5, 10]
    #largeWeightedSum = [10, 20]
    #runWeightedSumClassification(lrSweep,smallWeightedSum, sigmoid, sigmoidPrime, "weightedSumClassificationSmallSigmoid/", false)
    #runWeightedSumRegression(lrSweep,smallWeightedSum, tanH, tanHPrime, "weightedSumRegressionSmallTanH/", false)
    #runWeightedSumClassification([.001, .003, .005, .01], largeWeightedSum, sigmoid, sigmoidPrime, "weightedSumClassificationLargeSigmoid/", false)
    #runWeightedSumRegression([.001, .003, .005, .01], largeWeightedSum, tanH, tanHPrime, "weightedSumRegressionLargeTanH/", false)
    ##runWeightedSumClassification(ReLU, ReLUPrime, "weightedSumClassificationReLU/", false)

end

run()

end