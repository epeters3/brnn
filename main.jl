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

# Here is the main body of the module
function runDparity()
    dparityWindow = [1, 0, -1]
    testSetSize = 10000
    lr = .3
    dataSet = generateDparityData(testSetSize, dparityWindow)
    validation = generateDparityData(Int64(testSetSize / 10), dparityWindow)
    rParams::LearningParams = LearningParams(lr, sigmoid, sigmoidPrime, keepStats = false)
    oParams::LearningParams = LearningParams(lr, linear, linearPrime, keepStats = false)
    brnn::BrnnNetwork = BrnnNetwork(1, 10, 1, rParams, length(dparityWindow), oParams, false)
    learn(brnn, dataSet, validation, false, 10, .01, 50, 1000, 2)
    mkpath("dparity")
    displayGraphs(brnn, "dparity/", false, layerGraphs = false)
end

function runWeightedSumClassification(innerActivation::Function, innerActivationPrime::Function, name::String)
    window = [5,10]
    trainDataSize = 10000
    function weightedSumClassificationFcn(lr::Float64)
        rParams::LearningParams = LearningParams(lr, innerActivation, innerActivationPrime, keepStats = false)
        oParams::LearningParams = LearningParams(lr, softmax, softmaxPrime, keepStats = false)
        return BrnnNetwork(1, 15, 2, rParams, sum(window), oParams, false)
    end
    dataSet = generateWeightedSumData(trainDataSize, window[1], window[2], true)
    validation = generateWeightedSumData(Int64(trainDataSize / 10), window[1], window[2], true)
    lrSweep = [.001, .005, .01, .03,  .05, .1, .2]
    paramSweep(lrSweep, weightedSumClassificationFcn, dataSet, validation, window[1] + 1, name; minDelta = .0001, minEpochs = 200, maxEpochs = 200, numTries = 2)
end

function runWeightedSumRegression(innerActivation::Function, innerActivationPrime::Function, name::String)
    window = [5,10]
    trainDataSize = 10000
    function weightedSumRegressionFcn(lr::Float64)
        rParams::LearningParams = LearningParams(lr, innerActivation, innerActivationPrime, keepStats = false)
        oParams::LearningParams = LearningParams(lr, linear, linearPrime, keepStats = false)
        return BrnnNetwork(1, 15, 1, rParams, sum(window), oParams, false)
    end
    dataSet = generateWeightedSumData(trainDataSize, window[1], window[2], false)
    validation = generateWeightedSumData(Int64(trainDataSize / 10), window[1], window[2], false)
    lrSweep = [.001, .005, .01, .03,  .05, .1, .2]
    paramSweep(lrSweep, weightedSumRegressionFcn, dataSet, validation, window[1] + 1, name; isClassification = false,minDelta = .0001, minEpochs = 200, maxEpochs = 200, numTries = 2)
end


function runGesturesClassification(innerActivation::Function, innerActivationPrime::Function, name::String)
    dataSet = getGesturesDataSet(1:5)
    validationSet = getGesturesDataSet(6:8)
    function gesturesClassificationFcn(lr::Float64)
        rParams::LearningParams = LearningParams(lr, innerActivation, innerActivationPrime, keepStats = false)
        oParams::LearningParams = LearningParams(lr, softmax, softmaxPrime, keepStats = false)
        brnn::BrnnNetwork = BrnnNetwork(8, 10, 8, rParams, 10, oParams, false)
    end
    lrSweep = [.005, .01, .05, .1, .2]
    paramSweep(lrSweep, gesturesClassificationFcn, dataSet, validationSet, 10, name; isClassification = false,minDelta = .0001, minEpochs = 50, maxEpochs = 100, numTries = 1)
end

######################################
########## LSTM Experiments ##########
######################################

function runDparityLstm()
    dparityWindow = [1, 0, -1]
    testSetSize = 10000
    lr = .3
    dataSet = generateDparityData(testSetSize, dparityWindow)
    validation = generateDparityData(Int64(testSetSize / 10), dparityWindow)
    rParams::LearningParams = LearningParams(lr, sigmoid, sigmoidPrime, keepStats = false)
    oParams::LearningParams = LearningParams(lr, linear, linearPrime, keepStats = false)
    brnn::BrnnNetwork = BrnnNetwork(1, 10, 1, rParams, length(dparityWindow), oParams, true)
    learn(brnn, dataSet, validation, false, 10, .01, 50, 1000, 2)
    mkpath("dparity")
    displayGraphs(brnn, "dparity/", false, layerGraphs = false)
end


function runWeightedSumClassificationLstm(innerActivation::Function, innerActivationPrime::Function, name::String)
    window = [10,20]
    trainDataSize = 10000
    function weightedSumClassificationFcn(lr::Float64)
        rParams::LearningParams = LearningParams(lr, innerActivation, innerActivationPrime, keepStats = false)
        oParams::LearningParams = LearningParams(lr, softmax, softmaxPrime, keepStats = false)
        return BrnnNetwork(1, 30, 2, rParams, sum(window), oParams, true)
    end
    dataSet = generateWeightedSumData(trainDataSize, window[1], window[2], true)
    validation = generateWeightedSumData(Int64(trainDataSize / 10), window[1], window[2], true)
    lrSweep = [.001, .01, .1, .2, .3]
    paramSweep(lrSweep, weightedSumClassificationFcn, dataSet, validation, window[1] + 1, name; minDelta = .0001, minEpochs = 100, maxEpochs = 100, numTries = 1)
end

function runWeightedSumRegressionLstm()
    lr = .03
    window = [10,20]
    trainDataSize = 1000
    dataSet = generateWeightedSumData(trainDataSize, window[1], window[2], false)
    validation = generateWeightedSumData(Int64(trainDataSize / 10), window[1], window[2], false)
    rParams::LearningParams = LearningParams(lr, sigmoid, sigmoidPrime)
    oParams::LearningParams = LearningParams(lr, linear, linearPrime)
    brnn::BrnnNetwork = BrnnNetwork(1, 20, 1, rParams, sum(window), oParams, true)
    learn(brnn, dataSet, validation, false, 25, .0001, 80, 1000, window[1] + 1)
    
    mkpath("weightedSumRegression")
    displayGraphs(brnn, "weightedSumRegression/", false)
end

function runGesturesClassificationLstm()
    lr = .01
    dataSet = getGesturesDataSet(1:5)
    validationSet = getGesturesDataSet(31:32)
    rParams::LearningParams = LearningParams(lr, sigmoid, sigmoidPrime, keepStats = false)
    oParams::LearningParams = LearningParams(lr, softmax, softmaxPrime, keepStats = false)
    brnn::BrnnNetwork = BrnnNetwork(8, 20, 8, rParams, 20, oParams, true)

    learn(brnn, dataSet, validationSet, true, 25, .0001, 10, 100, 11)

    mkpath("gesturesClassification")
    displayGraphs(brnn, "gesturesClassification/", true)
   
end

#############################
#### Body of Main Module ####
#############################

function run()
    # BRNN Experiments
    ##################
    #runGesturesClassification(sigmoid, sigmoidPrime, "gesturesClassificationSigmoid/")
    #runDparity()
    #runWeightedSumClassification(sigmoid, sigmoidPrime, "weightedSumClassificationSigmoid/")
    runWeightedSumRegression(tanH, tanHPrime, "weightedSumRegressionSigmoid/")
    ##runWeightedSumClassification(ReLU, ReLUPrime, "weightedSumClassificationReLU/")


    # LSTM-BRNN Experiments
    #######################
    #runDparityLstm()
    #runWeightedSumClassificationLstm(sigmoid, sigmoidPrime, "weightedSumClassificationSigmoid/")
    #runWeightedSumClassificationLstm(ReLU, ReLUPrime, "weightedSumClassificationReLU/")
    #runWeightedSumRegressionLstm()
    #runGesturesClassificationLstm()
end

run()

end