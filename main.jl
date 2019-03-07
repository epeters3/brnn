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

using brnn: BrnnNetwork, LearningParams
using train: learn
using dataset: generateWeightedSumData, generateDparityData, DataSet, getGesturesDataSet
using ml_plots: displayLayerStatistics, displayLearningStatistics
using plugins: sigmoid, sigmoidPrime, linear, linearPrime, softmax, softmaxPrime


function displayGraphs(network::BrnnNetwork, namePrefix::String, isClassification::Bool)
    displayLayerStatistics(network.outputLayer.stats, "$(namePrefix)output")
    displayLayerStatistics(network.recurrentBackwardsLayer.stats, "$(namePrefix)backward")
    displayLayerStatistics(network.recurrentForwardsLayer.stats, "$(namePrefix)forward")
    displayLearningStatistics(network.stats, "$(namePrefix)brnn-stats", isClassification)
end


# Here is the main body of the module
function runDparity()
    dparityWindow = [1, 0, -1]
    testSetSize = 10000
    lr = .3
    dataSet = generateDparityData(testSetSize, dparityWindow)
    validation = generateDparityData(Int64(testSetSize / 10), dparityWindow)
    rParams::LearningParams = LearningParams(lr, sigmoid, sigmoidPrime)
    oParams::LearningParams = LearningParams(lr, linear, linearPrime)
    brnn::BrnnNetwork = BrnnNetwork(1, 10, 1, rParams, length(dparityWindow), oParams, false)
    learn(brnn, dataSet, validation, false, 10, .01, 50, 1000, 2)
    mkpath("dparity")
    displayGraphs(brnn, "dparity/", false)
end


function runWeightedSumClassification()
    lr = .03
    dataSet = generateWeightedSumData(10000, 5, 8, true)
    validation = generateWeightedSumData(1000, 5, 8, true)
    rParams::LearningParams = LearningParams(lr, sigmoid, sigmoidPrime, keepStats=false)
    oParams::LearningParams = LearningParams(lr, softmax, softmaxPrime, keepStats=false)
    brnn::BrnnNetwork = BrnnNetwork(1, 20, 2, rParams, 13, oParams, false)
    learn(brnn, dataSet, validation, true, 10, .0001, 80, 1000, 6)
    
    mkpath("weightedSumClassification")
    displayGraphs(brnn, "weightedSumClassification/", true)
end

function runWeightedSumRegression()
    lr = .01
    dataSet = generateWeightedSumData(10000, 10, 20, false)
    validation = generateWeightedSumData(1000, 10, 20, false)
    rParams::LearningParams = LearningParams(lr, sigmoid, sigmoidPrime)
    oParams::LearningParams = LearningParams(lr, linear, linearPrime)
    brnn::BrnnNetwork = BrnnNetwork(1, 10, 1, rParams, 30, oParams, false)
    learn(brnn, dataSet, validation, false, 25, .0001, 30, 1000, 11)
    
    mkpath("weightedSumRegression")
    displayGraphs(brnn, "weightedSumRegression/", false)
end

function runGesturesClassification()
    lr = .01
    dataSet = getGesturesDataSet(1:1)
    println(length(dataSet.examples))
    validationSet = getGesturesDataSet(31:31)
    rParams::LearningParams = LearningParams(lr, sigmoid, sigmoidPrime, keepStats=false)
    oParams::LearningParams = LearningParams(lr, softmax, softmaxPrime, keepStats=false)
    brnn::BrnnNetwork = BrnnNetwork(8, 20, 8, rParams, 20, oParams, false)

    learn(brnn, dataSet, validationSet, true, 25, .0001, 30, 100, 11)

    mkpath("gesturesClassification")
    displayGraphs(brnn, "gesturesClassification/", true)
   
end
function run()
    #runGesturesClassification()
    #runDparity()
    runWeightedSumClassification();
    #runWeightedSumRegression();
end

run()

end