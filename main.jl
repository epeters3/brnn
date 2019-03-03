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


function displayGraphs(network::BrnnNetwork, namePrefix::String)
    displayLayerStatistics(network.outputLayer.stats, "$(namePrefix)output")
    displayLayerStatistics(network.recurrentBackwardsLayer.stats, "$(namePrefix)backward")
    displayLayerStatistics(network.recurrentForwardsLayer.stats, "$(namePrefix)forward")
    displayLearningStatistics(network.stats, "$(namePrefix)brnn-stats")
end


# Here is the main body of the module
function runDparity()
    dataSet = generateDparityData(100, [1, 0, -1])
    validation = generateDparityData(10, [1, 0, -1])
    params::LearningParams = LearningParams(.1);
    brnn::BrnnNetwork = BrnnNetwork(1, 10, 1, params, 2, 2, params, params)
    learn(brnn, dataSet, validation, 20, .0001, 1000)
    mkpath("dparity")
    displayGraphs(brnn, "dparity/")
end


function runWeightedSumClassification()
    dataSet = generateWeightedSumData(10000, 10, 20, true)
    validation = generateWeightedSumData(1000, 10, 20, true)
    params::LearningParams = LearningParams(.09);
    brnn::BrnnNetwork = BrnnNetwork(1, 10, 1, params, 10, 20, params, params)
    learn(brnn, dataSet, validation, 25, .0001, 1000)
    
    mkpath("weightedSumClassification")
    displayGraphs(brnn, "weightedSumClassification/")
end

function runWeightedSumRegression()
    dataSet = generateWeightedSumData(10000, 10, 20, false)
    validation = generateWeightedSumData(1000, 10, 20, false)
    params::LearningParams = LearningParams(.01);
    brnn::BrnnNetwork = BrnnNetwork(1, 10, 1, params, 10, 20, params, params)
    learn(brnn, dataSet, validation, 25, .0001, 1000)
    
    mkpath("weightedSumRegression")
    displayGraphs(brnn, "weightedSumRegression/")
end

function runGesturesClassification()
    dataSet = getGesturesDataSet(1:1)
    println(length(dataSet.examples))
    validationSet = getGesturesDataSet(31:31)
    params::LearningParams = LearningParams(.01,keepStats=false);
    brnn::BrnnNetwork = BrnnNetwork(8, 20, 8, params, 10, 10, params, params)

    learn(brnn, dataSet, validationSet, 25, .0001, 100)

    mkpath("gesturesClassification")
    displayGraphs(brnn, "gesturesClassification/")
   
end
function run()
    runGesturesClassification()
    #runDparity()
    #runWeightedSumClassification();
    #runWeightedSumRegression();
end

run()

end