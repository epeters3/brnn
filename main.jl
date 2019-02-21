module main
#=
In order to find Modules that are not in the standard LOAD_PATH and
be able to import them, we need to update the LOAD_PATH variable for
the current folder explicitly. Then we will be able to import a
local module appropriately.
Source: https://stackoverflow.com/questions/51824403/import-modules-and-functions-from-a-file-in-a-specific-directory-in-julia-1-0
=#
push!(LOAD_PATH, "./")
# Import local project modules
import brnn: brnnNetwork, learn, learningParams
import dataset: generateWeightedSumData, generateDparityData, dataSet
import ml_plots: displayLayerStatistics, displayLearningStatistics

function displayGraphs(network::brnnNetwork, namePrefix::String)
    displayLayerStatistics(network.outputLayer.stats, "$(namePrefix)output")
    displayLayerStatistics(network.recurrentBackwardsLayer.stats, "$(namePrefix)backward")
    displayLayerStatistics(network.recurrentForwardsLayer.stats, "$(namePrefix)forward")
    displayLearningStatistics(network.stats, "$(namePrefix)brnn-stats")
end


# Here is the main body of the module
function runDparity()
    dataSet = generateDparityData(100, [1, 0, -1])
    validation = generateDparityData(10, [1, 0, -1])
    params::learningParams = learningParams(.1, 3);
    brnn::brnnNetwork = brnnNetwork(1, 10, 1, params, params, params)
    learn(brnn, dataSet, validation, 20, .0001, 1000)
    mkpath("dparity")
    displayGraphs(brnn, "dparity/")
end


function runWeightedSumClassification()
    dataSet = generateWeightedSumData(10000, 10, 20, true)
    validation = generateWeightedSumData(1000, 10, 20, true)
    params::learningParams = learningParams(.09, 10);
    brnn::brnnNetwork = brnnNetwork(1, 10, 1, params, params, params)
    learn(brnn, dataSet, validation, 25, .0001, 1000)
    
    mkpath("weightedSumClassification")
    displayGraphs(brnn, "weightedSumClassification/")
end

function runWeightedSumRegression()
    dataSet = generateWeightedSumData(10000, 10, 20, false)
    validation = generateWeightedSumData(1000, 10, 20, false)
    params::learningParams = learningParams(.01, 10);
    brnn::brnnNetwork = brnnNetwork(1, 10, 1, params, params, params)
    learn(brnn, dataSet, validation, 25, .0001, 1000)
    
    mkpath("weightedSumRegression")
    displayGraphs(brnn, "weightedSumRegression/")
end

function run()
    #runDparity()
    runWeightedSumClassification();
    #runWeightedSumRegression();
end

run()

end