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
import ml_plots: displayLayerStatistics

# Here is the main body of the module
function run()
    dataSet = generateDparityData(100, [1, 0, -1])
    validation = generateDparityData(10, [1, 0, -1])
    params::learningParams = learningParams(.1, 3);
    brnn::brnnNetwork = brnnNetwork(1, 1, 1, params, params, params)
    learn(brnn, dataSet, validation, 20, .0001, 1000)
    displayLayerStatistics(brnn.outputLayer.stats, "output.jpg")
    displayLayerStatistics(brnn.recurrentBackwardsLayer.stats, "backward.jpg")
    displayLayerStatistics(brnn.recurrentForwardsLayer.stats, "forward.jpg")
end

run()

end