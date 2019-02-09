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
import brnn: printArgs, brnnNetwork, learn, learningParams
import dataset: generateWeightedSumData, dataSet
# Here is the main body of the module
function run()
    dataSet = generateWeightedSumData(1000, 10, 20, true)
    brnn::brnnNetwork = brnnNetwork(1, 1, 1)
    params::learningParams = learningParams(.01, 30);
    learn(brnn, dataSet, params)
end






    printArgs()
    run()
end