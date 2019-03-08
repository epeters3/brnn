module ml_plots
using brnn: LayerStatistics, LearningStatistics
using utils: downSample
using Plots


function displayLayerStatistics(stats::LayerStatistics, name::String)
    sampledAvgWeightChange = downSample(stats.averageWeightChange, 100)
    plt = plot(1:length(sampledAvgWeightChange), sampledAvgWeightChange)
    savefig(plt, name);
end

function displayLearningStatistics(stats::LearningStatistics, name::String, isClassification::Bool)
    plt = plot(1:length(stats.trainErrors), hcat(stats.trainErrors, stats.valErrors), label = ["Train Error", "Val. Error"], xlabel = "Epochs", ylabel = isClassification ? "Loss" : "MSE")
    savefig(plt, name);
end

function displaySweepGraph(stats::LearningStatistics, name::String, isClassification::Bool, lrSweep::Array{Float64})
    plt = plot(lrSweep, hcat(stats.trainErrors, stats.valErrors), label = ["Train Error", "Val. Error"], xlabel = "Learning Rate", ylabel = isClassification ? "Loss" : "MSE")
    savefig(plt, name);
end

end # module ml plots