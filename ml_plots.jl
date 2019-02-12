module ml_plots
using brnn: layerStatistics, learningStatistics
using Plots


function displayLayerStatistics(stats::layerStatistics, name::String)
    plt = plot(1:length(stats.averageWeightChange), stats.averageWeightChange)
    savefig(plt, name);
end

function displayLearningStatistics(stats::learningStatistics, name::String)
    plt = plot(1:length(stats.trainErrors), hcat(stats.trainErrors, stats.valErrors), label = ["Train Error", "Val. Error"], xlabel = "Epochs", ylabel = "MSE")
    savefig(plt, name);
end

end # module ml plots