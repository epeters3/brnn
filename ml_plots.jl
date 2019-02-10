module ml_plots
using brnn: learningStatistics
using Plots


function displayLayerStatistics(stats::learningStatistics, name::String)
    plt = plot(1:length(stats.averageWeightChange), stats.averageWeightChange)
    savefig(plt, name);
end


end # module ml plots