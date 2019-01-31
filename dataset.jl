module dataset
import Base.show

struct dataItem
    features::Array{Float64}
    labels::Array{Float64}
end

mutable struct dataSet
    examples::Array{dataItem}
end

function Base.show(io::IO, item::dataItem)
    print("Data Item: features: $(item.features), labels: $(item.labels)")
end

function generateData(numItems::Int, inputDims::Int, outputDims::Int)
    data::dataSet = dataSet(Array{dataItem}(undef, numItems))
    for i in 1:numItems;
        data.examples[i] = dataItem([rand(),rand() * i,rand() * i * i], [rand() / 5])
    end
    return data
end

end