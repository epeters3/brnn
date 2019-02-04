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

"""
Generates a dataset whose features are a single time series input
of random bits. The output label is the parity of n arbitrarily delayed
(but consistent) previous inputs. For example, for `generateDparityData(10, [0,-2,-5])`,
the label of each instance would be set to the parity of the current input, the
input 2 steps back, and the input 5 steps back. Note: When there is not enough data
at the beginning or end of the set to compute the parity using parityIndices (calculated
in a naiive way), the label value is 0.0.
"""
function generateDparityData(numItems::Int, parityIndices::Array{Int})
    data::dataSet = dataSet(Array{dataItem}(undef, numItems))
    maxParityIndex = maximum(abs.(parityIndices))
    randBits = rand(0.0:1.0, numItems)

    for i in 1:numItems;
        label::Float64 = 0.0
        if maxParityIndex < i && i + maxParityIndex <= numItems
            # We have enough training examples to find the full parity.
            # Make the label be the parity of the numbers found at parityIndices
            for parityIndex in parityIndices;
                label += randBits[i + parityIndex]
            end
            label = label % 2 == 0 ? 0 : 1
        end
        data.examples[i] = dataItem([randBits[i]], [label])
    end

    return data
end

end