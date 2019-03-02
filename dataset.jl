module dataset
import Base.show
import CSV

struct dataItem
    features::Array{Float64,1}
    labels::Array{Float64,1}
end

mutable struct dataSet
    examples::Array{dataItem,1}
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
    data::dataSet = dataSet(Array{dataItem,1}(undef, numItems))
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

"""
Used by generateWeightedSumData to calculate the weighted sum for each window.
`sumRange` is a `UnitRange{Int64}` e.g. `1:10`
"""
function weightedSum(data::Array{Float64,1}, currIndex::Int, sumRange::UnitRange{Int64})
    sum::Float64 = 0.0
    maxIndex = length(data)
    windowSize = length(sumRange)
    for i in sumRange;
        if currIndex + i > 0 && currIndex + i <= maxIndex
            # Make sure the index we're accessing is valid
            sum += data[currIndex + i] * (1 - abs(i) / windowSize)
        end
    end
    return sum
end

"""
An implementation for generating the artificial data used in the original BRNN paper
by Mike Schuster and Kuldip K. Paliwal in 1997. The artificial data is generated as
follows. First, a stream of `numItems` random numbers between zero and one is created
as the one-dimensional (1-D) input data. The 1-D output data is obtained as the
weighted sum of the inputs within a window of `leftWindowSize` frames to the left and
`rightWindowSize` frames to the right with respect to the current frame.
"""
function generateWeightedSumData(numItems::Int, leftWindowSize::Int, rightWindowSize::Int, isClassification::Bool)
    data::dataSet = dataSet(Array{dataItem,1}(undef, numItems))
    randFloats = rand(Float64, numItems)
    # Make sure these args are positive
    leftWindowSize = abs(leftWindowSize)
    rightWindowSize = abs(rightWindowSize)

    for i in 1:numItems;
        leftSum::Float64 = (1 / leftWindowSize) * weightedSum(randFloats, i, -leftWindowSize:-1)
        rightSum::Float64 = (1 / rightWindowSize) * weightedSum(randFloats, i, 0:rightWindowSize - 1)
        label::Float64 = leftSum + rightSum
        if isClassification
            label = label <= 0.5 ? 0.0 : 1.0
        end
        data.examples[i] = dataItem([randFloats[i]], [label])
    end

    return data
end


function getGesturesDataSet()
    return readFromCSVs(["../EMG_data_for_gestures-master/01/1_raw_data_13-12_22.03.16.txt"], ["channel1","channel2","channel3","channel4","channel5","channel6","channel7","channel8"], "class")
end
function readFromCSVs(files::Array{String,1}, validIndicies::Array{String,1}, classIndex::String)
    dataset = dataSet(Array{dataItem,1}(undef, 0))
    nInputFeatures = length(validIndicies)
    validSymbols = Array{Symbol,1}(undef, nInputFeatures)
    classSymbol = Symbol(classIndex)
    for index in 1:nInputFeatures
        validSymbols[index] = Symbol(validIndicies[index])
    end
    for file in files;
        for row in CSV.File(file; delim = '\t')
            data = dataItem(Array{Float64,1}(undef, nInputFeatures), Array{Float64,1}(undef, 1))


            for column in 1:nInputFeatures
                data.features[column] = getproperty(row, validSymbols[column])
            end
            data.labels[1] = getproperty(row, classSymbol)
            print(data)
            push!(dataset.examples, data)
            break;

        end
    end

    return dataset
end


end