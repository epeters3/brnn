module utils

#=
Downsamples an array so it only contain `maxLength` items.
Evenly downsamples across the array so the sample is representative.
=#
function downSample(list::Array, maxLength::Int64)::Array
    if length(list) <= maxLength
        return list
    else
        sampleStepSize = round(Int64, length(list) / maxLength)
        sampledList = list[1:sampleStepSize:end]
        return sampledList[1:min(maxLength, length(sampledList))]
    end
end

end # module utils