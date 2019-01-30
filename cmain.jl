#=
In order to find Modules that are not in the standard LOAD_PATH and
be able to import them, we need to update the LOAD_PATH variable for
the current folder explicitly. Then we will be able to import a
local module appropriately.
Source: https://stackoverflow.com/questions/51824403/import-modules-and-functions-from-a-file-in-a-specific-directory-in-julia-1-0
=#
push!(LOAD_PATH, "./")
import main: main

# Here is the main body of the main module. Normal modules
# Do not need to be wrapped in Base.@ccallable: just this main module,
# Because PackageCompiler compiles it down.
Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main()
    return 0
end