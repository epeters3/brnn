# Implementation of Bidirectional Recurrent Neural Network (BRNN)

Authors: Evan Peterson & Tim Whiting

## Running The Program

From the terminal, with Julia >= 1.0.0 installed and added to `PATH`, run:

```shell
julia main.jl
```

## Developing

For a good dev experience, install the `Revise` julia package, then run it inside the julia REPL before running the project:

```julia
using Revise
include("main.jl")
```

The `Revise` package will watch for changes in the `main.jl` module and recompile only the submodules that have changed, saving lots of compile time while developing.
