using BenchmarkTools, MLJTime
using MLJBase: evaluate!

SUITE = BenchmarkGroup()

include("tsf.jl")
