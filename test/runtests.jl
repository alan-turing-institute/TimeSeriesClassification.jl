using MLJTime
using Test

X, y = ts_dataset("Chinatown")
train, test = partition(eachindex(y), 0.7, shuffle=true, rng=123)
rng = StableRNG(566) # seed to resproduce the results. 

@testset "interval based forest" begin
    @test unique(y) == [1.0 , 2.0]
    rng = StableRNG(566)
    model = TimeSeriesForestClassifier(n_trees=3, random_state=rng)
    mach = machine(model, X[train], y[train])
    fit!(mach)
    y_pred = predict(mach, X[test])
    y_pred = map(x -> x.prob_given_ref[1]==1 ? 1 : 2, y_pred)
    @test y_pred  == [2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 
    2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
    2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 
    2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
end
