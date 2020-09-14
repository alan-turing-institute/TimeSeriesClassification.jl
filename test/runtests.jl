using MLJTime
using Test

@testset "interval based forest" begin
    X, y = ts_dataset("Chinatown")
    train, test = partition(eachindex(y), 0.7)
    @test unique(y) == [1.0 , 2.0]
    rng = StableRNG(566) # seed to resproduce the results. 
    model = TimeSeriesForestClassifier(n_trees=3, random_state=rng)
    mach = machine(model, X[train], y[train])
    fit!(mach)
    y_pred = predict_mode(mach, X[test])
    @test accuracy(y_pred, y[test]) > 0.75    
end

@testset "distances" begin
    rng = StableRNG(566)
    a = rand(rng, Int64, 5)
    b = rand(rng, Int64, 5)
    @test dtw_distance(a, b, -1) == -2.9402683278543684e19
end 

