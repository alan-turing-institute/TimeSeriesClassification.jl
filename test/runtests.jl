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
    @test accuracy(y_pred, y[test]) >= 0.75    

    Interval_features, Intervals = MLJTime.IntervalBasedForest.InvFeatureGen(matrix(X[1:5]), 1, 3, 0);

    @test Interval_features  == [[762.0 419.7337251162933 -222.9714285714286 989.375 309.6476972763909 -107.32142857142857 482.3333333333333 530.5373490979489 127.67832167832174 1120.2 132.37711282544277 -45.00000000000023;
    650.3333333333334 349.0350507709314 -185.71428571428586 851.5 269.92485726851567 -96.85714285714286 503.1666666666667 417.6871019693033 65.44755244755245 961.6 148.43449733805144 -65.20000000000005; 
    1356.0 314.96158495918195 -161.02857142857155 1409.75 182.17162237845938 -4.809523809523809 493.3333333333333 568.8278751214215 139.86713286713294 1497.0 172.91182724151636 64.90000000000009; 
    899.0 457.84713606180827 -242.51428571428582 1235.125 404.771694909612 -159.48809523809524 731.5833333333334 823.9466507810644 202.69580419580421 1359.6 187.18520240660052 -113.69999999999982; 
    928.6666666666666 143.0967038986806 -65.142857142857 947.875 89.62531131645473 2.4404761904761907 370.1666666666667 442.7063121165981 105.18881118881121 953.4 99.99149963871929 56.80000000000018]]
end

@testset "distances" begin
    rng = StableRNG(566)
    a = rand(rng, Int64, 5)
    b = rand(rng, Int64, 5)
    @test dtw_distance(a, b, -1) == -2.9402683278543684e19
end 

@testset "KNN" begin
    X, y = ts_dataset("Chinatown")
    train, test = partition(eachindex(y), 0.7)
    model = TimeSeriesKNNClassifier()
    mach = machine(model, X[train], y[train])
    fit!(mach)
    y_pred = predict_mode(mach, X[test])
    @test accuracy(y_pred, y[test]) >= 0.70    
end
