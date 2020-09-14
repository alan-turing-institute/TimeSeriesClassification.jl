using MLJTime
using Test

@testset "interval based forest" begin
    X, y = ts_dataset("Chinatown")
    rng = StableRNG(566) # seed to resproduce the results. 
    
    Interval_features, Intervals = MLJTime.IntervalBasedForest.InvFeatureGen(matrix(X[1:5]), 1, 3, rng);

    @test Interval_features  == [[745.4444444444445 505.0103344571245 56.36945304437562 716.3157894736842 506.93973481750487 39.17543859649122 1057.6 311.9797714240104 -88.88484848484839 1197.5 66.31490531295862 13.4; 
    643.6111111111111 413.3816266853738 45.493292053663545 619.578947368421 415.16761217468587 31.45964912280703 904.5 265.43057263414266 -75.93333333333331 1046.5 36.27211968808366 9.4;
    944.1666666666666 620.4238826138522 100.38906088751294 939.1052631578947 603.3471356374133 83.81228070175439 1421.2 164.37342310185736 -7.357575757575845 1406.75 169.44689433565904 119.1;
    1023.6666666666666 700.7234916603737 62.30340557275547 982.8421052631579 703.8473842750836 40.710526315789444 1366.3 455.57339937953157 -146.3454545454545 1541.5 139.28507936363224 -102.0;
    659.2222222222222 434.21268244349017 67.04643962848297 660.3157894736842 422.00579941138494 57.31754385964911 982.8 108.05327903914397 -15.757575757575758 922.25 88.08092869628476 7.5]]
    
    train, test = partition(eachindex(y), 0.7)
    @test unique(y) == [1.0 , 2.0]
    
    model = TimeSeriesForestClassifier(n_trees=3, random_state=rng)
    mach = machine(model, X[train], y[train])
    fit!(mach)
    y_pred = predict_mode(mach, X[test])
    @test accuracy(y_pred, y[test]) >= 0.75    
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
    
    A = rand(StableRNG(566), 5, 5)
    index = MLJTime.select_sort(A, 3)
    @test index == [1.0 2.0 3.0; 1.0 3.0 4.0; 3.0 2.0 5.0; 1.0 5.0 4.0; 5.0 2.0 4.0]
    
    model = TimeSeriesKNNClassifier()
    mach = machine(model, X[train], y[train])
    fit!(mach)
    y_pred = predict_mode(mach, X[test])
    @test accuracy(y_pred, y[test]) >= 0.70    
end
