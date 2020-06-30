module IntervalBasedForest

using DecisionTree, Statistics

function randomforestflassifierFit(X, y; n_trees::Int=200, min_interval::Int=3,
                                  pruning_purity_threshold::Float64=0.67)
    transform_xt = InvFeatureGen(X, n_trees=n_trees, min_interval=min_interval)
    model = DecisionTreeClassifier(pruning_purity_threshold=pruning_purity_threshold)
    forest = Array{DecisionTreeClassifier,1}()
    for i in range(1, stop=n_trees)
        mdl = deepcopy(model)
        fit!(mdl, transform_xt[i], y)
        push!(forest, mdl)
    end
    forest
end

function InvFeatureGen(X; n_trees::Int=200, min_interval::Int=3)
    n_samps, series_length = size(X)
    transform_xt = Array{Array{Float64,2},1}()
    n_intervals = floor(Int, sqrt(series_length))
    intervals = zeros(Int, n_trees, 3*n_intervals, 2)
    for i in range(1, stop = n_trees)
       transformed_x = Array{Float64,2}(undef, 3*n_intervals, n_samps)
       for j in range(1, stop = n_intervals)
           intervals[i,j,1] = rand(1:(series_length - min_interval))
           len = rand(1:(series_length - intervals[i,j,1]))
           if len < min_interval
               len = min_interval
           end
           intervals[i,j,2] = intervals[i,j,1] + len
           x = Array(1:len+1)
           Y = X[:, intervals[i,j,1]:intervals[i,j,2]]
           means = mean(Y, dims=2)
           stds =  std(Y, dims=2)
           slope = (mean(transpose(x).*Y, dims=2) -
                    mean(x)*mean(Y, dims=2)) / (mean(x.*x) - mean(x)^2)
           transformed_x[3*j-2,:] =  means
           transformed_x[3*j-1,:] =  stds
           transformed_x[3*j,:]   =  slope
       end
           push!(transform_xt, transpose(transformed_x))
    end
    return transform_xt
end



function predict_single(forest, features)
    vv = zeros(Float64,2)
    for i in range(1, stop=200)
        vv += collect(proba_predict(forest,  (features[i])))
    end
    return vv
end

function proba_predict(forest, X)
    a = 0
    for tree in forest
        if predict(tree, X) == 1
            a += 1
        end
    end
    (a, length(forest)-a)
end

function InvFeatures(X, n_trees::Int=200, min_interval::Int=3)
    transform_xt = []
    series_length, = size(X)
    n_intervals = floor(Int, sqrt(series_length))
    intervals = zeros(Int, n_intervals, 2)
    for i in range(1, stop = n_trees)
        transformed_x = Array{Float64,1}(undef, 3*n_intervals)
        for j in range(1, stop = n_intervals)
            intervals[j,1] = rand(1:(series_length - min_interval))
            len = rand(1:(series_length - intervals[j,1]))
            if len < min_interval
                len = min_interval
            end
            intervals[j,2] = intervals[j,1] + len
            x = Array(1:len+1)
            Y = X[intervals[j,1]:intervals[j,2]]
            means = mean(Y)
            stds  = std(Y)
            slope = (mean(x.*Y) -
                     mean(x)*mean(Y)) / (mean(x.*x) - mean(x)^2)
            transformed_x[3*j-2] = means
            transformed_x[3*j-1] = stds
            transformed_x[3*j] = slope
        end
        push!(transform_xt, vcat(transformed_x))
    end
    return transform_xt
end

function predict_new(X1, forest)
    t_stamp = length(X1[1, :])
    c = Array{Float64}(undef, t_stamp)
    for i=1:t_stamp
        y_hat = InvFeatures(X1[i,:])
        a, b = predict_single(forest, y_hat)
        c[i] = a > b ? 1 : 2
    end
    return c
end

end # randomforestflassifierFit
