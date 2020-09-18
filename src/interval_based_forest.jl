module IntervalBasedForest

using DecisionTree: apply_tree_proba, build_tree, Node, prune_tree
using Statistics, StableRNGs

"""
   `TimeSeriesForestClassifier(m, Xmatrix::Array, yplain::Array)`

TimeSeriesForestClassifier builds a forest of trees from the training set
`(Xmatrix, yplain)` using random intervals and summary features.
"""
function TimeSeriesForestClassifier(m, Xmatrix::Array, yplain::Array)
    n_trees = m.n_trees
    Interval_features, Intervals = InvFeatureGen(Xmatrix, n_trees, m.min_interval,
                                                 m.random_state)
    forest = Array{Node,1}()
    for i in range(1, stop=n_trees)
        tree = build_tree(yplain, Interval_features[i],
                         m.n_subfeatures,
                         m.max_depth,
                         m.min_samples_leaf,
                         m.min_samples_split,
                         m.min_purity_increase)
        if m.post_prune
            tree = prune_tree(tree, m.merge_purity_threshold)
        end
        push!(forest, tree)
    end
    forest, Intervals
end

"""
   `predict_new(Xmatrix, forest, Intervals::Array, integers_seen)`

Find predictions for all cases in Xmatrix. Built on top of `apply_tree_proba`.
"""
function predict_new(Xmatrix::Array, forest, Intervals::Array, integers_seen)
    n_instance, s_length = size(Xmatrix)
    n_trees = length(forest)
    Interval_features = InvFeatureGen(Xmatrix, Intervals, n_trees)
    sum = zeros(n_instance, length(integers_seen))
    for i=1:n_trees
        sum += apply_tree_proba(forest[i], Interval_features[i], integers_seen)
    end
    return sum/n_trees
end

"""
   `InvFeatureGen(Xmatrix::Array, n_trees::Int, min_interval::Int, random_state)`

InvFeatureGen selects random intervals, generating the mean, standard deviation
and slope of the random intervals resulting 3*√m features.
"""
function InvFeatureGen(Xmatrix::Array, n_trees::Int, min_interval::Int, random_state)
    n_instance, series_length = size(Xmatrix)
    Interval_features = Array{Array{Float64,2},1}()
    n_intervals = floor(Int, sqrt(series_length))
    Intervals = zeros(Int, n_trees, n_intervals, 2)
    for i in range(1, stop = n_trees)
       interval_feature = Array{Float64,2}(undef, 3*n_intervals, n_instance)
       for j in range(1, stop = n_intervals)
           if typeof(random_state) == StableRNGs.LehmerRNG
               Intervals[i,j,1] = rand(random_state, 1:(series_length - min_interval))
               len = rand(random_state, 1:(series_length - Intervals[i,j,1]))
           else
               Intervals[i,j,1] = rand(1:(series_length - min_interval))
               len = rand(1:(series_length - Intervals[i,j,1]))
           end
           if len < min_interval
               len = min_interval
           end
           Intervals[i,j,2] = Intervals[i,j,1] + len
           Y = Xmatrix[:, Intervals[i,j,1]:Intervals[i,j,2]]
           x = Array(1:size(Y)[2])
           means = mean(Y, dims=2)
           stds =  std(Y, dims=2)
           slope = (mean(transpose(x).*Y, dims=2) -
                    mean(x)*mean(Y, dims=2)) / (mean(x.*x) - mean(x)^2)
           interval_feature[3*j-2,:] =  means
           interval_feature[3*j-1,:] =  stds
           interval_feature[3*j,:]   =  slope
       end
           push!(Interval_features, transpose(interval_feature))
    end
    return Interval_features, Intervals
end

function InvFeatureGen(Xmatrix::Array, Intervals::Array, n_trees::Int)
    n_instance, series_length = size(Xmatrix)
    Interval_features = Array{Array{Float64,2},1}()
    n_intervals = floor(Int, sqrt(series_length))
    for i in range(1, stop = n_trees)
       interval_feature = Array{Float64,2}(undef, 3*n_intervals, n_instance)
       for j in range(1, stop = n_intervals)
           Y = Xmatrix[:, Intervals[i,j,1]:Intervals[i,j,2]]
           x = Array(1:size(Y)[2])
           means = mean(Y, dims=2)
           stds =  std(Y, dims=2)
           slope = (mean(transpose(x).*Y, dims=2) -
                    mean(x)*mean(Y, dims=2)) / (mean(x.*x) - mean(x)^2)
           interval_feature[3*j-2,:] =  means
           interval_feature[3*j-1,:] =  stds
           interval_feature[3*j,:]   =  slope
       end
           push!(Interval_features, transpose(interval_feature))
    end
    return Interval_features
end

end # TimeSeriesForestClassifier
