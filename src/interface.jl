using .IntervalBasedForest
const IBF = IntervalBasedForest

"""
TimeSeriesForestClassifier(; kwargs...)

Overview: Input n series length m
for each tree
    sample sqrt(m) intervals
    find mean, sd and slope for each interval, concatenate to form new
    data set
    build decision tree on new data set
ensemble the trees with averaged probability estimates

## Hyperparameters

- `n_trees=200`            ensemble size(number of trees)

- `random_state`           seed for random numbers

- `min_interval=3`         minimum width of an interval

- `max_depth=-1`:          max depth of the decision tree (-1=any)

- `min_samples_leaf=1`:    max number of samples each leaf needs to have

- `min_samples_split=2`:   min number of samples needed for a split

- `min_purity_increase=0`: min purity needed for a split

- `n_subfeatures=0`: number of features to select at random (0 for all,
  -1 for square root of number of features)

- `post_prune=false`:      set to `true` for post-fit pruning

- `merge_purity_threshold=1.0`:  (post-pruning) merge leaves having `>=thresh`
                           combined purity

- `pdf_smoothing=0.0`:     threshold for smoothing the predicted scores

- `display_depth=5`:       max depth to show when displaying the tree

"""
@mlj_model mutable struct TimeSeriesForestClassifier<: MMI.Probabilistic # have another abstracttype?
    n_trees::Int                     =     200::(_  ≥ 1)
    random_state                     =  nothing
    min_interval::Int                =       3::(_  ≥ 1)
    max_depth::Int                   =  (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int            =       1::(_ ≥ 0)
    min_samples_split::Int           =       2::(_ ≥ 2)
    min_purity_increase::Float64     =     0.0::(_ ≥ 0)
    n_subfeatures::Int               =       0::(_ ≥ -1)
    post_prune::Bool                 =   false
    merge_purity_threshold::Float64  =     1.0::(_ ≤ 1)
    pdf_smoothing::Float64           =     0.0::(0 ≤ _ ≤ 1)
    display_depth::Int               =       5::(_ ≥ 1)
end

function MMI.fit(m::TimeSeriesForestClassifier, verbosity::Int, X, y)
    Xmatrix, yplain = MMI.matrix(X), MMI.int(y)
    classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
    integers_seen = MMI.int(classes_seen)
    forest, intervals = IBF.TimeSeriesForestClassifier(m, Xmatrix, yplain)
    fitresult = (forest, intervals, classes_seen, integers_seen)
    cache  = nothing
    return fitresult, nothing, nothing
end

function MMI.predict(m::TimeSeriesForestClassifier, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    forest, intervals, classes_seen, integers_seen = fitresult
    scores = IBF.predict_new(Xmatrix, forest, intervals, integers_seen)
    sm_scores = smooth(scores)
    return [MMI.UnivariateFinite(classes_seen, sm_scores[i, :])
                   for i in 1:size(sm_scores, 1)]
end

function smooth(X)
    a, b = size(X)
    sm_scores = zeros(a, b)
    indices = map(i -> CartesianIndex(i, findmax(X[i, :])[2]), [1:a;])
    sm_scores[indices] .= 1
    return sm_scores
end

MMI.fitted_params(::TimeSeriesForestClassifier, fitresult) =
    (forest=fitresult[1])

#=
 NOTE: Should we add pretty printing?
=#

"""
An adapted version of the NearestNeighbors knn to work with
time series data.       

## Hyperparameters

 -  `n_neighbors`            Int, set k for knn (default = 5)
 -  `weights`                mechanism for weighting a vote 'uniform', 'distance'
 -  `search_algorithm`       search method for neighbours default = "select_sort"
 -  `metric`                 distance measure for time series default = "dtw_distance"
 -  `metric_params`          dictionary for metric parameters

"""
@mlj_model mutable struct TimeSeriesKNNClassifier <: MMI.Probabilistic
    n_neighbors::Int                      =   5::(_  ≥ 1)
    weights::Symbol                       =   :uniform
    search_algorithm::Symbol              =   :select_sort
    metric::Symbol                        =   :dtw_distance
    metric_params::Array                  =   [-1.0]
end

function MMI.fit(m::TimeSeriesKNNClassifier, verbosity::Int, X, y)
    Xmatrix, yplain = MMI.matrix(X), MMI.int(y)
    classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
    integers_seen = MMI.int(classes_seen)
    fitresult = (Xmatrix, yplain, classes_seen, integers_seen)
    cache  = nothing
    return fitresult, nothing, nothing
end

function MMI.predict(m::TimeSeriesKNNClassifier, fitresult, Xnew)
    Xmatrix_new = MMI.matrix(Xnew)
    Xmatrix, yplain, classes_seen, integers_seen = fitresult
    y_pred, DistanceMatrix = Predict_new(m, Xmatrix, Xmatrix_new, yplain) 
    a, b = length(y_pred), length(integers_seen)
    probas = zeros(a, b)
    for i=1:a
        for j=1:b
            probas[i,j] = Int(y_pred[i]) == j ? 1 : 0
        end
    end
    return  [MMI.UnivariateFinite(classes_seen, probas[i, :])
                for i in 1:size(probas, 1)]
end

# MMI.fitted_params(::TimeSeriesKNNClassifier, fitresult) 