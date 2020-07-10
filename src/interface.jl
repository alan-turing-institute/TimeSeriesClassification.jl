using .IntervalBasedForest
const IBF = IntervalBasedForest

###  RandomForestClassifierTS
@mlj_model mutable struct TimeSeriesForestClassifier<: MMI.Probabilistic # have another abstracttype
    n_trees::Int                     =     200::(_  ≥ 1)
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
    X, yplain = MMI.matrix(X), MMI.int(y)
    classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
    integers_seen = MMI.int(classes_seen)
    forest, intervals = IBF.TimeSeriesForestClassifier(m, X, yplain)
    fitresult = (forest, intervals, classes_seen, integers_seen)
    cache  = nothing
    return fitresult, nothing, nothing
end

function MMI.predict(m::TimeSeriesForestClassifier, fitresult, X_new)
    X_new = MMI.matrix(X_new)
    forest, intervals, classes_seen, integers_seen = fitresult
    scores = IBF.predict_new(X_new, forest, intervals, integers_seen)
    sm_scores =  smooth(scores)
    return MMI.UnivariateFinite(classes_seen, sm_scores)
end

function  smooth(X)
    a, b = size(X)
    sm_scores = zeros(a, b)
    indices = map(i -> CartesianIndex(i, findmax(X[i, :])[2]), [1:a;])
    sm_scores[indices] .= 1
    return sm_scores
end
