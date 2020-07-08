using .IntervalBasedForest
const IBF = IntervalBasedForest

###  RandomForestClassifierTS
@mlj_model mutable struct TimeSeriesForestClassifier<: MMI.Probabilistic # have another abstracttype
    n_trees::Int                      =     200::(_  ≥ 1)
    min_interval::Int                 =       3::(_  ≥ 1)
    pruning_purity_threshold::Float64 =     0.67::(0 ≤ _ ≤ 1)
end

function MMI.fit(m::TimeSeriesForestClassifier, verbosity::Int, X, y)
    X, yplain = MMI.matrix(X), MMI.int(y)
    classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
    integers_seen = MMI.int(classes_seen)
    forest = IBF.TimeSeriesForestClassifier(X, yplain,
                                    n_trees=m.n_trees,
                                    min_interval=m.min_interval,
                                    pruning_purity_threshold=m.pruning_purity_threshold)
    fitresult = (forest, classes_seen, integers_seen)
    cache  = nothing

    return fitresult, nothing, nothing
end

function MMI.predict(m::TimeSeriesForestClassifier, fitresult, X_new)
    X_new = MMI.matrix(X_new)
    forest, classes_seen, integers_seen = fitresult
    # retrieve the predicted scores
    scores = IBF.predict_new(X_new, forest, integers_seen)
    # smooth if required
    # return vector of UF
    return MMI.UnivariateFinite(classes_seen, scores)
end
