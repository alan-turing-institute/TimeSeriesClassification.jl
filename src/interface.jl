using .IntervalBasedForest
const IBF = IntervalBasedForest

###  RandomForestClassifierTS
@mlj_model mutable struct RandomForestClassifierTS <: MMI.Probabilistic # have another abstracttype
    n_trees::Int                      =     200::(_  ≥ 1)
    min_interval::Int                 =       3::(_  ≥ 1)
    pruning_purity_threshold::Float64 =    0.67::(0 ≤ _ ≤ 1)
end

function MMI.fit(m::RandomForestClassifierTS, verbosity::Int, X, y)
    X, y = MMI.matrix(X), array(y)
    tree = IBF.randomforestflassifierFit(X, y,
                                    n_trees=m.n_trees,
                                    min_interval=m.min_interval,
                                    pruning_purity_threshold=m.pruning_purity_threshold)
    fitresult = tree
    cache  = nothing

    return fitresult, nothing, nothing
end

function MMI.predict(m::RandomForestClassifierTS, fitresult, X_new)
    tree = fitresult
    X_new = MMI.matrix(X_new)
    # retrieve the predicted scores
    scores = IBF.predict_new(X_new, tree)
    # smooth if required
    # return vector of UF
    return scores
end
