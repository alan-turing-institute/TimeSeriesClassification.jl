import MLJModelInterface
<<<<<<< HEAD
using .IntervalBasedForest
const IBF = IntervalBasedForest


###  RandomForestClassifierTS
MMI.@mlj_model mutable struct RandomForestClassifierTS <: MMI.Probabilistic # have another abstracttype
=======
import MLJModelInterface: @mlj_model
using .IntervalBasedForest
const IBF = IntervalBasedForest
const MMI = MLJModelInterface


###  RandomForestClassifierTS
@mlj_model mutable struct RandomForestClassifierTS <: MMI.Probabilistic # have another abstracttype
>>>>>>> 1ee676afa6699fcb954204a70d00527fda2affad
    n_trees::Int                      =     200::(_  ≥ 1)
    min_interval::Int                 =       3::(_  ≥ 1)
    pruning_purity_threshold::Float64 =    0.67::(0 ≤ _ ≤ 1)
end

function MMI.fit(m::RandomForestClassifierTS, X, y)

    tree = IBF.randomforestflassifierFit(X, y,
                                    n_trees=m.n_trees,
                                    min_interval=m.min_interval,
                                    pruning_purity_threshold=m.pruning_purity_threshold)
    fitresult = tree
    cache  = nothing

    return fitresult
end

function MMI.predict(m::RandomForestClassifierTS, fitresult, X_new)
    tree = fitresult
    # retrieve the predicted scores
    scores = IBF.predict_new(X_new, tree)
    # smooth if required
    # return vector of UF
    return scores
end
