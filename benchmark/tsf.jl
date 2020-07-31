# ==========
# TSF
# ==========

SUITE["tsf"] = BenchmarkGroup()
RESULTS = SUITE["tsf"]
RELPREC = Dict()

datasets = ["Adiac", "Chinatown"]

for dataset in datasets
   RESULTS[dataset] = BenchmarkGroup()
   RELPREC[dataset] = Dict()

   X, y = ts_dataset(dataset)
   train, test = partition(eachindex(y), 0.7, shuffle=true, rng=123)
   rng = StableRNG(566)

   for n_trees in [3, 5]
       RESULTS[dataset]["$n_trees trees"] = BenchmarkGroup()
       RELPREC[dataset]["$n_trees trees"] = Dict()
       model = TimeSeriesForestClassifier(n_trees=n_trees, random_state=rng)
       mach = machine(model, X[train], y[train])
       RESULTS[dataset]["$n_trees trees"]["fit"] = @benchmarkable fit!($mach, force=true)
       fit!(mach, force=true)
       RESULTS[dataset]["$n_trees trees"]["predict"] = @benchmarkable predict($mach, $X[$test])
       y_pred = predict_mode(mach, X[test])
       RELPREC[dataset]["$n_trees trees"]["Accuracy"] = accuracy(y_pred, y[test])
   end

end
