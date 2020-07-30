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
       RELPREC[dataset]["$n_trees trees"]["cross_entropy"] = evaluate!(mach,
                                                             measure=cross_entropy)
   end

end
