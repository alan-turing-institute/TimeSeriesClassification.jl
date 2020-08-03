# ==========
# TSF
# ==========
using BenchmarkTools, MLJTime

datasets = ["Adiac", "Chinatown"]

io =  open("benchmark/results.txt", "w")
write(io, "dataset,predict_bench,fit_bench,Accuracy\n")
for dataset in datasets
   X, y = ts_dataset(dataset)
   train, test = partition(eachindex(y), 0.7, shuffle=true, rng=123)
   for n_trees in [200]
       model = TimeSeriesForestClassifier(n_trees=n_trees)
       mach = machine(model, X[train], y[train])
       fit!(mach)
       fit_bench = @benchmark fit!($mach, force=true)
       fit_bench = ( fit_bench.times |> median )*10^-9   #As fit_bench.times is in nano seconds 
       y_pred = predict_mode(mach, X[test])
       predict_bench = @benchmark predict($mach, $X[$test])
       predict_bench = ( predict_bench.times |> median )*10^-9
       Accuracy = accuracy(y_pred, y[test])
       write(io, "$dataset,$predict_bench,$fit_bench,$Accuracy\n")
   end
end

close(io)

