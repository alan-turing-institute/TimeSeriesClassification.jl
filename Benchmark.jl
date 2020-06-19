using MLJTime

const univariate_data = univariate_datasets

TSdataset(univariate_data)


function Benchmarke_all(algorithms, datasets, train_test_split)
    Dict_measures = Dict{Pair{String,String}, Integer}()
    for table in datasets
        for algo in algorithms
            for cv in train_test_split

            end
        end
    end
end

function (args)
    body
end
