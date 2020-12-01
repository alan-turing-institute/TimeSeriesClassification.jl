
const DATA_DIR = joinpath(MODULE_DIR, "..", "data")
using CategoricalArrays

function MMI.matrix(table)
    cols = columns(table)
    a, b = length(cols[1]), length(cols)
    matrix = Matrix{Float64}(undef, a, b)
    @inbounds for i=1:b
        matrix[:, i] = cols[i]
    end
    return matrix
end

"""
load_dataset(fpath)

Load one of standard dataset like Chinatown etc assuming the file is a
comma separated file.

"""
function load_dataset(fname::String)
    fpath = joinpath(DATA_DIR, fname)
    data_raw = load(fpath, header_exists=false)
    Xmatrix = matrix(table(data_raw)) #check if we can use data_raw directly for y
    width = size(Xmatrix)[2]
    #return table(view(Xmatrix, :, 1:(width-1))), CategoricalArray(view(Xmatrix, :, width))   moving away from IndexedTable
    return Xmatrix[:, 1:(width-1)], CategoricalArray(view(Xmatrix, :, width))
end

#NOTE: permutedims(cat(mat, mat2, dims=3),[1,3,2]) For Motions, eg for multi-dimensional dataset.

"""
   `load_ts_file(fpath; return_array=false)`

`load_ts_file` takes the path to `ts` files and returns the table (IndexedTable),
where `fpath` is path to `ts` file located on your computer.
on `return_array=true` reurns an Array.
eg
```julia
   tableT = load_ts_file("/Users/your_user_name/../Adiac_TRAIN.ts")
   arr = load_ts_file("/Users/your_user_name/../Adiac_TRAIN.ts", return_array=true)
```
"""
function load_ts_file(fpath; return_array=false)
#`readuntil` and `readlines` are used in combination so that we read data
# required only for transformation into julia array and table.
    data = open(fpath) do input
          readuntil(input, "@data")
          readlines(input)
    end
    data = data[2:end] # removes  the empty string.
    # split the string and convert each element into Float64.
    arrays = map(i -> parse.(Float64, split(i, r"[:,]")) , data)
    Xmatrix = transpose(hcat(arrays...)) # Creates 2D Array.
    if return_array == true
        return Xmatrix
    else
        return table(eachcol(Xmatrix[:, 1:end-1])...), CategoricalArray(Xmatrix[:, end])
    end
end

"""
   `ts_dataset(dataset; split=false)`
`ts_dataset` takes two inputes, `dataset` is the name of datasets available in the
`data` folder & `split` specifile the train or test part.
Reurns dataset and target variable.
eg
```julia
   X, y = ts_dataset("Adiac")
   X_test, y_test = ts_dataset("Adiac", split="test")
```
"""
function ts_dataset(dataset::String; split=false)
    if split in ["test", "train"]
         fpath = joinpath(DATA_DIR, dataset, string(dataset, "_", uppercase(split), ".ts" ))
         return load_ts_file(fpath)
    elseif split == false
         v = []
         for i in ["test", "train"]
             fpath = joinpath(DATA_DIR, dataset, string(dataset, "_", uppercase(i), ".ts" ))
             push!(v, fpath)
         end
         test, train =  load_ts_file.(v)
         return merge(test[1], train[1]), vcat(test[2], train[2])
    else
        throw(ArgumentError("Invalid `split` value: $split"))
    end
end

"""
   `load_from_tsfile_to_NDArray(fpath)`

`load_from_tsfile_to_NDArray` takes the path to `ts` files and reurns an Array.
eg
```julia
   tableT = load_from_tsfile_to_NDArray("/Users/your_user_name/../BasicMotions_TRAIN.ts")
   arr = load_from_tsfile_to_NDArray("/Users/your_user_name/../BasicMotions_TRAIN.ts")
```
"""
function load_from_tsfile_to_NDArray(fpath)
    data = open(fpath) do input
        readuntil(input, "@data")
        readlines(input)
    end
    data = split.(data[2:end], r"[:]")
    instance = length(data)
    dimensions  = length(data[1]) - 1
    series_dim1 = split(data[1][1], r"[,]")
    serieslength = length(series_dim1)
    NDMatrix = zeros(instance, serieslength, dimensions)
    data = cat(data..., dims=3)
    @inbounds for i=1:instance
        @inbounds for j=1:dimensions
            NDMatrix[i,:,j] = parse.(Float64, split(data[j,1,i], r"[,]"))
        end
    end
    return NDMatrix, categorical(data[dimensions+1,1,:])
end

"""
   `load_NDdataset(dataset::String; split=false)`
`load_NDdataset` takes two inputes, `dataset` is the name of datasets available in the
`data` folder & `split` specifile the train or test part.
Reurns dataset and target variable.
eg
```julia
   X, y = ts_dataset("BasicMotions")
   X_test, y_test = ts_dataset("BasicMotions", split="test")
```
"""
function load_NDdataset(dataset::String; split=false)
    if split in ["test", "train"]
        fpath = joinpath(DATA_DIR, dataset, string(dataset, "_", uppercase(split), ".ts" ))
        X, y = load_from_tsfile_to_NDArray(fpath)
        return X, y
    elseif split == false
        v = []
        for i in ["test", "train"]
            fpath = joinpath(DATA_DIR, dataset, string(dataset, "_", uppercase(i), ".ts" ))
            push!(v, fpath)
        end
        test, train =  load_from_tsfile_to_NDArray.(v)
        return vcat(test[1], train[1]), vcat(test[2], train[2])
    else
        throw(ArgumentError("Invalid `split` value: $split"))
    end
end

univariate_datasets = [
    "ACSF1",
    "Adiac",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "ArrowHead",
    "Coffee",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MelbournePedestrian",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarlightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]
