
const DATA_DIR = joinpath(MODULE_DIR, "..", "data")

function MMI.matrix(table)
    cols = columns(table)
    n, p = length(cols[1]), length(cols)
    matrix = Matrix{Float64}(undef, n, p)
    for i=1:p
        matrix[:, i] = cols[i]
    end
    return matrix
end

function load_dataset(fname::String)
    fpath = joinpath(DATA_DIR, fname)
    data_raw = load(fpath, header_exists=false)
    Table = table(data_raw)
    data_table = matrix(Table)
    return data_table
end

function X_y_split(matrix::Array)
    l_index = length(matrix[1,:])
    return matrix[:, 1:l_index-1], matrix[:, l_index]
end


load_gunpoint() = load_dataset.(["GunPoint/train.csv", "GunPoint/test.csv"])

#NOTE: permutedims(cat(mat, mat2, dims=3),[1,3,2]) For Motions, eg for multi-dimensional dataset.

"""
    `TSdataset(dataset::Array)`
TSdataset takes subset of the `univariate_datasets` avilabe on the `timeseriesclassification`
website and adds `csv` files to the `data` folder after conversion (`ts` -> `csv`).
eg.
```julia
   TSdataset(["ACSF1", "Adiac"])
```
"""
function TSdataset(dataset::Array) #check on win
    for Dir in dataset
        exdir = string("data/", Dir)
        _link = string("http://timeseriesclassification.com/Downloads/", Dir ,".zip")
        Base.download(_link, Dir)
        fileFullPath = isabspath(Dir) ?  Dir : joinpath(pwd(), Dir)
        basePath = dirname(fileFullPath)
        outPath = (exdir == "" ? basePath : (isabspath(exdir) ? exdir : joinpath(pwd(),exdir)))
        isdir(outPath) ? "" : mkdir(outPath)
        zarchive = ZipFile.Reader(fileFullPath)
        for f in zarchive.files
            if f.name[end-1:end] == "ts"
                f.name = string(f.name[1:end-2], "csv")
                fullFilePath = joinpath(outPath,f.name)
                write(fullFilePath, read(f))
                open(fullFilePath) do input
                    readuntil(input, "@data")
                    write(fullFilePath, read(input))
                end
                run(`sed -i'.original' 's/:/,/g' $fullFilePath`)
                fullFilePath1 = string(fullFilePath, ".original")
                run(`rm $fullFilePath1`)
            end
        end
        close(zarchive)
        run(`rm $fileFullPath`)
    end
end

function TSdataset(filepath::String)
    exdir = string("data/", basename(filepath))
    fileFullPath = isabspath(filepath) ?  filepath : joinpath(pwd(), filepath)
    basePath = dirname(fileFullPath)
    outPath = (exdir == "" ? basePath : (isabspath(exdir) ? exdir : joinpath(pwd(),exdir)))
    isdir(outPath) ? "" : mkdir(outPath)
    files = readdir(fileFullPath, join=true)
    for f in files
        if f[end-1:end] == "ts"
            fullFilePath = joinpath(outPath, string(f[1:end-2], "csv"))
            write(fullFilePath, read(f))
            open(fullFilePath) do input
                readuntil(input, "@data")
                write(fullFilePath, read(input))
            end
            run(`sed -i'.original' 's/:/,/g' $fullFilePath`)
            fullFilePath1 = string(fullFilePath, ".original")
            run(`rm $fullFilePath1`)
        end
    end
end

"""
   `load_ts_file(fpath)`
`load_ts_file` takes the path to `ts` files and returns the table (IndexedTable)
and Array, where `fpath` is path to `ts` file located on your computer.
eg
```julia
   table_a, arr = load_ts_file("/Users/your_user_name/../GunPoint_TRAIN.ts")
```
"""
function load_ts_file(fpath)
#`readuntil` and `readlines` are used in combination so that we read data
# required only for transformation into julia array and table.
    data = open(fpath) do input
          readuntil(input, "@data")
          readlines(input)
    end
    data = data[2:end] # removes  the empty string.
    # split the string and convert each element into Float64.
    arrays = map(i -> parse.(Float64, split(i, r"[:,]")) , data)
    array = transpose(hcat(arrays...))*1 # Creates 2D Array.
    return table(eachcol(array[:, 1:end-1])...), table(array[:, end])
end

"""
   `ts_dataset(dataset, test_or_train)`
`ts_dataset` takes two inputes, `dataset` is the name of datasets available in the
`data` folder & `test_or_train` specifile the train or test part.
Reurns dataset and target variable.
eg
```julia
   X, y = ts_dataset("Adiac", nothing)
```
"""
function ts_dataset(dataset::String, split=nothing)
    if split in ["test", "train"]
         fpath = joinpath(DATA_DIR, dataset, string(dataset, "_", uppercase(split), ".ts" ))
         return load_ts_file(fpath)
    elseif split == nothing
         v = []
         for i in ["test", "train"]
             fpath = joinpath(DATA_DIR, dataset, string(dataset, "_", uppercase(i), ".ts" ))
             push!(v, fpath)
         end
         test, train =  load_ts_file.(v)
         return merge(test[1], train[1]), merge(test[2], train[2])
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
