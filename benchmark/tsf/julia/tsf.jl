# ==========
# TSF
# ==========
using BenchmarkTools, MLJTime

datasets = [
    #"ACSF1",
    "Adiac",
    # "AllGestureWiimoteX",
    # "AllGestureWiimoteY",
    # "AllGestureWiimoteZ",
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
    #"CinCECGTorso",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    # "DodgerLoopDay",
    # "DodgerLoopGame",
    # "DodgerLoopWeekend",
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
    # "GestureMidAirD1",
    # "GestureMidAirD2",
    # "GestureMidAirD3",
    # "GesturePebbleZ1",
    # "GesturePebbleZ2",
    "Ham",
    #"HandOutlines",
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
    # "MelbournePedestrian",
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
    # "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    # "PLAID",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    #"Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    # "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    # "StarlightCurves",
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
    "Yoga"
]

DATA_DIR_BENCH = "/Users/aa25desh/Univariate_ts"

@assert DATA_DIR_BENCH != false  "path to the dataset is missing"

io =  open("results.txt", "w")
write(io, "dataset,predict_bench,fit_bench,Accuracy\n")

for dataset in datasets

   Path_train = joinpath(DATA_DIR_BENCH, dataset, string(dataset, "_", uppercase("TRAIN"), ".ts" ))
   Path_test = joinpath(DATA_DIR_BENCH, dataset, string(dataset, "_", uppercase("TEST"), ".ts" ))
 
   test, train =  load_ts_file.([Path_test, Path_train])
   X_test, X_train, y_test, y_train = test[1], train[1], test[2], train[2]
   X_test, X_train = matrix(X_test), matrix(X_train)
   for n_trees in [200]
       model = TimeSeriesForestClassifier(n_trees=n_trees)
       mach = machine(model, X_train, y_train)
       fit!(mach)
       fit_bench = @benchmark fit!($mach, force=true)
       fit_bench = ( fit_bench.times |> mean )*10^-9   #As fit_bench.times is in nano seconds 
       y_pred = predict_mode(mach, X_test)
       predict_bench = @benchmark predict($mach, $X_test)
       predict_bench = ( predict_bench.times |> mean )*10^-9
       Accuracy = accuracy(y_pred, y_test)
       write(io, "$dataset,$predict_bench,$fit_bench,$Accuracy\n")
       print("$dataset,$predict_bench,$fit_bench,$Accuracy\n")
   end
end

close(io)
