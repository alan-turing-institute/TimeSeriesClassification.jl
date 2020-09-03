#!/usr/bin/env python3 -u

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

from sklearn.metrics import accuracy_score

import pandas as pd
import os
import numpy as np
import time
from sktime.utils.load_data import load_from_tsfile_to_dataframe

# Load data
data_path = "/home/ucfamml/Documents/Research/data/Univariate_ts/"

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
    "CinCECGTorso",
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


n_datasets = len(datasets)

# Run the fit and predict
for i, dataset in enumerate(datasets):
    print(f'Dataset: {i + 1}/{n_datasets} {dataset}')

    # pre-allocate results
    results = np.zeros(3)

    # load data
    train_file = os.path.join(data_path, f'{dataset}/{dataset}_TRAIN.ts')
    test_file = os.path.join(data_path, f'{dataset}/{dataset}_TEST.ts')

    x_train, y_train = load_from_tsfile_to_dataframe(train_file)
    x_test, y_test = load_from_tsfile_to_dataframe(test_file)

    tsf = KNeighborsTimeSeriesClassifier()

    # fit
    try:
        s = time.time()
        tsf.fit(x_train, y_train)
        results[0] = time.time() - s

        # predict
        s = time.time()
        y_pred = tsf.predict(x_test)
        results[1] = time.time() - s
    
    # catch and raise user exceptions
    except (KeyboardInterrupt, SystemExit):
        raise

    # skip over all other exception
    except:
        print("error - skipping: ", dataset)
        continue

    # score
    results[2] = accuracy_score(y_test, y_pred)

    print('{},{},{},{}\n'.format(dataset, results[0], results[1], results[2]))

    with open("results.txt", "a+") as f:
        f.write(f"{dataset},{results[0]},{results[1]},{results[2]}\n")

