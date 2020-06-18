module MLJTime

# IMPORTS
using IndexedTables, CSVFiles, ZipFile

# from Standard Library:
using Statistics, DecisionTree

# EXPORTS
export RandomForestClassifierTS, InvFeatureGen,
       predict_single, InvFeatures, predict_new,
       X_y_split
export load_gunpoint, TSdataset, univariate_datasets

#CONSTANTS
# the directory containing this file: (.../src/)
const MODULE_DIR = dirname(@__FILE__)

# Includes
include("IntervalBasedForest.jl")
include("datapath.jl")

end # module
