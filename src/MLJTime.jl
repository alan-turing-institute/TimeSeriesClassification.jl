module MLJTime

# IMPORTS
using IndexedTables, CSVFiles, ZipFile

# from Standard Library:
using Statistics, DecisionTree

# EXPORTS
export RandomForestClassifierTS, InvFeatureGen,
       predict_single, InvFeatures, predict_new

export load_gunpoint, TSdataset

#CONSTANTS
# the directory containing this file: (.../src/)
const MODULE_DIR = dirname(@__FILE__)

# Includes
include("IntervalBasedForest.jl")
include("datapath.jl")

end # module
