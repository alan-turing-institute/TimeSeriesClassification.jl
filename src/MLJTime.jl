module MLJTime

# IMPORTS
using IndexedTables, CSVFiles, ZipFile

# EXPORTS
export RandomForestClassifierFit, InvFeatureGen,
       predict_single, InvFeatures, predict_new,
       X_y_split, fit, predict
export load_gunpoint, TSdataset, univariate_datasets

export L1
#CONSTANTS
# the directory containing this file: (.../src/)
const MODULE_DIR = dirname(@__FILE__)

# Includes
include("IntervalBasedForest.jl")
include("datapath.jl")
include("measures.jl")
include("interface.jl")

end # module
