module MLJTime

# IMPORTS
using IndexedTables, CSVFiles, ZipFile
using  MLJBase: fit!, predict, machine, partition
import MLJModelInterface
import MLJModelInterface: @mlj_model, fit, predict, matrix
using StableRNGs

# EXPORTS
export RandomForestClassifierFit, InvFeatureGen,
       predict_single, InvFeatures, predict_new,
       X_y_split, fit!, predict, machine

export load_gunpoint, TSdataset, univariate_datasets,
       load_ts_file, ts_dataset, partition, matrix,
       array

export L1, StableRNG
#CONSTANTS
# the directory containing this file: (.../src/)
const MODULE_DIR = dirname(@__FILE__)
const MMI = MLJModelInterface

# Includes
include("IntervalBasedForest.jl")
include("datapath.jl")
include("measures.jl")
include("interface.jl")

end # module
