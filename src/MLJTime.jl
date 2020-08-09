module MLJTime

# IMPORTS
using IndexedTables, CSVFiles, ZipFile
using  MLJBase: fit!, predict, machine, partition, L1, CV, range, cross_entropy, 
       predict_mode, accuracy
import MLJModelInterface
import MLJModelInterface: @mlj_model, fit, predict, matrix
using StableRNGs

# EXPORTS
export RandomForestClassifierFit, InvFeatureGen, predict_new,
       fit!, predict, machine, fitted_params, predict_mode, accuracy

export univariate_datasets, load_dataset, load_from_tsfile_to_NDArray,
       load_ts_file, ts_dataset, partition, matrix, dwt_distance, load_NDdataset

export L1, StableRNG, CV, range, cross_entropy
#CONSTANTS
# the directory containing this file: (.../src/)
const MODULE_DIR = dirname(@__FILE__)
const MMI = MLJModelInterface

# Includes
include("IntervalBasedForest.jl")
include("datapath.jl")
include("interface.jl")
include("distances.jl")

end # module
