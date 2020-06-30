module MLJTime

# IMPORTS
using IndexedTables, CSVFiles, ZipFile
<<<<<<< HEAD
import MLJModelInterface: @mlj_model, fit, predict, matrix

# EXPORTS
export X_y_split, fit, predict

=======

# EXPORTS
export RandomForestClassifierFit, InvFeatureGen,
       predict_single, InvFeatures, predict_new,
       X_y_split, fit, predict
>>>>>>> 1ee676afa6699fcb954204a70d00527fda2affad
export load_gunpoint, TSdataset, univariate_datasets

export L1
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
