
const DATA_DIR = joinpath(MODULE_DIR, "..", "data")

function load_dataset(fname::String)
    fpath = joinpath(DATA_DIR, fname)
    data_raw = load(fpath, header_exists=false)
    data_table = Tables.matrix(data_raw)
end

load_gunpoint() = load_dataset.(["GunPoint/train.csv", "GunPoint/test.csv"])
#permutedims(cat(mat, mat2, dims=3),[1,3,2]) For Motions
