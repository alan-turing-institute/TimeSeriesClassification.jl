
const DATA_DIR = joinpath(MODULE_DIR, "..", "data")

function load_dataset(fname::String)
    fpath = joinpath(DATA_DIR, fname)
    data_raw = load(fpath, header_exists=false)
    Table = table(data_raw)
    data_table = MatrixI(Table)
    return data_table
end

load_gunpoint() = load_dataset.(["GunPoint/train.csv", "GunPoint/test.csv"])

function MatrixI(table)
    cols = columns(table)
    n, p = length(cols[1]), length(cols)
    matrix = Matrix{Float64}(undef, n, p)
    for i=1:p
        matrix[:, i] = cols[i]
    end
    return matrix
end
#permutedims(cat(mat, mat2, dims=3),[1,3,2]) For Motions
