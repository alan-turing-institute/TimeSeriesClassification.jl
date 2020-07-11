"""
    `TSdataset(dataset::Array)`
TSdataset takes subset of the `univariate_datasets` avilabe on the `timeseriesclassification`
website and adds `csv` files to the `data` folder after conversion (`ts` -> `csv`).
eg.
```julia
   TSdataset(["ACSF1", "Adiac"])
```
"""
function TSdataset(dataset::Array) #check on win
    for Dir in dataset
        exdir = string("data/", Dir)
        _link = string("http://timeseriesclassification.com/Downloads/", Dir ,".zip")
        Base.download(_link, Dir)
        fileFullPath = isabspath(Dir) ?  Dir : joinpath(pwd(), Dir)
        basePath = dirname(fileFullPath)
        outPath = (exdir == "" ? basePath : (isabspath(exdir) ? exdir : joinpath(pwd(),exdir)))
        isdir(outPath) ? "" : mkdir(outPath)
        zarchive = ZipFile.Reader(fileFullPath)
        for f in zarchive.files
            if f.name[end-1:end] == "ts"
                f.name = string(f.name[1:end-2], "csv")
                fullFilePath = joinpath(outPath,f.name)
                write(fullFilePath, read(f))
                open(fullFilePath) do input
                    readuntil(input, "@data")
                    write(fullFilePath, read(input))
                end
                run(`sed -i'.original' 's/:/,/g' $fullFilePath`)
                fullFilePath1 = string(fullFilePath, ".original")
                run(`rm $fullFilePath1`)
            end
        end
        close(zarchive)
        run(`rm $fileFullPath`)
    end
end

function TSdataset(filepath::String)
    exdir = string("data/", basename(filepath))
    fileFullPath = isabspath(filepath) ?  filepath : joinpath(pwd(), filepath)
    basePath = dirname(fileFullPath)
    outPath = (exdir == "" ? basePath : (isabspath(exdir) ? exdir : joinpath(pwd(),exdir)))
    isdir(outPath) ? "" : mkdir(outPath)
    files = readdir(fileFullPath, join=true)
    for f in files
        if f[end-1:end] == "ts"
            fullFilePath = joinpath(outPath, string(f[1:end-2], "csv"))
            write(fullFilePath, read(f))
            open(fullFilePath) do input
                readuntil(input, "@data")
                write(fullFilePath, read(input))
            end
            run(`sed -i'.original' 's/:/,/g' $fullFilePath`)
            fullFilePath1 = string(fullFilePath, ".original")
            run(`rm $fullFilePath1`)
        end
    end
end
