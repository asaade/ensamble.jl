# src/utils/file_utils.jl
module FileUtils

export ensure_dir, safe_read_csv, safe_read_toml

using CSV, TOML

using ...ATAErrors

"""
Ensures directory exists, creates if necessary
"""
function ensure_dir(file_path::String)::Nothing
    dir = dirname(file_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    return nothing
end

"""
Safely reads CSV file with error handling
"""
function safe_read_csv(file_path::String)::DataFrame
    try
        if !isfile(file_path)
            throw(FilePathError("CSV", file_path, "File does not exist"))
        end
        return CSV.read(file_path, DataFrame)
    catch e
        if e isa ATAError
            rethrow(e)
        else
            throw(FileError("Failed to read CSV file", file_path, e))
        end
    end
end

"""
Safely reads TOML file with error handling
"""
function safe_read_toml(file_path::String)::Dict
    try
        if !isfile(file_path)
            throw(FilePathError("TOML", file_path, "File does not exist"))
        end
        return TOML.parsefile(file_path)
    catch e
        if e isa ATAError
            rethrow(e)
        else
            throw(FileError("Failed to read TOML file", file_path, e))
        end
    end
end

end # module FileUtils
