module StringUtils

export upSymbol, upcase, upcaseKeys, cleanValues, uppercase_dataframe!, safe_read_csv, safe_read_yaml, safe_read_toml

using CSV: CSV
using YAML: YAML
using TOML: TOML
using DataFrames: DataFrames
using Logging

"""
    upSymbol(y::Any) -> Symbol

Converts any input into a Symbol with all uppercase characters. If the input
is not convertible to a string, returns the input untouched.

# Arguments

  - `y`: Any type, typically a string or symbol.

# Returns

  - A Symbol with its string representation converted to uppercase, or returns the input untouched.
"""
function upSymbol(y)::Symbol
    try
        return Symbol(uppercase(strip(string(y))))
    catch
        return y
    end
end

"""
    upcaseKeys(d::Dict{<:Union{String, Symbol}, Any}) -> Dict{Symbol, Any}

Recursively converts all keys in a dictionary to uppercase symbols, and ensures
the output dictionary is of type `Dict{Symbol, Any}`.

# Arguments

  - `d`: A dictionary with keys that are either `String` or `Symbol`.

# Returns

  - A standardized dictionary where all keys are symbols and converted to uppercase.
"""
function upcaseKeys(d::Dict{<:Union{String, Symbol}, Any})::Dict{Symbol, Any}
    return Dict(upSymbol(k) => isa(v, Dict) ? upcaseKeys(v) : v
                for (k, v) in d)
end


"""
    cleanValues(d::Dict{<:Union{String, Symbol}, Any}) -> Dict{Symbol, Any}

Recursively strips all string values in a dictionary, and ensures the output dictionary is of type `Dict{Symbol, Any}`.

# Arguments

  - `d`: A dictionary with keys that are either `String` or `Symbol`.

# Returns

  - A standardized dictionary where all keys are symbols and converted to uppercase.
"""
function cleanValues(d::Dict{<:Union{String, Symbol}, Any})::Dict{Symbol, Any}
    return Dict(k => isa(v, Dict) ? cleanValues(v) : isa(v, String) ? strip(v) : v
                for (k, v) in d)
end


"""
    uppercase_dataframe!(df::DataFrames.DataFrame)

Convert all strings in the DataFrame to uppercase, in place.
This operates on each column that is of type `String` or `AbstractString`.
"""
function uppercase_dataframe!(df::DataFrames.DataFrame)
    for col in eachcol(df)
        if eltype(col) <: AbstractString  # Only process string columns
            for i in 1:length(col)
                if !ismissing(col[i])  # Handle missing values
                    col[i] = uppercase(col[i])
                end
            end
        end
    end
    return df
end

"""
    upcase(x::Any) -> Any

Converts a string to uppercase and strips whitespace. If the input is not a string,
returns the input untouched.

# Arguments

  - `x`: Any input.

# Returns

  - An uppercase string with leading and trailing whitespace removed, or the input untouched.
"""
function upcase(x::Any)::Any
    try
        return uppercase(strip(x))
    catch
        return x
    end
end

"""
    safe_read_csv(file_path::String)::DataFrames.DataFrame

Safely reads a CSV file and returns a DataFrame. Logs an error if the file cannot be read.

# Arguments

  - `file_path`: The path to the CSV file.

# Returns

  - A `DataFrame` with the data from the CSV file, or an empty `DataFrame` if an error occurs.
"""
function safe_read_csv(file_path::String)::DataFrames.DataFrame
    try
        return CSV.read(file_path, DataFrames.DataFrame; stripwhitespace=true)
    catch e
        @error "Error reading CSV file: $file_path. Error: $e"
        return DataFrames.DataFrame()
    end
end

"""
    safe_read_yaml(file_path::String)::Dict{Symbol, Any}

Safely reads a YAML file and returns a dictionary with upcased symbol keys.

# Arguments

  - `file_path`: The path to the YAML file.

# Returns

  - A dictionary where keys are upcased symbols. Returns an empty dictionary if an error occurs.
"""
function safe_read_yaml(file_path::String)::Dict{Symbol, Any}
    try
        yaml_dict = YAML.load_file(file_path; dicttype=Dict{Union{String, Symbol}, Any})
        return yaml_dict
    catch e
        @error "Error reading YAML file: $file_path. Error: $e"
        return Dict{Symbol, Any}()
    end
end

"""
    safe_read_toml(file_path::String)::Dict{Symbol, Any}

Safely reads a TOML file and returns a dictionary with upcased symbol keys.

# Arguments

  - `file_path`: The path to the TOML file.

# Returns

  - A dictionary where keys are upcased symbols. Returns an empty dictionary if an error occurs.
"""
function safe_read_toml(file_path::String)::Dict{String, Any}
    try
        toml_dict = TOML.parsefile(file_path)
        return toml_dict
    catch e
        @error "Error reading TOML file: $file_path. Error: $e"
        return Dict{Symbol, Any}()
    end
end

end # module StringUtils
