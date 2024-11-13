# src/validation/config_validation.jl
module ConfigValidation

export validate_config_data, validate_files, validate_irt_parameters,
    IRTLimits, DEFAULT_IRT_LIMITS, validate_bank_columns

using DataFrames
using ..ATAErrors
using ..Utils

"""
Validation limits for IRT parameters
"""
struct IRTLimits
    min_a::Float64
    max_a::Float64
    min_b::Float64
    max_b::Float64
    min_c::Float64
    max_c::Float64
end

const DEFAULT_IRT_LIMITS = IRTLimits(0.4, 2.0, -3.5, 3.5, 0.0, 0.5)

"""
Validates required columns in the item bank
"""
function validate_bank_columns(bank::DataFrame)
    required_columns = [:ID, :B]
    missing_columns = filter(col -> !hasproperty(bank, col), required_columns)

    if !isempty(missing_columns)
        throw(ValidationError(
            "Missing required columns in item bank",
            "bank_columns",
            missing_columns
        ))
    end
end

"""
Validates the complete configuration data structure
"""
function validate_config_data(config_data::Dict{Symbol,Any})::Nothing
    # Check required sections
    required_sections = [:FILES, :FORMS, :IRT]
    missing_sections = filter(s -> !haskey(config_data, s), required_sections)

    if !isempty(missing_sections)
        throw(ConfigError(
            "Missing required configuration sections",
            "Configuration",
            missing_sections
        ))
    end

    # Validate each section
    validate_files_section(config_data[:FILES])
    validate_forms_section(config_data[:FORMS])
    validate_irt_section(config_data[:IRT])

    return nothing
end

"""
Validates the FORMS section of the configuration
"""
function validate_forms_section(config_data::Dict{Symbol,Any})::Nothing
    required_data = [:NUMFORMS, :ANCHORTESTS, :SHADOWTEST]

    # Check required data in FORMS
    for form_key in required_data
        if !haskey(config_data, form_key)
            throw(ConfigError(
                "Missing FORMS section",
                "Configuration",
                form_key
            ))
        end
    end

    return nothing
end


"""
Validates the FILES section of the configuration
"""
function validate_files_section(files_data::Dict{Symbol,Any})::Nothing
    required_files = [:ITEMSFILE, :CONSTRAINTSFILE]

    # Check required files
    for file_key in required_files
        if !haskey(files_data, file_key)
            throw(ConfigError(
                "Missing required file configuration",
                "FILES",
                file_key
            ))
        end

        file_path = files_data[file_key]
        if !isfile(file_path)
            throw(ValidationError(
                "File does not exist",
                "FILES.$file_key",
                file_path
            ))
        end
    end

    return nothing
end

"""
Validates the IRT section of the configuration
"""
function validate_irt_section(irt_dict::Dict{Symbol,Any})::Nothing

    # Validate required fields
    required_fields = [:METHOD, :THETA]
    for field in required_fields
        if !haskey(irt_dict, field)
            throw(IRTConfigError("IRT configuration missing required field: $field"))
        end
    end

    return nothing
end

"""
    validate_irt_config(config_data::Dict{Symbol, Any})::Dict{Symbol, Any}

Validates IRT configuration parameters.
"""
function validate_irt_config(config_data::Dict{Symbol, Any})::Dict{Symbol, Any}
    irt_dict = config_data[:IRT]

    # Validate relative target parameters
    weights = get(irt_dict, :RELATIVETARGETWEIGHTS, [1.0, 1.0])
    points = get(irt_dict, :RELATIVETARGETPOINTS, [-1.0, 1.0])
    if length(weights) != length(points)
        throw(IRTConfigError("Target weights and points must have same length"))
    end

        # Validate METHOD
    valid_methods = ["TCC", "TCC2", "TIC", "TIC2", "MIXED"]
    if !haskey(irt_data, :METHOD) || !(uppercase(irt_data[:METHOD]) in valid_methods)
        throw(ValidationError(
            "Invalid or missing IRT method",
            "IRT.METHOD",
            get(irt_data, :METHOD, nothing)
        ))
    end

    # Validate THETA points
    if haskey(irt_data, :THETA)
        theta = irt_data[:THETA]
        if !isa(theta, Vector) || !all(x -> isa(x, Number), theta)
            throw(ValidationError(
                "THETA must be a vector of numbers",
                "IRT.THETA",
                theta
            ))
        end
    end

    # Validate D parameter
    if haskey(irt_data, :D)
        D = irt_data[:D]
        if !isa(D, Number) || D <= 0
            throw(ValidationError(
                "D must be a positive number",
                "IRT.D",
                D
            ))
        end
    end

    return irt_dict
end

"""
Validates IRT parameters for each item
"""
function validate_irt_parameters(bank::DataFrame)
    invalid_items = DataFrame()

    if !("NUM_CATEGORIES" in names(bank))
        bank.NUM_CATEGORIES::Vector{Integer} .= 0
    end

    bank.NUM_CATEGORIES = convert(Vector{Union{Integer, Missing}}, bank.NUM_CATEGORIES)

    bank.CHECK .= false

    if !("C" in names(bank))
        bank.C .= 0.0
    end

    b_columns = filter(col -> occursin(r"^B\d*$|^B$", string(col)), names(bank))

    for row in eachrow(bank)
        row.MODEL in SUPPORTED_MODELS ||
            throw(ArgumentError("Unsupported model type: $model"))

        categories = length(collect(skipmissing(row[b_columns]))) + 1

        if ismissing(row.C)
            row.C = 0.0
        end

        if ismissing(row.A) || row.A <= 0.0 || !(0.0 ≤ row.C ≤ 1.0) || categories == 0
            push!(invalid_items, row)
            break
        else
            row.CHECK = true
        end

        if ismissing(row.NUM_CATEGORIES) || row.NUM_CATEGORIES != categories
            row.NUM_CATEGORIES = categories
        end

    end

    return invalid_items
end

end # module ConfigValidation
