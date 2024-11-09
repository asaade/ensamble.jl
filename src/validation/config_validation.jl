# src/validation/config_validation.jl
module ConfigValidation

export validate_config_data, validate_files, validate_irt_parameters,
    IRTLimits, DEFAULT_IRT_LIMITS, validate_bank_columns

using DataFrames
using ..ATAErrors

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
function validate_irt_parameters(bank::DataFrame, limits::IRTLimits = DEFAULT_IRT_LIMITS)
    invalid_items = DataFrame()

    for row in eachrow(bank)
        if !(limits.min_a <= row.A <= limits.max_a &&
             limits.min_b <= row.B <= limits.max_b &&
             limits.min_c <= row.C <= limits.max_c)
            push!(invalid_items, row)
        end
    end

    return invalid_items
end

end # module ConfigValidation
