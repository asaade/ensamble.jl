module ConstraintValidation

export find_closest, validate_type, validate_bounds, validate_onoff_field,
       validate_condition_syntax, validate_cond_id

using Logging
using ..CriteriaParser  # Ensure the parser is available for condition syntax checks

const VALID_TYPES = [
    "TEST", "NUMBER", "SCORE", "SUM",
    "ENEMIES", "ALLORNONE", "MAXUSE",
    "OVERLAP", "INCLUDE", "EXCLUDE"
]

"""
    find_closest(input::String, valid_labels::Vector{String})::String
                      valid_labels::Vector{String})::String

Find the closest valid label from the `valid_labels` list based on string similarity
(using Levenshtein distance).
"""
function find_closest(input::String, valid_labels::Vector{String})::String
    distances = [evaluate(Levenshtein(), lowercase(input), lowercase(label))
                 for
                 label in valid_labels]
    closest_label_index = argmin(distances)
    return valid_labels[closest_label_index]
end

"""
    validate_cond_id!(cond_id::String, cond_ids_seen::Set{String})

Check if the `cond_id` is unique. If not, raise an error indicating the duplication.
"""
function validate_cond_id(cond_id::String, cond_ids_seen::Set{String})
    if cond_id in cond_ids_seen
        error(
            "Duplicate CONSTRAINT_ID: '$cond_id' found. Please ensure all constraint IDs are unique.",
        )
    end
    push!(cond_ids_seen, cond_id)
    return cond_id
end

"""
    validate_type!(type::String, valid_types::Vector{String}, cond_id::String)

Check if `type` is valid. If not, raise an error and suggest the closest valid label.
"""
function validate_type(type::String, cond_id::String)
    matched_type = findfirst(t -> lowercase(type) == lowercase(t), VALID_TYPES)
    if matched_type === nothing
        closest_label = find_closest(type, VALID_TYPES)
        error(
            "Invalid TYPE '$type' for constraint '$cond_id'. Suggestion: Use '$closest_label'.",
        )
    end
    return type
end

"""
    validate_bounds!(lb::Number, ub::Number, cond_id::String)

Ensure LB and UB are valid numbers and LB ≤ UB.
"""
function validate_bounds(type::String, lb, ub, cond_id::String)
    if type in ["NUMBER", "SCORE", "SUM", "MAXUSE", "OVERLAP"]
        if (lb === nothing || ub === nothing)
            error("Bounds `LB` and `UB` must be specified for constraint '$cond_id' of type '$type'.")
        elseif lb > ub
            error("UB must be greater than or equal to LB in '$cond_id'.")
        end
    end
end

"""
    validate_onoff_field(onoff::String, cond_id::String)

Ensures the `ONOFF` field contains either "ON" or "OFF".
"""
function validate_onoff_field(onoff::String, cond_id::String)
    if !(onoff in ["ON", "OFF"])
        error("Invalid ONOFF value '$onoff' for constraint '$cond_id'. Use 'ON' or 'OFF'.")
    end
end

"""
    validate_condition_syntax(condition::String, cond_id::String)

Checks the syntax of `CONDITION` by parsing it with the `CriteriaParser`.
"""
function validate_condition_syntax(condition::String, cond_id::String)
    try
        CriteriaParser.parse_criteria(condition)
    catch e
        error("Syntax error in CONDITION for constraint '$cond_id': $e")
    end
end

end  # module ConstraintValidation
