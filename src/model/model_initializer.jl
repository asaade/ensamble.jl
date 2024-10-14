module ModelInitializer

export read_constraints, initialize_model!, configure_solver!

using CSV
using DataFrames
using JuMP
using StatsBase
using StringDistances

using ..Configuration
using ..Utils

# Submodules
include("constraints.jl")
include("criteria_parser.jl")
include("solvers.jl")
using .Constraints
using .CriteriaParser
using .SolverConfiguration

# Module constants
const APPLYING_CONSTRAINT_MESSAGE = "Applying constraint: "
const INITIALIZING_MODEL_MESSAGE = "Initializing optimization model..."
const MODEL_FILE = "./results/model.lp"

"""
    find_closest(input, valid_labels)

Find the closest valid label from the `valid_labels` list based on string similarity
(using Levenshtein distance).
"""
function find_closest(input::AbstractString, valid_labels::Vector{AbstractString})::AbstractString
    distances = [evaluate(Levenshtein(), lowercase(input), lowercase(label)) for label in valid_labels]
    closest_label_index = argmin(distances)
    return valid_labels[closest_label_index]
end

"""
    validate_cond_id!(cond_id, cond_ids_seen)

Check if the `cond_id` is unique. If not, raise an error indicating the duplication.
"""
function validate_cond_id!(cond_id::AbstractString, cond_ids_seen::Set{AbstractString})
    if cond_id in cond_ids_seen
        error("Duplicate CONSTRAINT_ID: '$cond_id' found. Please ensure all constraint IDs are unique.")
    end
    push!(cond_ids_seen, cond_id)
    return cond_id
end

"""
    validate_type!(type, valid_types, cond_id)

Check if `type` is valid. If not, raise an error and suggest the closest valid label.
"""
function validate_type!(type::String, valid_types::Vector{String}, cond_id::String)
    matched_type = findfirst(t -> lowercase(type) == lowercase(t), valid_types)
    if matched_type === nothing
        closest_label = find_closest(type, valid_types)
        error("Invalid TYPE '$type' for constraint '$cond_id'. Suggestion: Use '$closest_label'.")
    end
    return type
end

"""
    validate_bounds!(lb, ub, cond_id)

Ensure LB and UB are valid numbers and LB ≤ UB.
"""
function validate_bounds!(lb::Number, ub::Number, cond_id::AbstractString)
    if !(isa(lb, Number) && isa(ub, Number))
        error("LB and UB must be valid numbers in '$cond_id'.")
    end
    if lb > ub
        error("UB must be greater than or equal to LB in '$cond_id'.")
    end
end

"""
    read_constraints(file_path, parms) -> Dict{String, Constraint}

Read constraints from a CSV file, returning a dictionary of Constraint objects.
This function checks for consistency and potential errors in the user-provided constraint file.
"""
function read_constraints(file_path::AbstractString, parms::Parameters)
    df = CSV.read(file_path, DataFrame; missingstring=nothing)
    uppercase_dataframe!(df)

    constraints = Dict{AbstractString, Constraint}()
    cond_ids_seen = Set{AbstractString}()
    valid_types = ["TEST", "NUMBER", "SUM", "ENEMIES", "ALLORNONE", "MAXUSE", "OVERLAP", "INCLUDE", "EXCLUDE"]

    # Track counts for specific types
    type_counts = Dict("test" => 0, "overlap" => 0, "maxuse" => 0)

    for row in eachrow(df)
        if row[:ONOFF] == "ON"
            row = map(upcase, row)
            cond_id = validate_cond_id!(String(row[:CONSTRAINT_ID]), cond_ids_seen)
            type = validate_type!(String(row[:TYPE]), valid_types, cond_id)

            # Track and enforce rules for TEST, OVERLAP, and MAXUSE
            if lowercase(type) in keys(type_counts)
                type_counts[lowercase(type)] += 1
            end

            condition_expr = String(row[:CONDITION])
            lb = get(row, :LB, 0)
            ub = get(row, :UB, 0)
            if type in ["TEST", "NUMBER", "SUM", "MAXUSE", "OVERLAP"]
                validate_bounds!(lb, ub, cond_id)
            end

            if type == "TEST"
                parms.max_items = ub
            end

            # Parse the condition or default to a true condition
            condition = strip(condition_expr) == "" ? df -> trues(size(df, 1)) : CriteriaParser.parse_criteria(condition_expr)
            constraints[cond_id] = Constraint(cond_id, type, eval(condition), lb, ub)
        end
    end

    # Ensure specific constraints appear exactly once
    for (type, count) in type_counts
        if type == "test" && count != 1
            error("Constraint 'TEST' must be included exactly once, but $count found.")
        elseif count > 1 && type in ["overlap", "maxuse"]
            error("Constraint '$type' can only appear once, but $count found.")
        end
    end

    # Apply conflict-related constraints
    conflict_constraints = Dict{AbstractString, Constraint}()
    for (constraint_id, constraint) in constraints
        if constraint.type in ["ALLORNONE", "ENEMIES", "INCLUDE"]
            conflict_constraints[constraint_id] = constraint
        end
    end
    run_conflict_checks!(parms, conflict_constraints)

    return constraints
end

"""
    initialize_model!(model, parms, constraints)

Initialize the optimization model, adding variables, the objective function,
and constraints based on the provided parameters.
"""
function initialize_model!(model::Model, parms::Parameters, constraints::Dict{AbstractString, Constraint})
    @info INITIALIZING_MODEL_MESSAGE
    num_items = size(parms.bank, 1)
    num_forms = parms.num_forms + (parms.shadow_test > 0 ? 1 : 0)

    # Declare model variables
    @variable(model, y >= 0.0)
    @variable(model, x[1:num_items, 1:num_forms], Bin)

    # Set the objective and apply constraints
    set_objective!(model, parms)
    constraint_add_anchor!(model, parms)
    apply_constraints!(model, parms, constraints)
    write_to_file(model, MODEL_FILE)

    return model
end

"""
    set_objective!(model, parms)

Apply the objective function specific to the method being used.
"""
function set_objective!(model::Model, parms::Parameters)
    y = model[:y]
    if parms.method in ["TIC2", "TIC3"]
        @objective(model, Max, y)
    else
        @objective(model, Min, y)
    end

    # Delegate to the appropriate objective function
    if parms.method == "TCC"
        objective_match_characteristic_curve!(model, parms)
    elseif parms.method == "TIC"
        objective_match_information_curve!(model, parms)
    elseif parms.method == "TIC2"
        objective_max_info(model, parms)
    elseif parms.method == "TIC3"
        objective_info_relative2(model, parms)
    else
        throw(ArgumentError("Unknown method: $(parms.method)"))
    end

    return model
end

"""
    apply_individual_constraint!(model, parms, constraint)

Apply an individual constraint to the model based on the constraint type.
"""
function apply_individual_constraint!(model::Model, parms::Parameters, constraint::Constraint)
    lb, ub = constraint.lb, constraint.ub
    bank = parms.bank

    if constraint.type == "TEST"
        parms.max_items = ub
        constraint_items_per_form(model, parms, lb, ub)
    elseif constraint.type == "NUMBER"
        condition = constraint.condition(bank)
        constraint_item_count(model, parms, condition, lb, ub)
    elseif constraint.type == "SUM"
        item_vals = constraint.condition(bank)
        constraint_item_sum(model, parms, item_vals, lb, ub)
    elseif constraint.type == "ENEMIES"
        condition = constraint.condition(bank)
        constraint_enemies_in_form(model, parms, condition)
    elseif constraint.type == "ALLORNONE"
        condition = constraint.condition(bank)
        constraint_friends_in_form(model, parms, condition)
    elseif constraint.type == "INCLUDE"
        condition = constraint.condition(bank)
        constraint_include_items(model, condition)
    elseif constraint.type == "EXCLUDE"
        condition = constraint.condition(bank)
        constraint_exclude_items(model, parms, condition)
    elseif constraint.type == "MAXUSE"
        parms.max_item_use = ub
        condition = constraint.condition(bank)
        constraint_max_use(model, parms, condition, ub)
    elseif constraint.type == "OVERLAP"
        constraint_forms_overlap(model, parms, lb, ub)
    else
        throw(ArgumentError("Unknown constraint type: $(constraint.type)"))
    end

    return model
end

"""
    apply_constraints!(model, parms, constraints)

Apply all constraints to the optimization model.
"""
function apply_constraints!(model::Model, parms::Parameters, constraints::Dict{AbstractString, Constraint})
    for (constraint_id, constraint) in constraints
        @debug "$APPLYING_CONSTRAINT_MESSAGE $constraint_id"
        apply_individual_constraint!(model, parms, constraint)
    end
    return model
end

"""
    normalize_condition_values(values)

Converts Boolean vectors to group values (`true` becomes a group identifier) and normalizes other vector types.
"""
function normalize_condition_values(values::Union{Vector, BitVector})::Vector{Any}
    # if ndims(values) == 2
    #     values = valuesues[1, :]
    # end
    return isa(values, BitVector) ? [x ? 1 : missing for x in values] : values
end

"""
    run_conflict_checks!(parms, conflict_constraints)

Run conflict checks for friends, enemies, and anchors.
"""
function run_conflict_checks!(parms::Parameters, conflict_constraints::Dict{AbstractString, Constraint})
    friends_constraints = find_all_constraints_by_type(conflict_constraints, "ALLORNONE")
    enemies_constraints = find_all_constraints_by_type(conflict_constraints, "ENEMIES")
    include_constraints = find_all_constraints_by_type(conflict_constraints, "INCLUDE")
    friends_constraints = vcat(friends_constraints, include_constraints)

    # Check conflicts between friends and enemies
    try
        if !isempty(friends_constraints) && !isempty(enemies_constraints)
            for friends_constraint in friends_constraints
                friends_values = normalize_condition_values(friends_constraint.condition(parms.bank))
                for enemies_constraint in enemies_constraints
                    enemies_values = normalize_condition_values(enemies_constraint.condition(parms.bank))
                    conflict_df = apply_conflict_rule(friends_values, enemies_values, one_to_many_conflict_rule)
                    if !isempty(conflict_df)
                        log_conflicts("Friends-Enemies", conflict_df)
                    end
                end
            end
        end
    catch e
        @warn "Conflict checks for Friends-Enemies failed: $e"
    end

    # Check conflicts between anchors and friends/enemies
    try
        if parms.anchor_tests > 0
            anchor_values = normalize_condition_values(parms.bank.ANCHOR)
            for enemies_constraint in enemies_constraints
                enemies_values = normalize_condition_values(enemies_constraint.condition(parms.bank))
                conflict_df = apply_conflict_rule(anchor_values, enemies_values, one_to_many_conflict_rule)
                if !isempty(conflict_df)
                    log_conflicts("Anchor-Enemies", conflict_df)
                end
            end
            for friends_constraint in friends_constraints
                friends_values = normalize_condition_values(friends_constraint.condition(parms.bank))
                conflict_df = apply_conflict_rule(friends_values, anchor_values, all_or_none_conflict_rule)
                if !isempty(conflict_df)
                    log_conflicts("Anchor-Friends", conflict_df)
                end
            end
        end
    catch e
        @warn "Conflict checks for Anchors-Friends/Enemies failed: $e"
    end
end

"""
    find_all_constraints_by_type(constraints, type)

Helper function to retrieve all constraints by their type.
"""
function find_all_constraints_by_type(constraints::Dict{AbstractString, Constraint}, type::AbstractString)::Vector{Constraint}
    return [constraint for constraint in values(constraints) if constraint.type == type]
end

"""
    log_conflicts(conflict_type, conflicting_rows, max_rows_to_log)

Logs conflicts found in data.
"""
function log_conflicts(conflict_type::AbstractString, conflicting_rows::DataFrame, max_rows_to_log::Int = 5)
    num_conflicts = size(conflicting_rows, 1)
    if num_conflicts > 0
        @warn "$num_conflicts conflicting $conflict_type found"
        @warn "Displaying the first $max_rows_to_log rows of the conflict:", first(conflicting_rows, max_rows_to_log)
    end
end

"""
    one_to_many_conflict_rule(values1, values2)

Checks for one-to-many conflicts between two sets of values.
"""
function one_to_many_conflict_rule(values1::Vector, values2::Vector)::Vector{Tuple}
    counts = Dict{Tuple{Any, Any}, Int}()
    for (v1, v2) in zip(values1, values2)
        if !ismissing(v1) && !ismissing(v2)
            pair = (v1, v2)
            counts[pair] = get(counts, pair, 0) + 1
        end
    end
    return [key for key in keys(counts) if counts[key] > 1]
end

"""
    all_or_none_conflict_rule(friends_values, anchor_values)

Checks for all-or-none conflicts between two sets of values.
"""
function all_or_none_conflict_rule(friends_values::Vector, anchor_values::Vector)::Vector{Tuple}
    group_dict = Dict{Any, Set{Any}}()
    for (friend, anchor) in zip(friends_values, anchor_values)
        if !ismissing(friend) && !ismissing(anchor)
            if !haskey(group_dict, friend)
                group_dict[friend] = Set{Any}()
            end
            push!(group_dict[friend], anchor)
        end
    end
    return [(friend, anchors) for (friend, anchors) in group_dict if length(anchors) > 1]
end

"""
    apply_conflict_rule(values1, values2, rule)

Applies a conflict rule between two sets of values and returns a DataFrame with conflicting pairs if any exist.
"""
function apply_conflict_rule(values1::Vector, values2::Vector, rule::Function)::DataFrame
    conflicting_pairs = rule(values1, values2)
    if !isempty(conflicting_pairs)
        return DataFrame(:col1 => map(x -> x[1], conflicting_pairs), :col2 => map(x -> x[2], conflicting_pairs))
    else
        return DataFrame()  # Return empty DataFrame if no conflicts are found
    end
end

end
