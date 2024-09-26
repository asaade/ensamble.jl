module ModelInitializer

export read_constraints, initialize_model!, configure_solver!

using CSV
using DataFrames
using JuMP
using StatsBase
using StringDistances

using ..Configuration
using ..Utils

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
    find_closest(input::String, valid_labels::Vector{String}) -> String

Find the closest valid label from the `valid_labels` list based on string similarity
(using Levenshtein distance).
"""
function find_closest(input::String, valid_labels::Vector{String})::String
    distances = [evaluate(Levenshtein(), lowercase(input), lowercase(label))
                 for label in valid_labels]
    closest_label_index = argmin(distances)  # Find index of the closest match
    return valid_labels[closest_label_index]
end

"""
    validate_cond_id!(cond_id::String, cond_ids_seen::Set{String}) -> String

Check if the `cond_id` is unique. If not, raise an error indicating the duplication.
"""
function validate_cond_id!(cond_id::String, cond_ids_seen::Set{String})::String
    if cond_id in cond_ids_seen
        error("Duplicate CONSTRAINT_ID: '$cond_id' found. Please ensure all constraint IDs are unique.")
    end
    push!(cond_ids_seen, cond_id)
    return cond_id
end

"""
    validate_type!(type::String, valid_types::Vector{String}, cond_id::String) -> String

Check if `type` is valid. If not, raise an error and suggest the closest valid label.
"""
function validate_type!(type::String, valid_types::Vector{String}, cond_id::String)::String
    matched_type = findfirst(t -> lowercase(type) == lowercase(t), valid_types)
    if matched_type === nothing
        closest_label = find_closest(type, valid_types)
        error("Invalid TYPE '$type' for constraint '$cond_id'. Suggestion: Use '$closest_label'.")
    end
    return type
end

"""
    validate_bounds!(lb, ub, cond_id::String)

Ensure LB and UB are valid numbers and LB <= UB.
"""
function validate_bounds!(lb, ub, cond_id::String)
    if isnothing(lb) || isnothing(ub)
        error("LB and UB are required for constraint '$cond_id'.")
    end
    @assert(isa(lb, Number) && isa(ub, Number),
            "LB and UB must be valid numbers in '$cond_id'.")
    @assert(lb <= ub, "UB must be greater than or equal to LB in '$cond_id'.")
end

"""
    read_constraints(file_path::String, parms::Parameters) -> Dict{String, Constraint}

Read constraints from a CSV file, returning a dictionary of Constraint objects.
This function checks for consistency and potential errors in the user-provided constraint file.
"""
function read_constraints(file_path::String, parms::Parameters)
    df = CSV.read(file_path, DataFrame; missingstring=nothing)
    uppercase_dataframe!(df)

    constraints = Dict{String, Constraint}()
    cond_ids_seen = Set{String}()
    valid_types = ["TEST", "NUMBER", "SUM", "ENEMIES", "ALLORNONE", "MAXUSE",
                   "OVERLAP", "INCLUDE", "EXCLUDE"]

    test_constraint_count = 0
    overlap_count = 0
    maxuse_count = 0

    for row in eachrow(df)
        if row[:ONOFF] == "ON"
            row = map(upcase, row)
            cond_id = validate_cond_id!(String(row[:CONSTRAINT_ID]), cond_ids_seen)
            type = validate_type!(String(row[:TYPE]), valid_types, cond_id)

            # Track and enforce rules for TEST, OVERLAP, and MAXUSE
            if lowercase(type) == "test"
                test_constraint_count += 1
            elseif lowercase(type) == "overlap"
                overlap_count += 1
            elseif lowercase(type) == "maxuse"
                maxuse_count += 1
            end

            # 'Condition', 'LB', 'UB' required for certain types
            condition_expr = String(row[:CONDITION])
            lb = get(row, :LB, 0)
            ub = get(row, :UB, 0)
            if type in ["TEST", "NUMBER", "SUM", "MAXUSE", "OVERLAP"]
                validate_bounds!(lb, ub, cond_id)
            end

            # Parse the condition or set it to always true if empty
            condition = if strip(condition_expr) == ""
                df -> trues(size(df, 1))
            else
                CriteriaParser.parse_criteria(condition_expr)
            end

            # Store the constraint in the dictionary
            if haskey(constraints, cond_id)
                error("Duplicate CONSTRAINT_ID '$cond_id' detected. Please ensure all CONSTRAINT_IDs are unique.")
            end
            constraints[cond_id] = Constraint(cond_id, type, eval(condition), lb, ub)
        end
    end

    # Ensure the 'TEST' constraint is present exactly once
    if test_constraint_count != 1
        error("Constraint 'TEST' must be included exactly once, but $test_constraint_count found.")
    end

    # Ensure 'OVERLAP' appears at most once
    if overlap_count > 1
        error("Constraint 'OVERLAP' can only appear once, but $overlap_count found.")
    end

    # Ensure 'MAXUSE' appears at most once
    if maxuse_count > 1
        error("Constraint 'MAXUSE' can only appear once, but $maxuse_count found.")
    end

    # Identify and store conflict-related constraints
    conflict_constraints = Dict{String, Constraint}()
    for (constraint_id, constraint) in constraints
        if constraint.type in ["ALLORNONE", "ENEMIES", "INCLUDE"]
            conflict_constraints[constraint_id] = constraint
        end
    end

    # Apply conflict-related constraints
    run_conflict_checks!(parms, conflict_constraints)

    return constraints
end

"""
    initialize_model!(model::Model, parms::Parameters, constraints::Dict{String, Constraint})

Initialize the optimization model, adding variables, the objective function,
and constraints based on the provided parameters.
"""
function initialize_model!(model::Model,
                           parms::Parameters,
                           constraints::Dict{String, Constraint})
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
    apply_objective!(model::Model, parms::Parameters, original_parms)

Apply the objective function specific to the method being used.
"""
function set_objective!(model::Model, parms::Parameters)
    y = model[:y]
    if parms.method in ["TIC2", "TIC3"]
        @objective(model, Max, y)
    else
        @objective(model, Min, y)
    end

    if parms.method in ["TCC"]
        objective_match_characteristic_curve!(model, parms)
    elseif parms.method in ["TIC"]
        objective_match_information_curve!(model, parms)
    elseif parms.method == "TIC2"
        objective_max_info(model, parms)
    elseif parms.method == "TIC3"
        objective_info_relative2(model, parms)
    elseif parms.method == "TC"
        throw(ArgumentError("Method TC not implemented yet."))
    else
        throw(ArgumentError("Unknown method: $(parms.method)"))
    end

    return model
end

"""
    apply_individual_constraint!(model::Model, parms::Parameters, constraint::Constraint)

Apply an individual constraint to the model based on the constraint type.
"""
function apply_individual_constraint!(model::Model, parms::Parameters,
                                      constraint::Constraint)
    lb, ub = constraint.lb, constraint.ub
    bank = parms.bank
    # items = 1:size(bank, 1)

    if constraint.type == "TEST"
        if !(lb <= parms.n <= ub)
            throw(DomainError("N=$(parms.n) are not between 'lb'=$lb and 'ub'=$ub in the TEST constraint"))
        end
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
        condition = constraint.condition(bank)
        parms.max_item_use = ub
        constraint_max_use(model, parms, condition, ub)
    elseif constraint.type == "OVERLAP"
        constraint_forms_overlap(model, parms, lb, ub)
    else
        throw(ArgumentError("Unknown constraint type: $(constraint.type)"))
    end

    return model
end

"""
    apply_constraints!(model::Model, parms::Parameters, constraints::Dict{String, Constraint})

Apply the constraints from the configuration to the optimization model.
"""
function apply_constraints!(model::Model, parms::Parameters,
                            constraints::Dict{String, Constraint})
    # Apply all constraints
    for (constraint_id, constraint) in constraints
        @debug string(APPLYING_CONSTRAINT_MESSAGE, " ", constraint_id)
        apply_individual_constraint!(model, parms, constraint)
    end
    return model
end

"""
    run_conflict_checks!(parms::Parameters, conflict_constraints::Dict{String, Constraint})

Run all conflict checks (Friends, Enemies, Anchor combinations) after applying non-conflict constraints.
The failure or logging of these checks will not affect the other constraints.
"""
function run_conflict_checks!(parms::Parameters,
                              conflict_constraints::Dict{String, Constraint})
    try
        friends_constraints = find_all_constraints_by_type(conflict_constraints,
                                                           "ALLORNONE")
        enemies_constraints = find_all_constraints_by_type(conflict_constraints, "ENEMIES")
        include_constraints = find_all_constraints_by_type(conflict_constraints, "INCLUDE")
        friends_constraints = vcat(friends_constraints, include_constraints)

        # Check conflicts between friends and enemies
        if !isempty(friends_constraints) && !isempty(enemies_constraints)
            for friends_constraint in friends_constraints
                friends_values = friends_constraint.condition(parms.bank)

                for enemies_constraint in enemies_constraints
                    enemies_values = enemies_constraint.condition(parms.bank)

                    # Apply one-to-many conflict rule between friends and enemies
                    conflict_df = apply_conflict_rule(friends_values, enemies_values,
                                                      one_to_many_conflict_rule)
                    if !isempty(conflict_df)
                        log_conflicts("Friends-Enemies", conflict_df)
                    end
                end
            end
        end

        # Check conflicts between anchors and enemies/friends
        if parms.anchor_tests > 0
            anchor_values = parms.bank.ANCHOR

            for enemies_constraint in enemies_constraints
                enemies_values = enemies_constraint.condition(parms.bank)

                # One-to-many conflict rule between anchor and enemies
                conflict_df = apply_conflict_rule(anchor_values, enemies_values,
                                                  one_to_many_conflict_rule)
                if !isempty(conflict_df)
                    log_conflicts("Anchor-Enemies", conflict_df)
                end
            end

            for friends_constraint in friends_constraints
                friends_values = friends_constraint.condition(parms.bank)

                # All-or-none conflict rule between anchor and friends
                conflict_df = apply_conflict_rule(friends_values, anchor_values,
                                                  all_or_none_conflict_rule)
                if !isempty(conflict_df)
                    log_conflicts("Anchor-Friends", conflict_df)
                end
            end
        end
    catch e
        @warn "Conflict checks for Friends, Enemies, and Anchor failed: $e"
    end
end

"""
    find_constraint_by_type(constraints::Dict{String, Constraint}, type::String)

Helper function to retrieve a constraint by its type.
"""
function find_constraint_by_type(constraints::Dict{String, Constraint}, type::String)
    for constraint in values(constraints)
        if constraint.type == type
            return constraint
        end
    end
    return nothing
end

"""
    find_all_constraints_by_type(constraints::Dict{String, Constraint}, type::String)

Helper function to retrieve all constraints by their type.
"""
function find_all_constraints_by_type(constraints::Dict{String, Constraint},
                                      type::String)::Vector{Constraint}
    return [constraint for constraint in values(constraints) if constraint.type == type]
end

"""
    log_conflicts(conflict_type::String, conflicting_rows::DataFrame, max_rows_to_log::Int=5)

Logs conflicts found in data.
If a conflict is found, the first `max_rows_to_log` rows of the conflict will be logged.
"""
function log_conflicts(conflict_type::String, conflicting_rows::DataFrame,
                       max_rows_to_log::Int=5)
    if size(conflicting_rows, 1) > 0
        @warn "Conflicting $conflict_type found"
        @warn "Displaying the first $max_rows_to_log rows of the conflict:",
              first(conflicting_rows, max_rows_to_log)
    end
end

"""
    one_to_many_conflict_rule(values1::Vector, values2::Vector)::Vector{Tuple}

Checks for one-to-many conflicts between two sets of values.
E.g., Multiple friends linked to the same enemy or anchors linked to multiple enemies.
"""
function one_to_many_conflict_rule(values1::Vector, values2::Vector)::Vector{Tuple}
    # Count occurrences of each (value1, value2) pair
    counts = Dict{Tuple{Any, Any}, Int}()
    for (v1, v2) in zip(values1, values2)
        if !ismissing(v1) && !ismissing(v2) && v1 > 0 && v2 > 0
            pair = (v1, v2)
            counts[pair] = get(counts, pair, 0) + 1
        end
    end

    # Return conflicting pairs where a friend or enemy appears more than once
    return [key for key in keys(counts) if counts[key] > 1]
end

"""
    all_or_none_conflict_rule(values1::Vector, values2::Vector)::Vector{Tuple}

Checks for all-or-none conflicts between two sets of values.
E.g., All friends in a group should be linked to the same anchor.
"""
function all_or_none_conflict_rule(friends_values::Vector,
                                   anchor_values::Vector)::Vector{Tuple}
    # Group by friends_values and check if anchor_values has consistent values for each group
    group_dict = Dict{Any, Set{Any}}()

    for (friend, anchor) in zip(friends_values, anchor_values)
        if !ismissing(friend) && !ismissing(anchor) && friend > 0 && anchor > 0
            # Group friends by category (friend) and record their associated anchors
            if !haskey(group_dict, friend)
                group_dict[friend] = Set{Any}()
            end
            push!(group_dict[friend], anchor)
        end
    end

    # Return groups where friends are associated with multiple anchors (conflicts)
    conflicting_pairs = [(friend, anchors)
                         for (friend, anchors) in group_dict if length(anchors) > 1]

    return conflicting_pairs
end

"""
    apply_conflict_rule(values1::Vector, values2::Vector, rule::Function)::DataFrame

Applies a conflict rule between two sets of values.
Returns a DataFrame with conflicting pairs if any exist.
"""
function apply_conflict_rule(values1::Vector, values2::Vector, rule::Function)::DataFrame
    # Apply the provided rule to check for conflicts between two sets of values
    conflicting_pairs = rule(values1, values2)

    # If conflicts are found, return the conflicting rows
    if !isempty(conflicting_pairs)
        return DataFrame(:col1 => map(x -> x[1], conflicting_pairs),
                         :col2 => map(x -> x[2], conflicting_pairs))
    else
        return DataFrame()  # Return empty DataFrame if no conflicts are found
    end
end

end
