module ModelInitializer

export read_constraints, initialize_model!, configure_solver!

using CSV
using DataFrames
using JuMP
using Logging
using StatsBase
using StringDistances

using ..Configuration
using ..Utils

# Submodules
include("constraints.jl")
include("criteria_parser.jl")
include("solvers.jl")
include("constraint_validation.jl")

using .Constraints
using .ConstraintValidation
using .CriteriaParser
using .SolverConfiguration

# Module constants
const APPLYING_CONSTRAINT_MESSAGE = "Applying constraint: "
const INITIALIZING_MODEL_MESSAGE = "Initializing optimization model..."
const MODEL_FILE = "./results/model.lp"

const CONFLICTING_CONSTRAINTS = ["ALLORNONE", "ENEMIES", "INCLUDE", "EXCLUDE"]

level = Logging.Info

"""
    read_constraints(file_path::String, parms::Parameters)

Read constraints from a CSV file, returning a dictionary of Constraint objects.
This function checks for consistency and potential errors in the user-provided constraint file.
"""
function read_constraints(file_path::String, parms::Parameters)
    df = CSV.read(
        file_path,
        DataFrame;
        stripwhitespace = true,
        pool = false,
        stringtype = String,
        missingstring = nothing
    )
    uppercase_dataframe!(df)

    constraints = Dict{String, Constraint}()
    cond_ids_seen = Set{String}()

    # Track counts for specific types
    type_counts = Dict("test" => 0, "overlap" => 0, "maxuse" => 0)

    for row in eachrow(df)
        row = map(upcase, row)
        if row[:ONOFF] == "ON"
            cond_id = validate_cond_id(String(row[:CONSTRAINT_ID]), cond_ids_seen)
            type = validate_type(String(row[:TYPE]), cond_id)

            # Track and enforce rules for TEST, OVERLAP, and MAXUSE
            if lowercase(type) in keys(type_counts)
                type_counts[lowercase(type)] += 1
            end

            condition_expr = String(row[:CONDITION])
            lb = get(row, :LB, 0)
            ub = get(row, :UB, 0)
            validate_bounds(type, lb, ub, cond_id)

            # Parse the condition or default to a true condition
            condition = if strip(condition_expr) == ""
                df -> trues(size(df, 1))
            else
                validate_condition_syntax(condition_expr, cond_id)
            end
            constraints[cond_id] = Constraint(cond_id, type, eval(condition), lb, ub)
        end
    end

    # Ensure specific constraints appear exactly once
    for (type, count) in type_counts
        if type == "TEST" && count != 1
            error("Constraint 'TEST' must be included exactly once, but $count found.")
        elseif count > 1 && type in ["overlap", "maxuse"]
            parms.max_item_use = max(2, parms.max_item_use)
            warn("Constraint '$type' can only appear once, but $count found. MAX_ITEM_USE set as >= 2")
        end
    end

    # Apply conflict-related constraints
    conflict_constraints = Dict{String, Constraint}()
    for (constraint_id, constraint) in constraints
        if constraint.type in CONFLICTING_CONSTRAINTS
            conflict_constraints[constraint_id] = constraint
        end
    end
    run_conflict_checks!(parms, conflict_constraints)

    return constraints
end

"""
    initialize_model!(model::Model, parms::Parameters,
                           constraints::Dict{String, Constraint})

Initialize the optimization model, adding variables, the objective function,
and constraints based on the provided parameters.
"""
function initialize_model!(
        model::Model, parms::Parameters, constraints::Dict{String, Constraint}
)
    @info INITIALIZING_MODEL_MESSAGE
    num_items = size(parms.bank, 1)
    num_forms = parms.num_forms + (parms.shadow_test_size > 0 ? 1 : 0)

    # Declare model variables
    @variable(model, y>=0.0)
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
    elseif parms.method == "TCC2"
        # objective_match_mean_var!(model, parms, 3.0)
        @error("TCC2 is temporarily disabled")
    elseif parms.method == "MIXED"
        objective_match_characteristic_curve!(model, parms)
        objective_match_information_curve!(model, parms)
    elseif parms.method == "TIC"
        objective_match_information_curve!(model, parms)
    elseif parms.method == "TIC2"
        objective_max_info(model, parms)
    elseif parms.method == "TIC3"
        objective_info_relative(model, parms)
    else
        throw(ArgumentError("Unknown method: $(parms.method)"))
    end

    return model
end

"""
    apply_individual_constraint!(model::Model, parms::Parameters,
                                      constraint::Constraint)

Apply an individual constraint to the model based on the constraint type.
"""
function apply_individual_constraint!(
        model::Model, parms::Parameters, constraint::Constraint
)
    lb, ub = constraint.lb, constraint.ub
    bank = parms.bank

    if constraint.type == "TEST"
        parms.max_items = ub
        constraint_items_per_form(model, parms, lb, ub)
    elseif constraint.type == "NUMBER"
        selected_items = constraint.condition(bank)
        constraint_item_count(model, parms, selected_items, lb, ub)
    elseif constraint.type == "SCORE"
        selected_items = constraint.condition(bank)
        constraint_score_sum(model, parms, selected_items, lb, ub)
    elseif constraint.type == "SUM"
        item_vals = constraint.condition(bank)
        constraint_item_sum(model, parms, item_vals, lb, ub)
    elseif constraint.type == "ENEMIES"
        selected_items = constraint.condition(bank)
        constraint_enemies(model, parms, selected_items)
    elseif constraint.type == "ALLORNONE"
        selected_items = constraint.condition(bank)
        constraint_friends_in_form(model, parms, selected_items)
    elseif constraint.type == "INCLUDE"
        selected_items = constraint.condition(bank)
        constraint_fix_items(model, selected_items)
    elseif constraint.type == "EXCLUDE"
        selected_items = constraint.condition(bank)
        constraint_exclude_items(model, selected_items)
    elseif constraint.type == "MAXUSE"
        parms.max_item_use = ub
        selected_items = constraint.condition(bank)
        constraint_max_use(model, parms, selected_items)
    elseif constraint.type == "OVERLAP"
        constraint_forms_overlap(model, parms, lb, ub)
    else
        throw(ArgumentError("Unknown constraint type: $(constraint.type)"))
    end

    return model
end

"""
    apply_constraints!(model::Model, parms::Parameters,
                            constraints::Dict{String, Constraint})

Apply all constraints to the optimization model.
"""
function apply_constraints!(
        model::Model, parms::Parameters, constraints::Dict{String, Constraint}
)
    for (constraint_id, constraint) in constraints
        @debug "$APPLYING_CONSTRAINT_MESSAGE $constraint_id"
        apply_individual_constraint!(model, parms, constraint)
    end
    return model
end

"""
    normalize_condition_values(values::Union{Vector, BitVector})::Vector{Any}

Converts Boolean vectors to group values (`true` becomes a group identifier) and normalizes other vector types.
"""
function normalize_condition_values(values::Union{Vector, BitVector})::Vector{Any}
    try
        # Handle BitVector: convert true to 1, false to missing
        if isa(values, BitVector)
            return [x ? 1 : missing for x in values]
        else
            # Handle other Vector types and explicitly keep `missing` intact
            return [ismissing(x) ? missing : x for x in values]
        end
    catch e
        @error "Failed to normalize condition values: $e"
        return []  # Return empty in case of failure
    end
end

"""
    run_conflict_checks!(parms::Parameters, conflict_constraints::Dict{String, Constraint})

Run conflict checks for friends, enemies, and anchors.
All INCLUDED items are equivalent to friends.  While the test for EXLUDED as enemies does not match
their exact characteristics, it may be a useful approximation without adding more rules.
"""
function run_conflict_checks!(
        parms::Parameters, conflict_constraints::Dict{String, Constraint}
)
    # Find friends, enemies, and include constraints
    bank = parms.bank
    friends_constraints = find_all_constraints_by_type(conflict_constraints, "ALLORNONE")
    enemies_constraints = find_all_constraints_by_type(conflict_constraints, "ENEMIES")
    include_constraints = find_all_constraints_by_type(conflict_constraints, "INCLUDE")
    exclude_constraints = find_all_constraints_by_type(conflict_constraints, "EXCLUDE")
    friends_constraints = vcat(friends_constraints, include_constraints)
    enemies_constraints = vcat(enemies_constraints, exclude_constraints)

    # Check conflicts between friends and enemies
    try
        if !isempty(friends_constraints) && !isempty(enemies_constraints)
            for friends_constraint in friends_constraints
                @debug "Checking Friends Constraint: $(friends_constraint.id)"
                friends_values = normalize_condition_values(
                    friends_constraint.condition(bank)
                )
                for enemies_constraint in enemies_constraints
                    @debug "Checking Enemies Constraint: $(enemies_constraint.id)"
                    enemies_values = normalize_condition_values(
                        enemies_constraint.condition(bank)
                    )

                    # Apply one-to-many conflict rule between friends and enemies
                    conflict_df = apply_conflict_rule(
                        friends_values, enemies_values, one_to_many_conflict_rule
                    )

                    # Log the conflict if any were found
                    if !isempty(conflict_df)
                        log_conflicts("Friends-Enemies", conflict_df)
                    else
                        @debug "No conflicts found between Friends and Enemies."
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
            @debug "Running anchor tests conflict check."
            anchor_values = normalize_condition_values(bank.ANCHOR)

            # Check anchors against enemies
            for enemies_constraint in enemies_constraints
                @debug "Checking Anchor-Enemies Constraint: $(enemies_constraint.id)"
                enemies_values = normalize_condition_values(
                    enemies_constraint.condition(bank)
                )

                # Apply conflict rule between anchors and enemies
                conflict_df = apply_conflict_rule(
                    anchor_values, enemies_values, one_to_many_conflict_rule
                )

                if !isempty(conflict_df)
                    log_conflicts("Anchor-Enemies", conflict_df)
                else
                    @debug "No conflicts found between Anchor and Enemies."
                end
            end

            # Check anchors against friends
            for friends_constraint in friends_constraints
                @debug "Checking Anchor-Friends Constraint: $(friends_constraint.id)"
                friends_values = normalize_condition_values(
                    friends_constraint.condition(bank)
                )

                # Apply conflict rule between anchors and friends
                conflict_df = apply_conflict_rule(
                    friends_values, anchor_values, all_or_none_conflict_rule
                )

                if !isempty(conflict_df)
                    log_conflicts("Anchor-Friends", conflict_df)
                else
                    @debug "No conflicts found between Anchor and Friends."
                end
            end
        end
    catch e
        @warn "Conflict checks for Anchors-Friends/Enemies failed: $e"
    end
end

"""
    find_all_constraints_by_type(
        constraints::Dict{String, Constraint}, type::String

)::Vector{Constraint}

Helper function to retrieve all constraints by their type.
"""
function find_all_constraints_by_type(
        constraints::Dict{String, Constraint}, type::String
)::Vector{Constraint}
    return [constraint for constraint in values(constraints) if constraint.type == type]
end

"""
    log_conflicts(
        conflict_type::String, conflicting_rows::DataFrame, max_rows_to_log::Int = 5

)

Logs conflicts found in data.
"""
function log_conflicts(
        conflict_type::String, conflicting_rows::DataFrame, max_rows_to_log::Int = 5
)
    num_conflicts = size(conflicting_rows, 1)
    if num_conflicts > 0
        @warn "$num_conflicts conflicting $conflict_type found"
        @warn "Displaying the first $max_rows_to_log rows of the conflict:",
        first(conflicting_rows, max_rows_to_log)
    end
end

"""
    one_to_many_conflict_rule(values1::Vector, values2::Vector)::Vector{Tuple}

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
    conflicts = [key for key in keys(counts) if counts[key] > 1]
    if !isempty(conflicts)
        @info "One-to-Many conflict found: $conflicts"
    end
    return conflicts
end

"""
    all_or_none_conflict_rule(
        friends_values::Vector, anchor_values::Vector

)::Vector{Tuple}

Checks for all-or-none conflicts between two sets of values.
"""
function all_or_none_conflict_rule(
        friends_values::Vector, anchor_values::Vector
)::Vector{Tuple}
    group_dict = Dict{Any, Set{Any}}()
    for (friend, anchor) in zip(friends_values, anchor_values)
        if !ismissing(friend) && !ismissing(anchor)
            if !haskey(group_dict, friend)
                group_dict[friend] = Set{Any}()
            end
            push!(group_dict[friend], anchor)
        end
    end
    conflicts = [(friend, anchors)
                 for (friend, anchors) in group_dict if length(anchors) > 1]
    if !isempty(conflicts)
        @info "All-or-None conflict found: $conflicts"
    end
    return conflicts
end

"""
    apply_conflict_rule(values1::Vector, values2::Vector, rule::Function)::DataFrame

Applies a conflict rule between two sets of values and returns a DataFrame with conflicting pairs if any exist.
"""
function apply_conflict_rule(values1::Vector, values2::Vector, rule::Function)::DataFrame
    @debug "Applying conflict rule: $(rule)"
    conflicting_pairs = rule(values1, values2)

    if !isempty(conflicting_pairs)
        @info "Conflicting pairs found: $conflicting_pairs"
        return DataFrame(
            :col1 => map(x -> x[1], conflicting_pairs),
            :col2 => map(x -> x[2], conflicting_pairs)
        )
    else
        @debug "No conflicts found using rule $(rule)"
        return DataFrame()  # Return empty DataFrame if no conflicts are found
    end
end

end
