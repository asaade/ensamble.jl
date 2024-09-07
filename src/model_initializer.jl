using CSV
using DataFrames
using JuMP

# Include external modules
include("CriteriaParser.jl")
include("constraints.jl")
include("types.jl")
include("utils.jl")

# Module constants
const APPLYING_CONSTRAINT_MESSAGE = "Applying constraint: "
const INITIALIZING_MODEL_MESSAGE = "Initializing optimization model..."
const MODEL_FILE = "./data/model.lp"

"""
    read_constraints(file_path::String) -> Dict{String, Constraint}

Read constraints from a CSV file, returning a dictionary of Constraint objects.
"""
function read_constraints(file_path::String)
    df = CSV.read(file_path, DataFrame; missingstring = nothing)
    constraints = Dict{String, Constraint}()

    for row in eachrow(df)
        if row[:ONOFF] == "ON"
            row = map(up!, row)
            cond_id = row[:CONSTRAINT_ID]
            type = row[:TYPE]
            condition_expr = row[:CONDITION]
            lb = row[:LB]
            ub = row[:UB]
            condition = if strip(condition_expr) == ""
                Meta.parse("df -> true")
            else
                CriteriaParser.parse_criteria(condition_expr)
            end
            constraints[cond_id] = Constraint(cond_id, type, eval(condition), lb, ub)
        end
    end

    return constraints
end

"""
    initialize_model!(model::Model, parms::Parameters, constraints::Dict{String, Constraint})

Initialize the optimization model, adding variables, the objective function,
and constraints based on the provided parms.
"""
function initialize_model!(model::Model,
                           parms::Parameters,
                           constraints::Dict{String, Constraint})
    println(INITIALIZING_MODEL_MESSAGE)
    num_items = size(parms.bank, 1)
    num_forms = parms.num_forms + (parms.shadow_test > 0 ? 1 : 0)

    @variable(model, y>=0.0)
    @variable(model, x[1:num_items, 1:num_forms], Bin)

    set_objective!(model, parms)
    apply_constraints!(model, parms, constraints)
    write_to_file(model, MODEL_FILE)

    return model
end

"""
    apply_constraints!(model::Model, parms::Parameters, constraints::Dict{String, Constraint})

Apply the constraints from the configuration to the optimization model.
"""
function apply_constraints!(model::Model,
                            parms::Parameters,
                            constraints::Dict{String, Constraint})
    apply_objective!(model, parms)
    constraint_add_anchor!(model, parms)

    for (constraint_id, constraint) in constraints
        parms.verbose > 1 && println(APPLYING_CONSTRAINT_MESSAGE, constraint_id)
        apply_individual_constraint!(model, parms, constraint)
    end

    return model
end

"""
    set_objective!(model::Model, parms::Parameters)

Set the objective function based on the method provided in parms.
"""
function set_objective!(model::Model, parms::Parameters)
    y = model[:y]
    if parms.method in ["TIC2", "TIC3"]
        @objective(model, Max, y)
    else
        @objective(model, Min, y)
    end
end

"""
    apply_objective!(model::Model, parms::Parameters, original_parms)

Apply the objective function specific to the method being used.
"""
function apply_objective!(model::Model, parms::Parameters)
    if parms.method in ["TCC"]
        objective_match_characteristic_curve!(model, parms)
    elseif parms.method in ["TIC"]
        objective_match_information_curve!(model, parms)
    elseif parms.method == "TIC2"
        objective_max_info(model, parms)
    elseif parms.method == "TIC3"
        objective_info_relative2(model, parms)
    elseif parms.method == "TC"
        println("Method TC not implemented yet.")
    else
        error("Unknown method: ", parms.method)
    end
end

"""
    apply_individual_constraint!(model::Model, parms::Parameters, constraint::Constraint)

Apply an individual constraint to the model based on the constraint type.
"""
function apply_individual_constraint!(model::Model, parms::Parameters,
                                      constraint::Constraint)
    lb, ub = constraint.lb, constraint.ub
    bank = parms.bank
    items = 1:size(bank, 1)

    if constraint.type == "TEST"
        constraint_items_per_form(model, parms, lb, ub)
    elseif constraint.type == "NUMBER"
        condition = constraint.condition(bank)
        selected_items = items[condition]
        constraint_item_count(model, parms, selected_items, lb, ub)
    elseif constraint.type == "SUM"
        item_vals = constraint.condition(bank)
        constraint_item_sum(model, parms, item_vals, lb, ub)
    elseif constraint.type == "ENEMIES"
        condition = constraint.condition(bank)
        constraint_enemies_in_form(model, parms, condition)
    elseif constraint.type == "FRIENDS"
        condition = constraint.condition(bank)
        constraint_friends_in_form(model, parms, condition)
    elseif constraint.type == "MAXUSE"
        constraint_max_use(model, parms, ub)
    elseif constraint.type == "OVERLAP"
        constraint_forms_overlap(model, parms, lb, ub)
    else
        error("Unknown constraint type: ", constraint.type)
    end
end
