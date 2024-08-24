using JuMP
using CSV
using DataFrames

# Include external modules
include("CriteriaParser.jl")
include("constraints.jl")
include("types.jl")
include("utils.jl")

# Module constants
const MODEL_FILE = "./data/model.mps"
const INITIALIZING_MODEL_MESSAGE = "Initializing optimization model..."
const APPLYING_CONSTRAINT_MESSAGE = "Applying constraint: "

"""
    read_constraints(file_path::String) -> Dict{String, Constraint}

Read constraints from a CSV file, returning a dictionary of Constraint objects.
"""
function read_constraints(file_path::String)
    df = CSV.read(file_path, DataFrame; missingstring=nothing)
    constraints = Dict{String,Constraint}()

    for row in eachrow(df)
        if row[:ONOFF] != "OFF"
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
    initialize_model!(model::Model, parameters::Params, constraints::Dict{String, Constraint})

Initialize the optimization model, adding variables, the objective function,
and constraints based on the provided parameters.
"""
function initialize_model!(model::Model,
                           parameters::Params,
                           original_parameters::Params,
                           constraints::Dict{String,Constraint})
    println(INITIALIZING_MODEL_MESSAGE)
    num_items = size(parameters.bank, 1)
    num_forms = parameters.num_forms + (parameters.shadow_test > 0 ? 1 : 0)

    @variable(model, y >= 0.0)
    @variable(model, x[1:num_items, 1:num_forms], Bin)

    set_objective!(model, parameters)
    apply_constraints!(model, parameters, original_parameters, constraints)
    write_to_file(model, MODEL_FILE)

    return model
end

"""
    apply_constraints!(model::Model, parameters::Params, constraints::Dict{String, Constraint})

Apply the constraints from the configuration to the optimization model.
"""
function apply_constraints!(model::Model,
                            parameters::Params,
                            original_parameters::Params,
                            constraints::Dict{String,Constraint})
    apply_objective!(model, parameters, original_parameters)
    constraint_prevent_overlap!(model, parameters)
    constraint_add_anchor!(model, parameters)

    for (constraint_id, constraint) in constraints
        parameters.verbose > 1 && println(APPLYING_CONSTRAINT_MESSAGE, constraint_id)
        apply_individual_constraint!(model, parameters, constraint)
    end

    return model
end

"""
    set_objective!(model::Model, parameters::Params)

Set the objective function based on the method provided in parameters.
"""
function set_objective!(model::Model, parameters::Params)
    y = model[:y]
    if parameters.method in ["TIC2", "TIC3"]
        @objective(model, Max, y)
    else
        @objective(model, Min, y)
    end
end

"""
    apply_objective!(model::Model, parameters::Params, original_parameters)

Apply the objective function specific to the method being used.
"""
function apply_objective!(model::Model, parameters::Params, original_parameters::Params)
    if parameters.method in ["TCC"]
        objective_match_characteristic_curve!(model, parameters)
    elseif parameters.method in ["TIC"]
        objective_match_information_curve!(model, parameters)
    elseif parameters.method == "TIC2"
        objective_max_info(model, parameters)
    elseif parameters.method == "TIC3"
        objective_info_relative2(model, parameters, original_parameters)
    elseif parameters.method == "TC"
        objective_match_items(model, parameters)
    else
        error("Unknown method: ", parameters.method)
    end
end

"""
    apply_individual_constraint!(model::Model, parameters::Params, constraint::Constraint)

Apply an individual constraint to the model based on the constraint type.
"""
function apply_individual_constraint!(model::Model, parameters::Params,
                                      constraint::Constraint)
    lb, ub = constraint.lb, constraint.ub
    bank = parameters.bank
    items = 1:size(bank, 1)

    if constraint.type == "TEST"
        constraint_items_per_form(model, parameters, lb, ub)
    elseif constraint.type == "NUMBER"
        condition = constraint.condition(bank)
        selected_items = items[condition]
        constraint_item_count(model, parameters, selected_items, lb, ub)
    elseif constraint.type == "SUM"
        item_vals = constraint.condition(bank)
        constraint_item_sum(model, parameters, item_vals, lb, ub)
    elseif constraint.type == "ENEMIES"
        condition = constraint.condition(bank)
        #  selected_items = items[condition]
        constraint_enemies_in_form(model, parameters, condition)
    elseif constraint.type == "FRIENDS"
        condition = constraint.condition(bank)
        #  selected_items = items[condition]
        constraint_friends_in_form(model, parameters, condition)
    else
        error("Unknown constraint type: ", constraint.type)
    end
end