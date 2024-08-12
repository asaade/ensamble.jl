using JuMP

include("constraints.jl")
include("constraint_reader.jl")


# Initialize the optimization model
function initialize_model!(model::Model, parameters::Params, constraints::Dict{String, Constraint})
    println("Initializing optimization model...")
    num_items = size(parameters.bank, 1)
    num_forms = parameters.num_forms + (parameters.shadow_test_size > 0 ? 1 : 0)

    @variable(model, y >= 0.0)
    @variable(model, x[1:num_items, 1:num_forms], Bin)
    @objective(model, Min, y)

    apply_constraints!(model, parameters, constraints)
    write_to_file(model, "data/model.lp")

    return model
end

# Add constraints to the model
function apply_constraints!(model::Model, parameters::Params, constraints::Dict{String, Constraint})
    apply_objective!(model, parameters)
    constraint_prevent_overlap!(model, parameters)
    constraint_add_anchor!(model, parameters)

    for (constraint_id, constraint) in constraints
        println("Applying constraint: ", constraint_id)
        apply_individual_constraint!(model, parameters, constraint)
    end

    return model
end

# Apply objective function based on the method
function apply_objective!(model::Model, parameters::Params)
    if parameters.method == "TCC"
        objective_match_characteristic_curve!(model, parameters)
    elseif parameters.method == "TC"
        objective_match_items(model, parameters)
    else
        error("Unknown method: ", parameters.method)
    end
end

# Helper function to apply individual constraints
function apply_individual_constraint!(model::Model, parameters::Params, constraint::Constraint)
    lb, ub = constraint.lb, constraint.ub
    bank = parameters.bank
    items = 1:size(bank, 1)

    if constraint.type == "TEST"
        constraint_items_per_version(model, parameters, lb, ub)
    elseif constraint.type == "NUMBER"
        condition = Base.invokelatest(constraint.condition, bank)
        selected_items = items[condition]
        constraint_item_count(model, parameters, selected_items, lb, ub)
    elseif constraint.type == "SUM"
        item_vals = Base.invokelatest(constraint.condition, bank)
        constraint_item_sum(model, parameters, item_vals, lb, ub)
    else
        error("Unknown constraint type: ", constraint.type)
    end
end
