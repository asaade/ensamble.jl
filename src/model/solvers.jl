using JuMP
using CPLEX
using Cbc
using GLPK
using Gurobi
using HiGHS
using SCIP

function load_solver_config(toml_file::String)
    config = safe_read_toml(toml_file)
    return config
end

function configure_solver!(model::Model, parms::Parameters, solver_name::String="cplex")
    parms.verbose > 1 && @info "Configuring $solver_name solver."
    config = load_solver_config("data/solver_config.toml")
    solver_options = config[lowercase(solver_name)]

    name = upSymbol(solver_name)

    if name == :CPLEX
        set_optimizer(model, CPLEX.Optimizer)
    elseif name == :CBC
        set_optimizer(model, Cbc.Optimizer)
    elseif name == :SCIP
        set_optimizer(model, SCIP.Optimizer)
    elseif name == :GLPK
        set_optimizer(model, GLPK.Optimizer)
    elseif name == :HIGHS
        set_optimizer(model, HiGHS.Optimizer)
        set_time_limit_sec(model, 120)
    elseif name == :GUROBI
        set_optimizer(model, Gurobi.Optimizer)
    else
        error("Unsupported solver: $name")
    end

    # Set common options
    for (opt, val) in solver_options
        try
            set_optimizer_attribute(model, String(opt), val)
        catch e
            @warn "Option $opt not supported by $solver_name solver."
        end
    end

    set_time_limit_sec(model, 120.0)
    parms.verbose <= 1 && set_silent(model)

    return model
end

function conflicting_constraints(model)
    list_of_conflicting_constraints = ConstraintRef[]

    for (F, S) in list_of_constraint_types(model)
        for con in all_constraints(model, F, S)
            if get_attribute(con, MOI.ConstraintConflictStatus()) == MOI.IN_CONFLICT
                push!(list_of_conflicting_constraints, con)
            end
        end
    end

    return list_of_conflicting_constraints
end

function check_constraints(model)
    compute_conflict!(model)

    # if get_attribute(model, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
    #     iis_model, _ = copy_conflict(model)
    #     print(iis_model)
    # end

    return conflicting_constraints(model)
end

# # Example usage:

# model = Model()
# configure_solver!(model, parms, "scip")
