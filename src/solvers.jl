using JuMP
using YAML
using CPLEX
using Cbc
using SCIP
using GLPK
using HiGHS
using SYMPHONY_jll
using Gurobi

function load_solver_config(yaml_file::String)
    config = YAML.load_file(yaml_file)
    return config
end

function configure_solver!(model::Model, parameters::Params, solver_name::String="cplex")
    config = load_solver_config("data/solver_config.yaml")
    solver_options = config[solver_name]

    if solver_name == "cplex"
        set_optimizer(model, CPLEX.Optimizer)
    elseif solver_name == "cbc"
        set_optimizer(model, Cbc.Optimizer)
    elseif solver_name == "scip"
        set_optimizer(model, SCIP.Optimizer)
    elseif solver_name == "glpk"
        set_optimizer(model, GLPK.Optimizer)
    elseif solver_name == "highs"
        set_optimizer(model, HiGHS.Optimizer)
    elseif solver_name == "symphony"
        set_optimizer(model, SYMPHONY_jll.Optimizer)
    elseif solver_name == "gurobi"
        set_optimizer(model, Gurobi.Optimizer)
    else
        error("Unsupported solver: $solver_name")
    end

    # Set common options
    for (opt, val) in solver_options
        try
            set_optimizer_attribute(model, opt, val)
        catch e
            @warn "Option $opt not supported by $solver_name solver."
        end
    end

    parameters.verbose <= 1 && set_silent(model)

    return model
end

# # Example usage:

# model = Model()
# configure_solver!(model, parameters, "scip")
