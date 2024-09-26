module SolverConfiguration

export configure_solver!

using JuMP
using CPLEX
using Cbc
using GLPK
using Gurobi
using HiGHS
using SCIP

using ..Configuration
using ..Utils

"""
    load_solver_config(toml_file::String) -> Dict

Reads and loads the solver configuration from a TOML file.
The configuration file should contain the settings for different solvers.

# Arguments

  - `toml_file::String`: The path to the TOML file containing solver configurations.

# Returns

  - A dictionary with the configuration data for different solvers.
"""
function load_solver_config(toml_file::String)
    config = safe_read_toml(toml_file)
    return config
end

"""
    configure_solver!(model::Model, parms::Parameters, solver_name::String="cplex") -> Model

Configures the specified optimization solver in JuMP based on the input solver name and the
settings in the `solver_config.toml` file. It supports solvers like CPLEX, CBC, GLPK, Gurobi, HiGHS,
and SCIP. It also sets various options, including the verbosity and time limits, for the solver.

# Arguments

  - `model::Model`: The optimization model created in JuMP.
  - `parms::Parameters`: The parameters struct which controls verbosity settings.
  - `solver_name::String`: The name of the solver to use (default: "cplex").

# Returns

  - The configured optimization model with the selected solver and options applied.
"""
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

    # Set common options from the config file
    for (opt, val) in solver_options
        try
            set_optimizer_attribute(model, String(opt), val)
        catch e
            @warn "Option $opt not supported by $solver_name solver. Error: $e"
        end
    end

    # Set global time limit and verbosity
    set_time_limit_sec(model, 120.0)
    parms.verbose <= 1 && set_silent(model)

    return model
end

"""
    conflicting_constraints(model::Model) -> Vector{ConstraintRef}

Identifies the constraints that are in conflict in the model. This feature is supported
by certain solvers (e.g., CPLEX) that offer conflict resolution capabilities. It retrieves
all constraints marked as `IN_CONFLICT` from the model.

# Arguments

  - `model::Model`: The optimization model for which conflicting constraints need to be analyzed.

# Returns

  - A vector containing references to the constraints that are in conflict.
"""
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

"""
    check_constraints(model::Model) -> Vector{ConstraintRef}

Runs a conflict analysis on the optimization model to identify conflicting constraints.
This function calls `compute_conflict!`, a feature supported by specific solvers like CPLEX,
to compute the infeasibility and check which constraints are causing the issue.

# Arguments

  - `model::Model`: The optimization model on which to run conflict analysis.

# Returns

  - A vector containing references to the constraints that are in conflict.
"""
function check_constraints(model)
    compute_conflict!(model)
    return conflicting_constraints(model)
end

end  # module SolverConfiguration
