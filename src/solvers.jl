using JuMP

ENV["CPLEX_STUDIO_BINARIES"] = "/home/asaade/.bin/ILOG/CPLEX_Studio/cplex/bin/x86-64_linux/"
using CPLEX
function CPLEX_solver(model)
    println("Configuring solver CPLEX");
    set_optimizer(model, CPLEX.Optimizer);
    set_attribute(model, "CPX_PARAM_EPAGAP", 0.05);
    set_attribute(model, "CPX_PARAM_EPGAP", 0.05);
    set_attribute(model, "CPX_PARAM_EPINT", 1e-6);
    set_attribute(model, "CPX_PARAM_PREIND", 1);
    set_time_limit_sec(model, 90);

    return model
end


using Cbc
function CBC_solver(model)
    println("Configuring solver Cbc");
    set_optimizer(model, Cbc.Optimizer);
    set_attribute(model, "seconds", "90");
    set_attribute(model, "allowableGap", 0.05);
    set_attribute(model, "ratioGap", 0.05);
    set_attribute(model, "threads", 10);
    set_attribute(model, "timeMode", "elapsed");
    set_attribute(model, "logLevel", 1)
    set_time_limit_sec(model, 120);
    return model
end

function select_solver(model, solverType=:Cbc, verbose=false)
    if (solverType == :CPLEX)
        model = CPLEX_solver(model)
    elseif solverType == :Cbc
        model = CBC_solver(model)
    else
        println("Unknown solver type");
    end
    # if !verbose
    set_silent(model)
    # end
    return model
end
