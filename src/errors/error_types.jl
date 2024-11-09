module ATAErrors

export ATAError, ConfigError, ValidationError, OptimizationError, ConstraintError
export handle_error, log_error

using Logging

"""
Base error type for all ATA-specific errors
"""
abstract type ATAError <: Exception end

"""
Configuration-related errors
"""
struct ConfigError <: ATAError
    message::String
    component::String
    details::Any

    ConfigError(msg::String, comp::String, det=nothing) = new(msg, comp, det)
end

"""
Data validation errors
"""
struct ValidationError <: ATAError
    message::String
    context::String
    invalid_data::Any

    ValidationError(msg::String, ctx::String, data=nothing) = new(msg, ctx, data)
end

"""
Optimization-related errors
"""
struct OptimizationError <: ATAError
    message::String
    solver_status::Symbol
    model_info::Any

    OptimizationError(msg::String, status::Symbol, info=nothing) = new(msg, status, info)
end

"""
Constraint-related errors
"""
struct ConstraintError <: ATAError
    message::String
    constraint_id::String
    details::Any

    ConstraintError(msg::String, id::String, det=nothing) = new(msg, id, det)
end

"""
Handles error by logging and optionally rethrowing
"""
function handle_error(error::ATAError; rethrow::Bool=true)
    error_type = typeof(error)

    # Log error with context
    if error_type == ConfigError
        @error "Configuration Error: $(error.message)" component=error.component details=error.details
    elseif error_type == ValidationError
        @error "Validation Error: $(error.message)" context=error.context data=error.invalid_data
    elseif error_type == OptimizationError
        @error "Optimization Error: $(error.message)" status=error.solver_status info=error.model_info
    elseif error_type == ConstraintError
        @error "Constraint Error: $(error.message)" constraint=error.constraint_id details=error.details
    end

    # Optionally rethrow the error
    rethrow && throw(error)
end

"""
Logs an error with appropriate context
"""
function log_error(error::ATAError)
    error_type = typeof(error)
    error_context = Dict{String, Any}()

    if error_type == ConfigError
        error_context["component"] = error.component
        error_context["details"] = error.details
    elseif error_type == ValidationError
        error_context["context"] = error.context
        error_context["invalid_data"] = error.invalid_data
    elseif error_type == OptimizationError
        error_context["solver_status"] = error.solver_status
        error_context["model_info"] = error.model_info
    elseif error_type == ConstraintError
        error_context["constraint_id"] = error.constraint_id
        error_context["details"] = error.details
    end

    @error error.message error_context
end

end # module ATAErrors
