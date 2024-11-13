module ATALogger
using Logging

const LOG_LEVELS = Dict(
    "DEBUG" => Logging.Debug,
    "INFO" => Logging.Info,
    "WARN" => Logging.Warn,
    "ERROR" => Logging.Error
)

function setup_logger(level::String = "INFO", log_file::String = "ata.log")
    logger = SimpleLogger(
        open(log_file, "a"),
        get(LOG_LEVELS, uppercase(level), Logging.Info)
    )
    return global_logger(logger)
end

log_operation(operation::String, details::Dict) = @info "Operation: $operation" details...
end
