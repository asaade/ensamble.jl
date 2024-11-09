# Create a logger that logs to a file
struct FileLogger <: AbstractLogger
    io::IO
end

# Implement the custom logger (this can be expanded as needed)
function Logging.shouldlog(logger::FileLogger, level::LogLevel, _module, group, id)
    return level >= Logging.Warn
end

function Logging.handle_message(
        logger::FileLogger, level::LogLevel, message, _module, group, id, file, line; kwargs...
)
    return println(logger.io, "[$(Logging.level_string(level))] $message")
end

# Usage
open("logfile.log", "a") do io
    global_logger(FileLogger(io))
    @warn logger "This will be written to the logfile"
end
