module Configuration

export configure, Config, Parameters, Constraint

using DataFrames

using ..ATAErrors
using ..ConfigValidation
using ..Utils

# Include submodules
include("config_loader.jl")
include("assembly_config_loader.jl")
include("bank_data_loader.jl")
include("irt_data_loader.jl")

# Use modules in the current scope for Configuration
using .ConfigLoader
using .AssemblyConfigLoader
using .BankDataLoader
using .IRTDataLoader

# Define a struct for holding constraint information
struct Constraint
    id::String
    type::String
    condition::Function
    lb::Number
    ub::Number
end

# Struct for the configuration
struct Config
    forms::Dict{Symbol, Any}
    items_file::String
    anchor_items_file::Union{String, Missing}
    forms_file::String
    constraints_file::String
    results_file::String
    tcc_file::String
    plot_file::String
    solver::String
    verbose::Int
    report_categories::Vector{String}
    report_sums::Vector{String}
end

# Final struct for the parameters
mutable struct Parameters
    n::Int
    num_forms::Int
    max_items::Int
    max_item_use::Int
    f::Int
    shadow_test_size::Int
    bank::DataFrame
    anchor_tests::Int
    anchor_size::Int
    method::String
    theta::Union{Vector{Float64}, Nothing}
    score_matrix::Matrix{Float64}
    info_matrix::Matrix{Float64}
    tau::Matrix{Float64}
    tau_info::Vector{Float64}
    tau_mean::Vector{Float64}
    tau_var::Vector{Float64}
    r::Int
    k::Int
    D::Float64
    relative_target_weights::Union{Vector{Float64}, Nothing}
    relative_target_points::Union{Vector{Float64}, Nothing}
    verbose::Int
end

"""
    transform_config_to_flat(basic_config::BasicConfig)::Config

Transform the nested NestedConfig structure into the flatter Config version.
"""
function transform_config_to_flat(basic_config::BasicConfig)::Config
    return Config(
        Dict{Symbol, Any}(),  # forms (placeholder, could be added)
        basic_config.items_file,
        basic_config.anchor_items_file,
        basic_config.forms_file,
        basic_config.constraints_file,
        basic_config.results_file,
        basic_config.tcc_file,
        basic_config.plot_file,
        basic_config.solver,
        basic_config.verbose,
        basic_config.report_categories,
        basic_config.report_sums
    )
end

"""
    transform_parameters_to_flat(forms_config::AssemblyConfig, irt_data::IRTModelData, bank::DataFrame, flat_config::Config)::Parameters

Transform the necessary components (forms_config, irt_data, bank, flat_config)
into the flat Parameters structure.
"""
function transform_parameters_to_flat(
        forms_config::AssemblyConfig,
        irt_data::IRTModelData,
        bank::DataFrame,
        flat_config::Config
)::Parameters
    # Transform to the flat Parameters structure
    return Parameters(
        forms_config.form_size,             # n
        forms_config.num_forms,             # num_forms
        forms_config.max_items,             # max_items
        forms_config.max_item_use,          # max_item_use
        forms_config.f,                     # f
        forms_config.shadow_test_size,           # shadow_test_size
        bank,                               # bank
        forms_config.anchor_tests,          # anchor_tests
        forms_config.anchor_size,           # anchor_size
        irt_data.method,                    # method
        irt_data.theta,                     # theta (Vector{Float64})
        irt_data.score_matrix,                  # p (Matrix{Float64})
        irt_data.info_matrix,               # info (Matrix{Float64})
        irt_data.tau,                       # tau (Matrix{Float64})
        irt_data.tau_info,                  # tau_info (Vector{Float64})
        irt_data.tau_mean,
        irt_data.tau_var,
        irt_data.r,
        irt_data.k,
        irt_data.D,
        irt_data.relative_target_weights,   # relative_target_weights (Vector{Float64})
        irt_data.relative_target_points,    # relative_target_points (Vector{Float64})
        flat_config.verbose
    )
end

"""
    configure(inFile::String = "data/config.toml")::Parameters

Entry point function that loads and prepares the system configuration.
Reads from a TOML configuration file, loads data, and returns the system parameters.
"""
function configure(config_file::String = "data/config.toml")::Tuple{Config,Parameters}
    try
        # Load and validate TOML configuration
        config_data::Dict{Symbol, Any} = upcaseKeys(safe_read_toml(config_file))
        validate_config_data(config_data)

        # Load configurations
        basic_config = load_config(config_data)
        forms_config = load_assembly_config(config_data)

        # Load and validate data
        bank = read_bank_file(basic_config.items_file, basic_config.anchor_items_file)
        irt_data = load_irt_data(config_data, forms_config, bank)

        # Transform to flat structures
        flat_config = transform_config_to_flat(basic_config)
        parameters = transform_parameters_to_flat(forms_config, irt_data, bank, flat_config)

        return (flat_config, parameters)

    catch e
        if e isa ATAError
            rethrow(e)
        else
            throw(ConfigError(
                "Configuration failed",
                config_file,
                e
            ))
        end
    end
end

end # module Configuration
