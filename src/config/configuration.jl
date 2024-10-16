module Configuration

# Export main configuration-related functions and types
export configure, Config, Parameters, Constraint

using DataFrames

using ..Utils

# Include all the necessary module files
include("config_loader.jl")    # Loads configuration file names
include("assembly_config_loader.jl")  # Loads and checks form configurations
include("bank_data_loader.jl")  # Loads and checks item bank and anchor test data
include("irt_data_loader.jl")  # Configures and calculates IRT model data

# Use modules in the current scope for Configuration
using .ConfigLoader
using .AssemblyConfigLoader
using .BankDataLoader
using .IRTDataLoader

# Define a struct for holding constraint information
struct Constraint
    id::AbstractString
    type::AbstractString
    condition::Function
    lb::Number
    ub::Number
end

# Define a struct for the configuration
struct Config
    forms::Dict{Symbol, Any}
    items_file::AbstractString
    anchor_items_file::Union{String, Missing}
    forms_file::AbstractString
    constraints_file::AbstractString
    results_file::AbstractString
    tcc_file::AbstractString
    plot_file::AbstractString
    solver::AbstractString
    verbose::Int
    report_categories::Vector{String}
    report_sums::Vector{String}
end

# Final struct for the parameters (types updated for consistency)
mutable struct Parameters
    n::Int                          # form_size from AssemblyConfig
    num_forms::Int                  # num_forms from AssemblyConfig
    max_items::Int                  # max_items from AssemblyConfig
    operational_items::Int          # operational_items from AssemblyConfig
    max_item_use::Int               # max_item_use from AssemblyConfig
    f::Int                          # f from AssemblyConfig
    shadow_test::Int                # shadow_test from AssemblyConfig
    bank::DataFrame                 # item bank DataFrame from BankDataLoader
    anchor_tests::Int               # anchor_tests from AssemblyConfig
    anchor_size::Int                # anchor_size from AssemblyConfig
    method::AbstractString          # method from IRTModelData
    theta::Union{Vector{Float64}, Nothing}  # theta from IRTModelData
    p::Union{Matrix{Float64}, Nothing}      # p matrix from IRTModelData
    info::Union{Matrix{Float64}, Nothing}   # info matrix from IRTModelData
    tau::Union{Matrix{Float64}, Nothing}    # tau matrix from IRTModelData
    tau_info::Union{Vector{Float64}, Nothing}  # tau_info from IRTModelData
    r::Int
    k::Int
    D::Float64
    relative_target_weights::Union{Vector{Float64}, Nothing}  # relative_target_weights from IRTModelData
    relative_target_points::Union{Vector{Float64}, Nothing}   # relative_target_points from IRTModelData
    verbose::Int                    # verbosity from BasicConfig
end


function clean_bank!(bank::DataFrame)
    bank.B_THRESHOLDS = replace(bank.B_THRESHOLDS, nothing => missing)
    return bank
end



"""
    transform_config_to_flat(basic_config::BasicConfig)::Config

Transform the nested NestedConfig structure into the flatter Config version.
"""
function transform_config_to_flat(basic_config::BasicConfig)::Config
    return Config(Dict{Symbol, Any}(),  # forms (placeholder, could be added)
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
                  basic_config.report_sums)
end

"""
    transform_parameters_to_flat(forms_config::AssemblyConfig, irt_data::IRTModelData, bank::DataFrame, flat_config::Config)::Parameters

Transform the necessary components (forms_config, irt_data, bank, flat_config)
into the flat Parameters structure.
"""
function transform_parameters_to_flat(forms_config::AssemblyConfig,
                                      irt_data::IRTModelData,
                                      bank::DataFrame,
                                      flat_config::Config)::Parameters
    # Transform to the flat Parameters structure
    return Parameters(forms_config.form_size,             # n
                      forms_config.num_forms,             # num_forms
                      forms_config.max_items,             # max_items
                      forms_config.operational_items,     # operational_items
                      forms_config.max_item_use,          # max_item_use
                      forms_config.f,                     # f
                      forms_config.shadow_test,           # shadow_test
                      clean_bank!(bank),                               # bank
                      forms_config.anchor_tests,          # anchor_tests
                      forms_config.anchor_size,           # anchor_size
                      irt_data.method,                    # method
                      irt_data.theta,                     # theta (Vector{Float64})
                      irt_data.p,                         # p (Matrix{Float64})
                      irt_data.info,                      # info (Matrix{Float64})
                      irt_data.tau,                       # tau (Matrix{Float64})
                      irt_data.tau_info,                  # tau_info (Vector{Float64})
                      irt_data.r,
                      irt_data.k,
                      irt_data.D,
                      irt_data.relative_target_weights,   # relative_target_weights (Vector{Float64})
                      irt_data.relative_target_points,    # relative_target_points (Vector{Float64})
                      flat_config.verbose)
end

"""
    configure(inFile::String = "data/config.toml")::Parameters

Entry point function that loads and prepares the system configuration.
Reads from a TOML configuration file, loads data, and returns the system parameters.
"""
function configure(inFile::String="data/config.toml")::Tuple{Config, Parameters}
    # Read TOML configuration
    config_data = upcaseKeys(safe_read_toml(inFile))

    # Load configuration, forms, item bank, and IRT data
    basic_config::BasicConfig = load_config(config_data)
    forms_config::AssemblyConfig = load_assembly_config(config_data)
    bank::DataFrame = read_bank_file(basic_config.items_file,
                                     basic_config.anchor_items_file)
    irt_data::IRTModelData = load_irt_data(config_data, bank)

    # Transform the nested configuration to the flat Config struct
    flat_config = transform_config_to_flat(basic_config)

    # Transform and return the flat Parameters struct
    return (flat_config,
            transform_parameters_to_flat(forms_config, irt_data, bank, flat_config))
end

end # module Configuration
