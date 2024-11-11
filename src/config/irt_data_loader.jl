# IRTDataLoader.jl

module IRTDataLoader

using DocStringExtensions
using DataFrames
using LinearAlgebra
using StatsBase
using ..Utils
using ..ATAErrors
using ..ConfigValidation
using ..AssemblyConfigLoader
export IRTModelData, load_irt_data

# Custom error types
"""
Base error type for IRT-related errors
"""
abstract type IRTError <: ATAError end

struct IRTConfigError <: IRTError
    message::String
end

struct IRTCalculationError <: IRTError
    message::String
end

# Constants
const SUPPORTED_DICHOTOMOUS_MODELS = ["1PL", "2PL", "3PL"]
const SUPPORTED_POLYTOMOUS_MODELS = ["PCM", "GPCM", "GRM"]
const SUPPORTED_MODELS = vcat(SUPPORTED_DICHOTOMOUS_MODELS, SUPPORTED_POLYTOMOUS_MODELS)

"""
    IRTModelData

Mutable structure for holding IRT model data.

$(FIELDS)
"""
mutable struct IRTModelData
    method::String
    theta::Vector{Float64}
    score_matrix::Matrix{Float64}
    info_matrix::Matrix{Float64}
    tau::Matrix{Float64}
    tau_info::Vector{Float64}
    tau_mean::Vector{Float64}
    item_score_means::Vector{Float64}
    relative_target_weights::Vector
    relative_target_points::Vector{Float64}
    k::Int
    r::Int
    D::Float64

    # Inner constructor with validation
    function IRTModelData(
        method::String,
        theta::Vector{Float64},
        score_matrix::Matrix{Float64},
        info_matrix::Matrix{Float64},
        tau::Matrix{Float64},
        tau_info::Vector{Float64},
        tau_mean::Vector{Float64},
        item_score_means::Vector{Float64},
        relative_target_weights::Vector,
        relative_target_points::Vector{Float64},
        k::Int,
        r::Int,
        D::Float64
    )
        # Validate dimensions
        if size(score_matrix, 2) != length(theta)
            throw(IRTCalculationError("Score matrix rows must match theta values"))
        end
        if size(info_matrix) != size(score_matrix)
            throw(IRTCalculationError("Information matrix must match score matrix dimensions"))
        end
        if length(relative_target_weights) != length(relative_target_points)
            throw(IRTConfigError("Target weights and points must have same length"))
        end

        new(method, theta, score_matrix, info_matrix, tau, tau_info, tau_mean, item_score_means,
            relative_target_weights, relative_target_points, k, r, D)
    end
end


"""
    calculate_num_categories(bank::DataFrame)::Vector{Int}

Calculate the number of categories for all items in the bank.
"""
function calculate_num_categories(bank::DataFrame)::Vector{Int}
    num_items = nrow(bank)
    num_categories = Vector{Int}(undef, num_items)
    b_columns = filter(col -> occursin(r"^B\d*$|^B$", string(col)), names(bank))

    for idx in 1:num_items
        model_type = bank[idx, :MODEL_TYPE]

        if !(model_type in SUPPORTED_MODELS)
            throw(IRTModelError("Unsupported model type: $model_type"))
        end

        num_categories[idx] = if model_type in SUPPORTED_DICHOTOMOUS_MODELS
            2
        else
            bs = [bank[idx, col] for col in b_columns if !ismissing(bank[idx, col])]
            length(bs) + 1
        end
    end

    return num_categories
end

"""
    get_tau(irt_dict::Dict{Symbol, Any}, score_matrix::Matrix{Float64}, r::Int, N::Int)::Matrix{Float64}

Calculate or retrieve tau values.
"""
function get_tau(
    irt_dict::Dict{Symbol, Any},
    score_matrix::Matrix{Float64},
    r::Int,
    N::Int
)::Matrix{Float64}
    tau = get(irt_dict, :TAU, nothing)
    return tau !== nothing && !isempty(tau) ? hcat(tau...) : calc_tau(score_matrix, r, N)
end

"""
    get_tau_info(irt_dict::Dict{Symbol, Any}, info_matrix::Matrix{Float64}, N::Int)::Vector{Float64}

Calculate or retrieve tau information values.
"""
function get_tau_info(
    irt_dict::Dict{Symbol, Any},
    info_matrix::Matrix{Float64},
    N::Int
)::Vector{Float64}
    tau_info = get(irt_dict, :TAU_INFO, nothing)
    return tau_info !== nothing && !isempty(tau_info) ?
           Vector{Float64}(tau_info) : calc_info_tau(info_matrix, N)
end

"""
    load_irt_data(config_data::Dict{Symbol, Any}, forms_config, bank::DataFrame)::IRTModelData

Load and calculate IRT model data from configuration and item bank.
"""
function load_irt_data(
    config_data::Dict{Symbol, Any},
    forms_config::AssemblyConfig,
    bank::DataFrame
)::IRTModelData
    # Validate configuration
    irt_dict = config_data[:IRT] ## validate_irt_config(config_data)

    # Extract configuration parameters
    method = irt_dict[:METHOD]
    theta = irt_dict[:THETA]
    N = forms_config.form_size
    D = get(irt_dict, :D, 1.0)
    relative_target_weights = get(irt_dict, :RELATIVETARGETWEIGHTS, [1.0, 1.0])
    relative_target_points = get(irt_dict, :RELATIVETARGETPOINTS, [-1.0, 1.0])

    # Calculate matrices
    score_matrix = expected_score_matrix(bank, theta)
    info_matrix = expected_info_matrix(bank, theta)

    # Determine r value based on model types
    r = get(irt_dict, :R, 2)
    if any(x -> x ∉ SUPPORTED_DICHOTOMOUS_MODELS, bank.MODEL_TYPE)
        r = 1
    end

    # Ensure num_categories is present
    if !("NUM_CATEGORIES" in names(bank)) || isempty(bank.NUM_CATEGORIES) ||
       any(x -> ismissing(x), bank.NUM_CATEGORIES)
        bank.NUM_CATEGORIES = calculate_num_categories(bank)
    end

    # Calculate tau values
    tau = get_tau(irt_dict, score_matrix, r, N)
    tau_info = get_tau_info(irt_dict, info_matrix, N)
    tau_mean = tau[1, :]
    item_score_means = map(x -> mean(trim(x, prop=0.1)),  eachcol(score_matrix))

    return IRTModelData(
        method, theta, score_matrix, info_matrix, tau, tau_info,
        tau_mean, item_score_means, relative_target_weights, relative_target_points,
        length(theta), r, D
    )
end

end # module
