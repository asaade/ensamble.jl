module IRTDataLoader

export IRTModelData, load_irt_data

using DataFrames
using ..Utils

"""
    IRTModelData

Data structure for holding the IRT model data.
"""
mutable struct IRTModelData
    method::String          # Method (e.g., TCC, TIC, etc.)
    theta::Vector{Float64}          # Theta points (ability levels)
    p_matrix::Matrix{Float64}     # 3D array: (items, theta, categories) for probability
    info_matrix::Matrix{Float64}  # 3D array: (items, theta, categories) for information
    tau::Matrix{Float64}            # Tau matrix (aggregated probabilities or information)
    tau_info::Vector{Float64}       # Tau info (aggregated information at theta points)
    relative_target_weights::Vector{Float64}  # Weights for relative target points
    relative_target_points::Vector{Float64}   # Target theta points for equating tests
    k::Int                          # Number of theta points
    r::Int                          # Number of powers for tau calculation
    D::Float64                      # Scaling constant for IRT models
end

"""
    load_irt_data(config_data::Dict{Symbol, Any}, bank::DataFrame) -> IRTModelData

Loads and calculates IRT-related parameters like theta, p_matrix, tau, and information matrices
based on the input configuration and item bank.

# Arguments

  - `config_data::Dict{Symbol, Any}`: Configuration dictionary containing IRT-related settings.
  - `bank::DataFrame`: DataFrame with item parameters (`A`, `B`, `C`) used for IRT calculations.

# Returns

  - An `IRTModelData` struct containing IRT parameters and matrices for the assembly process.
"""
function load_irt_data(config_data::Dict{Symbol, Any}, forms_config,
                       bank::DataFrame)::IRTModelData
    # Ensure IRT configuration exists
    if !haskey(config_data, :IRT)
        throw(ArgumentError("Configuration must contain the 'IRT' key."))
    end

    irt_dict = config_data[:IRT]
    method = get(irt_dict, :METHOD, missing)
    theta = get(irt_dict, :THETA, [-1.8, -0.5, 0.0, 0.5, 1.8])
    D = get(irt_dict, :D, 1.0)
    r = get(irt_dict, :R, 2)
    if any(x->x ∉ ["1PL", "2PL", "3PL"], bank[:, "MODEL_TYPE"])
        r = 1
    end
    N = forms_config.form_size # get(config_data[:FORMS], :N, 1)

    relative_target_weights = get(irt_dict, :RELATIVETARGETWEIGHTS, [1.0, 1.0])
    relative_target_points = get(irt_dict, :RELATIVETARGETPOINTS, [-1.0, 1.0])
    if length(relative_target_weights) != length(relative_target_points)
        throw(ArgumentError("The length of 'RELATIVETARGETWEIGHTS' and 'RELATIVETARGETPOINTS' must match."))
    end

    # Calculate 3D probability and information matrices for dichotomous items
    p_matrix = expected_score_matrix(bank, theta)
    info_matrix = expected_info_matrix(bank, theta)

    # Compute tau and tau_info using the calculated p_matrix and info_matrix
    tau = get_tau(irt_dict, p_matrix, r, N)
    tau_info = get_tau_info(irt_dict, info_matrix, N)

    # Return IRTModelData struct with all required data
    return IRTModelData(method, theta, p_matrix, info_matrix, tau, tau_info,
                        relative_target_weights, relative_target_points, length(theta), r,
                        D)
end

"""
    get_tau(irt_dict::Dict{Symbol, Any}, p_matrix::Matrix{Float64}, r::Int, k::Int, N::Int)::Matrix{Float64}

Calculates or retrieves the tau values based on the probability matrix.
"""
function get_tau(irt_dict::Dict{Symbol, Any}, p_matrix::Matrix{Float64}, r::Int,
                 N::Int)::Matrix{Float64}
    tau = get(irt_dict, :TAU, nothing)

    if tau !== nothing && !isempty(tau)
        return hcat(tau...)  # Directly concatenate tau columns if provided
    end

    return calc_tau(p_matrix, r, N)
end

"""
    get_tau_info(irt_dict::Dict{Symbol, Any}, info_matrix::Matrix{Float64}, k::Int, N::Int)::Vector{Float64}

Calculates or retrieves the tau_info values based on the information matrix.
"""
function get_tau_info(irt_dict::Dict{Symbol, Any}, info_matrix::Matrix{Float64},
                      N::Int)::Vector{Float64}
    tau_info = get(irt_dict, :TAU_INFO, nothing)

    if tau_info !== nothing && !isempty(tau_info)
        return Vector{Float64}(tau_info)
    end

    return calc_info_tau(info_matrix, N)
end

end # module IRTDataLoader
