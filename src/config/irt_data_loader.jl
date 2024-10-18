module IRTDataLoader

export IRTModelData, load_irt_data

using DataFrames
using ..Utils

"""
    IRTModelData

Data structure for holding the IRT model data.
"""
mutable struct IRTModelData
    method::AbstractString          # Method (e.g., TCC, TIC, etc.)
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
function load_irt_data(config_data::Dict{Symbol, Any}, bank::DataFrame)::IRTModelData
    # Ensure IRT configuration exists
    if !haskey(config_data, :IRT)
        throw(ArgumentError("Configuration must contain the 'IRT' key."))
    end

    irt_dict = config_data[:IRT]
    method = get(irt_dict, :METHOD, missing)
    theta = get(irt_dict, :THETA, missing)
    D = get(irt_dict, :D, 1.0)
    r = get(irt_dict, :R, 2)
    N = get(config_data[:FORMS], :N, 1)

    relative_target_weights = get(irt_dict, :RELATIVETARGETWEIGHTS, [1.0, 1.0])
    relative_target_points = get(irt_dict, :RELATIVETARGETPOINTS, [-1.0, 1.0])
    if length(relative_target_weights) != length(relative_target_points)
        throw(ArgumentError("The length of 'RELATIVETARGETWEIGHTS' and 'RELATIVETARGETPOINTS' must match."))
    end

    # Extract item parameters directly from the bank DataFrame
    a = bank.A
    b = bank.B
    c = bank.C

    # Calculate 3D probability and information matrices for dichotomous items
    (p_matrix, info_matrix) = create_irt_item_data(theta, a, b, c, D)

    # Compute tau and tau_info using the calculated p_matrix and info_matrix
    tau = get_tau(irt_dict, p_matrix, r, length(theta), N)
    tau_info = get_tau_info(irt_dict, info_matrix, length(theta), N)

    # Return IRTModelData struct with all required data
    return IRTModelData(method, theta, p_matrix, info_matrix, tau, tau_info,
        relative_target_weights, relative_target_points, length(theta), r, D)
end


"""
    create_irt_item_data(theta::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}, c::Vector{Float64}, D::Float64)

Creates the probability and information matrices for dichotomous items only.
"""
function create_irt_item_data(theta::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}, c::Vector{Float64}, D::Float64)
    num_items = length(a)

    # Initialize 2D arrays for probabilities and information
    p_matrix = Matrix{Float64}(undef, num_items, length(theta))
    info_matrix = Matrix{Float64}(undef, num_items, length(theta))

    # Use direct element iteration
    for (t_idx, θ) in enumerate(theta)
        # Vectorized computation for probabilities and information for all items at this theta
        p_matrix[:, t_idx] = Probability(θ, b, a, c; d=D)
        info_matrix[:, t_idx] = Information(θ, b, a, c; d=D)
    end

    return (p_matrix, info_matrix)
end


"""
    get_tau(irt_dict::Dict{Symbol, Any}, p_matrix::Matrix{Float64}, r::Int, k::Int, N::Int)::Matrix{Float64}

Calculates or retrieves the tau values based on the probability matrix.
"""
function get_tau(irt_dict::Dict{Symbol, Any}, p_matrix::Matrix{Float64}, r::Int, k::Int, N::Int)::Matrix{Float64}
    tau = get(irt_dict, :TAU, nothing)

    if tau !== nothing && !isempty(tau)
        return hcat(tau...)  # Directly concatenate tau columns if provided
    end

    return calc_tau(p_matrix, r, k, N)
end

"""
    get_tau_info(irt_dict::Dict{Symbol, Any}, info_matrix::Matrix{Float64}, k::Int, N::Int)::Vector{Float64}

Calculates or retrieves the tau_info values based on the information matrix.
"""
function get_tau_info(irt_dict::Dict{Symbol, Any}, info_matrix::Matrix{Float64}, k::Int, N::Int)::Vector{Float64}
    tau_info = get(irt_dict, :TAU_INFO, nothing)

    if tau_info !== nothing && !isempty(tau_info)
        return Vector{Float64}(tau_info)
    end

    return calc_info_tau(info_matrix, k, N)
end

end # module IRTDataLoader
