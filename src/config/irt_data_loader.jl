module IRTDataLoader

export IRTModelData, load_irt_data

using DataFrames

using ..Utils

# Define the IRTModelData struct
mutable struct IRTModelData
    method::AbstractString
    theta::Vector{Float64}
    p::Matrix{Float64}
    info::Matrix{Float64}
    tau::Matrix{Float64}
    tau_info::Vector{Float64}
    relative_target_weights::Vector{Float64}
    relative_target_points::Vector{Float64}
    k::Int
    r::Int
    D::Float64
end

"""
    load_irt_data(config_data::Dict{Symbol, Any}, bank::DataFrame) -> IRTModelData

Loads and calculates IRT-related parameters like theta, p, tau, and information matrices
based on the input configuration and item bank.

# Arguments

  - `config_data::Dict{Symbol, Any}`: Configuration dictionary containing IRT-related settings.
  - `bank::DataFrame`: DataFrame with item parameters (`A`, `B`, `C`) used for IRT calculations.

# Returns

  - An `IRTModelData` struct containing IRT parameters and matrices for the assembly process.

# Throws

  - `ArgumentError` if required fields are missing or invalid in the configuration or bank.    # Check if the required IRT configuration exists
"""
function load_irt_data(config_data::Dict{Symbol, Any}, bank::DataFrame)::IRTModelData
    # Check if the required IRT configuration exists
    if !haskey(config_data, :IRT)
        throw(ArgumentError("Configuration must contain the 'IRT' key."))
    end
    irt_dict = config_data[:IRT]

    # Validate method
    method = get(irt_dict, :METHOD, missing)
    if ismissing(method) || !(method in ["TCC", "TIC", "TIC2", "TIC3"])
        throw(ArgumentError("Invalid or missing 'METHOD' in IRT configuration. Supported methods are: TCC, TIC, TIC2, TIC3."))
    end

    # Validate theta
    theta = get(irt_dict, :THETA, missing)
    if theta === missing || !isa(theta, Vector{Float64})
        throw(ArgumentError("Invalid or missing 'THETA'. It must be a vector of Float64 values."))
    end

    # Validate the item bank and extract item parameters
    a, b, c = extract_item_params(bank)

    k = length(theta)
    D = Float64(get(irt_dict, :D, 1.0))
    r = Int(get(irt_dict, :R, 1))
    N = get(config_data[:FORMS], :N, 1)

    # Validate target weights and points
    relative_target_weights = get(irt_dict, :RELATIVETARGETWEIGHTS, [1.0, 1.0])
    relative_target_points = get(irt_dict, :RELATIVETARGETPOINTS, [-1.0, 1.0])

    if length(relative_target_weights) != length(relative_target_points)
        throw(ArgumentError("The length of 'RELATIVETARGETWEIGHTS' and 'RELATIVETARGETPOINTS' must match."))
    end

    # Calculate probability matrix (p) and information matrix
    p_matrix = calculate_probabilities(theta, a, b, c, D)
    info_matrix = calculate_information(theta, a, b, c, D)

    # Calculate tau and tau_info
    tau = get_tau(irt_dict, p_matrix, r, k, N)
    tau_info = get_tau_info(irt_dict, info_matrix, k, N)

    return IRTModelData(method, theta, p_matrix, info_matrix, tau, tau_info,
                        relative_target_weights, relative_target_points, k, r, D)
end

"""
    extract_item_params(bank::DataFrame) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}

Extracts the item parameters (a, b, c) from the item bank. Throws an error if required columns are missing.

# Throws

  - `ArgumentError` if the required item parameters (A, B, C) are missing in the bank DataFrame.
"""
function extract_item_params(bank::DataFrame)
    for param in [:A, :B, :C]
        if !(param in upSymbol.(names(bank)))
            throw(ArgumentError("Missing required column '$param' in item bank."))
        end
    end

    a = bank[!, :A]
    b = bank[!, :B]
    c = bank[!, :C]

    return a, b, c
end

"""
    calculate_probabilities(theta::Vector{Float64}, a::Vector{Float64},
                            b::Vector{Float64}, c::Vector{Float64}, D::Float64) -> Matrix{Float64}

Calculates the probability matrix (p) at each theta point using IRT models.

# Throws

  - `ArgumentError` if the lengths of the parameter vectors (a, b, c) do not match.
"""
function calculate_probabilities(theta::Vector{Float64}, a::Vector{Float64},
                                 b::Vector{Float64}, c::Vector{Float64},
                                 D::Float64)::Matrix{Float64}
    if !(length(a) == length(b) == length(c))
        throw(ArgumentError("Length of item parameters (a, b, c) must be the same."))
    end

    p = [Probability(t, b, a, c; d=D) for t in theta]
    return hcat(p...)  # Concatenate vectors horizontally
end

"""
    calculate_information(theta::Vector{Float64}, a::Vector{Float64},
                          b::Vector{Float64}, c::Vector{Float64}, D::Float64) -> Matrix{Float64}

Calculates the information matrix at each theta point using IRT models.
"""
function calculate_information(theta::Vector{Float64}, a::Vector{Float64},
                               b::Vector{Float64}, c::Vector{Float64},
                               D::Float64)::Matrix{Float64}
    if !(length(a) == length(b) == length(c))
        throw(ArgumentError("Length of item parameters (a, b, c) must be the same."))
    end

    info = [Information(t, b, a, c; d=D) for t in theta]
    return hcat(info...)  # Concatenate vectors horizontally
end

"""
    get_tau(irt_dict::Dict{Symbol, Any}, p_matrix::Matrix{Float64}, r::Int, k::Int, N::Int) -> Matrix{Float64}

Retrieves or calculates the tau matrix based on the probability matrix (p). If the values for tau are
provided in the config, it is used; otherwise, it is calculated.
"""
function get_tau(irt_dict::Dict{Symbol, Any}, p_matrix::Matrix{Float64}, r::Int, k::Int,
                 N::Int)::Matrix{Float64}
    tau = get(irt_dict, :TAU_INFO, nothing)

    if tau !== nothing && !isempty(tau)
        return hcat(tau...)  # Directly concatenate tau columns
    end

    return calc_tau(p_matrix, r, k, N)
end

"""
    get_tau_info(irt_dict::Dict{Symbol, Any}, info_matrix::Matrix{Float64}, k::Int, N::Int) -> Vector{Float64}

Retrieves or calculates the tau_info vector based on the information matrix. If tau_info is provided
in the config, it is used; otherwise, it is calculated.
"""
function get_tau_info(irt_dict::Dict{Symbol, Any}, info_matrix::Matrix{Float64}, k::Int,
                      N::Int)::Vector{Float64}
    tau = get(irt_dict, :TAU_INFO, nothing)

    if tau !== nothing && !isempty(tau)
        return Vector{Float64}(tau)
    end

    return calc_info_tau(info_matrix, k, N)
end

end # module IRTDataLoader
