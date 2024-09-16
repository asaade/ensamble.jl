module IRTDataLoader

using DataFrames
using ..StatsFunctions

export IRTModelData, load_irt_data

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
    load_irt_data(config_data::Dict{Symbol, Any}, bank::DataFrame)::IRTModelData

Loads and calculates IRT-related parameters like theta, p, tau, and information functions.
"""
function load_irt_data(config_data::Dict{Symbol, Any}, bank::DataFrame)::IRTModelData
    # Load IRT-related data from the config dictionary

    irt_dict = config_data[:IRT]

    method = irt_dict[:METHOD]

    # Only proceed if method is valid
    if method in ["TCC", "TIC", "TIC2", "TIC3"]
        theta = get(irt_dict, :THETA, missing)
        a, b, c = extract_item_params(bank)
        k = length(theta)
        D = Float64(irt_dict[:D])
        r = Int(irt_dict[:R])
        N = config_data[:FORMS][:N]

        relative_target_weights = get(irt_dict, :RELATIVETARGETWEIGHTS, [1.0, 1.0])
        relative_target_points = get(irt_dict, :RELATIVETARGETPOINTS, [-1.0, 1.0])

        # Calculate probability matrix p and information matrix
        p_matrix = calculate_probabilities(theta, a, b, c, D)
        info_matrix = calculate_information(theta, a, b, c, D)

        # Calculate tau and tau_info
        tau = get_tau(irt_dict, p_matrix, r, k, N)
        tau_info = get_tau_info(irt_dict, info_matrix, k, N)

        # Return the IRTModelData struct
        return IRTModelData(method,
                            theta,
                            p_matrix,
                            info_matrix,
                            tau,
                            tau_info,
                            relative_target_weights,
                            relative_target_points,
                            k,
                            r,
                            D)
    else
        error("Unsupported method: $method")
    end
    @info "Ended IRT configuration"
end


# Helper function to extract item parameters a, b, c from bank
function extract_item_params(bank::DataFrame)
    a::Vector{Float64} = bank[!, :A]
    b::Vector{Float64} = bank[!, :B]
    c::Vector{Float64} = bank[!, :C]
    return a, b, c
end

# Helper function to calculate the probability matrix p
function calculate_probabilities(theta::Vector{Float64}, a, b, c,
                                 D::Float64)::Matrix{Float64}
    p = [StatsFunctions.Probability(t, b, a, c; d = D) for t in theta]
    p_matrix = reduce(hcat, p)
    return Matrix{Float64}(p_matrix)
end

# Helper function to calculate the information matrix
function calculate_information(theta::Vector{Float64}, a::Vector{Float64},
                               b::Vector{Float64}, c::Vector{Float64},
                               D::Float64)::Matrix{Float64}
    info = [StatsFunctions.Information(t, b, a, c; d = D) for t in theta]
    info_matrix = reduce(hcat, info)
    return Matrix{Float64}(info_matrix)
end

# Helper function to calculate tau
function get_tau(irt_dict::Dict{Symbol, Any}, p_matrix::Matrix{Float64}, r::Int, k::Int,
                 N::Int)::Matrix{Float64}
    if haskey(irt_dict, :TAU) && length(irt_dict[:TAU]) > 0 && irt_dict[:TAU] != [[]]
        tau_vectors = irt_dict[:TAU]
        tau = reduce(hcat, tau_vectors)
        return Matrix{Float64}(tau)
    else
        return StatsFunctions.calc_tau(p_matrix, r, k, N)
    end
end

# Helper function to calculate tau_info
function get_tau_info(irt_dict::Dict{Symbol, Any}, info_matrix::Matrix{Float64}, k::Int,
                      N::Int)::Vector{Float64}
    if haskey(irt_dict, :TAU_INFO) && length(irt_dict[:TAU_INFO]) > 0 &&
       irt_dict[:TAU] != [[]]
        return Vector{Float64}(irt_dict[:TAU_INFO])
    else
        return StatsFunctions.calc_info_tau(info_matrix, k, N)
    end
end

end # module IRTDataLoader
