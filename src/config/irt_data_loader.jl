module IRTDataLoader

export IRTModelData, load_irt_data

using DataFrames

using ..Utils

"""
Define the IRTModelData struct to hold the precalculated IRT data
"""
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

    k = length(theta)
    D = Float64(get(irt_dict, :D, 1.0))
    r = Int(get(irt_dict, :R, 2))
    N = get(config_data[:FORMS], :N, 1)

    # Validate target weights and points
    relative_target_weights = get(irt_dict, :RELATIVETARGETWEIGHTS, [1.0, 1.0])
    relative_target_points = get(irt_dict, :RELATIVETARGETPOINTS, [-1.0, 1.0])

    if length(relative_target_weights) != length(relative_target_points)
        throw(ArgumentError("The length of 'RELATIVETARGETWEIGHTS' and 'RELATIVETARGETPOINTS' must match."))
    end

    # Extract item parameters, taking into account both dichotomous and polytomous items
    a, b, c, b_thresholds, model_type = extract_item_params(bank)

    # Validate model types
    valid_models = ["3PL", "dichotomous", "GRM", "PCM", "GPCM"]
    if any(mt -> !(mt in valid_models), model_type)
        throw(ArgumentError("Invalid model type found in item bank. Supported types are: 3PL, dichotomous, GRM, PCM, GPCM."))
    end

    # Calculate probability and information matrices
    p_matrix = calculate_probabilities(theta, b, a, c, b_thresholds, model_type, D)
    info_matrix = calculate_information(theta, b, a, c, b_thresholds, model_type, D)

    # Calculate tau and tau_info
    tau = get_tau(irt_dict, p_matrix, r, k, N)
    tau_info = get_tau_info(irt_dict, info_matrix, k, N)

    return IRTModelData(method, theta, p_matrix, info_matrix, tau, tau_info,
                        relative_target_weights, relative_target_points, k, r, D)
end



"""
    extract_item_params(bank::DataFrame) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Vector{Float64}}, Vector{AbstractString}}

Extracts the item parameters (A, B, C) and thresholds (for polytomous items) from the item bank.
"""
function extract_item_params(bank::DataFrame)
    for param in [:A, :B]
        if !(param in upSymbol.(names(bank)))
            throw(ArgumentError("Missing required column '$param' in item bank."))
        end
    end

    a = bank.A
    b = bank.B
    c = "C" in names(bank) ? bank.C : fill(0.0, size(bank, 1))  # Optional guessing param

    # Ensure b_thresholds is a vector of vectors or missing values
    b_thresholds = bank.B_THRESHOLDS
    if !isa(b_thresholds, Vector{Vector{Union{Float64, Missing}}})
        throw(ArgumentError("B_THRESHOLDS must be a vector of vectors or missing values."))
    end

    model_type = bank.MODEL_TYPE

    return a, b, c, b_thresholds, model_type
end




"""
    calculate_probabilities(theta::Vector{Float64},
                    b::Vector{Float64},
                    a::Vector{Float64},
                    c::Vector{Float64},
                    b_thresholds,
                    model_type::Vector{AbstractString},
                    D::Float64) -> Matrix{Float64}

Calculates the probability matrix (p) for both dichotomous and polytomous items.
"""
function calculate_probabilities(theta::Vector{Float64},
                                 b::Vector{Float64},
                                 a::Vector{Float64},
                                 c::Vector{Float64},
                                 b_thresholds,
                                 model_type,
                                 D::Float64)::Matrix{Float64}
    num_items = length(a)
    num_theta = length(theta)
    p_matrix = Matrix{Float64}(undef, num_items, num_theta)

    for i in 1:num_items
        if model_type[i] == "3PL"  # Dichotomous
            p_matrix[i, :] = Probability.(theta, b[i], a[i], c[i]; d=D)
        elseif model_type[i] == "GRM"
            if !ismissing(b_thresholds[i])
                p_matrix[i, :] = prob_grm(a[i], b_thresholds[i], theta)
            else
                throw(ArgumentError("Missing threshold values for item $(i) in GRM model."))
            end
        elseif model_type[i] == "PCM"
            if !ismissing(b_thresholds[i])
                p_matrix[i, :] = prob_pcm(a[i], b_thresholds[i], theta)
            else
                throw(ArgumentError("Missing threshold values for item $(i) in PCM model."))
            end
        elseif model_type[i] == "GPCM"
            if !ismissing(b_thresholds[i])
                p_matrix[i, :] = prob_gpcm(a[i], b_thresholds[i], theta)
            else
                throw(ArgumentError("Missing threshold values for item $(i) in GPCM model."))
            end
        else
            throw(ArgumentError("Unsupported model type: $(model_type[i])"))
        end
    end

    return p_matrix
end


"""
    calculate_information(theta::Vector{Float64},
                           b::Vector{Float64},
                           a::Vector{Float64},
                           c::Vector{Float64},
                           b_thresholds,
                           model_type::Vector{AbstractString},
                           D::Float64) -> Matrix{Float64}

Calculates the information matrix for both dichotomous and polytomous items.
"""
function calculate_information(theta::Vector{Float64},
                               b::Vector{Float64},
                               a::Vector{Float64},
                               c::Vector{Float64},
                               b_thresholds,
                               model_type,
                               D::Float64)::Matrix{Float64}
    num_items = length(a)
    num_theta = length(theta)
    info_matrix = Matrix{Float64}(undef, num_theta, num_items)  # Matrix: theta points x items

    for t in 1:num_theta
        for i in 1:num_items
            θ = theta[t]
            if model_type[i] == "3PL"  # Dichotomous
                info_matrix[t, i] = Information(θ, b[i], a[i], c[i]; d=D)
            elseif model_type[i] == "GRM"
                if !ismissing(b_thresholds[i])
                    info_matrix[t, i] = info_grm(a[i], b_thresholds[i], θ)
                else
                    throw(ArgumentError("Missing threshold values for item $(i) in GRM model."))
                end
            elseif model_type[i] == "PCM"
                if !ismissing(b_thresholds[i])
                    info_matrix[t, i] = info_pcm(a[i], b_thresholds[i], θ)
                else
                    throw(ArgumentError("Missing threshold values for item $(i) in PCM model."))
                end
            elseif model_type[i] == "GPCM"
                if !ismissing(b_thresholds[i])
                    info_matrix[t, i] = info_gpcm(a[i], b_thresholds[i], θ)
                else
                    throw(ArgumentError("Missing threshold values for item $(i) in GPCM model."))
                end
            else
                throw(ArgumentError("Unsupported model type: $(model_type[i])"))
            end
        end
    end

    return info_matrix'
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
