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
    score_matrix::Matrix{Float64}       # 3D array: (items, theta, categories) for probability
    info_matrix::Matrix{Float64}    # 3D array: (items, theta, categories) for information
    tau::Matrix{Float64}            # Tau mean (aggregated expected scores for versions)
    tau_info::Vector{Float64}       # Tau info (aggregated information at theta points)
    tau_mean::Vector{Float64}       # Tau mean (aggregated expected scores for versions)
    tau_var::Vector{Float64}        # Tau variance (expected scores var)
    relative_target_weights::Vector{Float64}  # Weights for relative target points
    relative_target_points::Vector{Float64}   # Target theta points for equating tests
    k::Int                          # Number of theta points
    r::Int                          # Number of powers for tau calculation
    D::Float64                      # Scaling constant for IRT models
end

"""
    load_irt_data(config_data::Dict{Symbol, Any}, bank::DataFrame) -> IRTModelData

Loads and calculates IRT-related parameters like theta, score_matrix, tau, and information matrices
based on the input configuration and item bank.

# Arguments

  - `config_data::Dict{Symbol, Any}`: Configuration dictionary containing IRT-related settings.
  - `bank::DataFrame`: DataFrame with item parameters (`A`, `B`, `C`) used for IRT calculations.

# Returns

  - An `IRTModelData` struct containing IRT parameters and matrices for the assembly process.
"""
function load_irt_data(
        config_data::Dict{Symbol, Any}, forms_config, bank::DataFrame
)::IRTModelData
    # Ensure IRT configuration exists
    if !haskey(config_data, :IRT)
        throw(ArgumentError("Configuration must contain the 'IRT' key."))
    end

    irt_dict = config_data[:IRT]
    method = get(irt_dict, :METHOD, missing)
    theta = get(irt_dict, :THETA, [-1.8, -0.5, 0.0, 0.5, 1.8])
    N = forms_config.form_size # get(config_data[:FORMS], :N, 1)

    relative_target_weights = get(irt_dict, :RELATIVETARGETWEIGHTS, [1.0, 1.0])
    relative_target_points = get(irt_dict, :RELATIVETARGETPOINTS, [-1.0, 1.0])
    if length(relative_target_weights) != length(relative_target_points)
        throw(
            ArgumentError(
            "The length of 'RELATIVETARGETWEIGHTS' and 'RELATIVETARGETPOINTS' must match.",
        ),
        )
    end

    # Calculate 2D expected scores and information matrices.
    D = get(irt_dict, :D, 1.0)
    score_matrix = expected_score_matrix(bank, theta)
    info_matrix = expected_info_matrix(bank, theta)

    # Only for dichotomous items
    # the expected scores are the same as probabilities and can be used in the local equating
    # model (using R powers of these probabilities). For the rest, the method cannot be applied so
    # R = 1.
    r = get(irt_dict, :R, 2)
    if any(x -> x ∉ ["1PL", "2PL", "3PL"], bank[:, "MODEL_TYPE"])
        r = 1
    end

    # Check if num_categories is already in Bank
    if !("NUM_CATEGORIES" in names(bank)) || isempty(bank.NUM_CATEGORIES) || any(x -> ismissing(x), bank.NUM_CATEGORIES)
        # Compute num_categories and add to Parameters
        bank.NUM_CATEGORIES = calculate_num_categories(bank)
    end

    # Compute tau and tau_info using the calculated score_matrix and info_matrix
    tau = get_tau(irt_dict, score_matrix, r, N)
    tau_info = get_tau_info(irt_dict, info_matrix, N)

    tau_mean, tau_var = calc_expected_scores_reference!(score_matrix, N)

    # Return IRTModelData struct with all required data
    return IRTModelData(
        method,
        theta,
        score_matrix,
        info_matrix,
        tau,
        tau_info,
        tau_mean,
        tau_var,
        relative_target_weights,
        relative_target_points,
        length(theta),
        r,
        D
    )
end

# Helper function to calculate num_categories if missing
# Calculate the number of categories from item parameters in the item bank
# Calculate the number of categories for all items in the bank
function calculate_num_categories(bank::DataFrame)::Vector{Int}
    num_items = nrow(bank)
    num_categories = Vector{Int}(undef, num_items)

    # Identify B parameter columns (e.g., B, B1, B2, B3...)
    b_columns = filter(col -> occursin(r"^B\d*$|^B$", string(col)), names(bank))

    # Calculate categories for each item based on model type
    for idx in 1:num_items
        model_type = bank[idx, :MODEL_TYPE]

        num_categories[idx] = if model_type in ["3PL", "2PL", "1PL"]
            2  # Dichotomous models (2 categories)
        elseif model_type in ["PCM", "GPCM", "GRM"]
            # For polytomous models, count the number of non-missing thresholds in B columns
            bs = [bank[idx, col] for col in b_columns if !ismissing(bank[idx, col])]
            length(bs) + 1  # Number of categories is thresholds count + 1
        else
            error("Unsupported model type: $model_type")
        end
    end

    return num_categories
end



"""
    get_tau(irt_dict::Dict{Symbol, Any}, score_matrix::Matrix{Float64}, r::Int, k::Int, N::Int)::Matrix{Float64}

Calculates or retrieves the tau values based on the probability matrix.
"""
function get_tau(
        irt_dict::Dict{Symbol, Any}, score_matrix::Matrix{Float64}, r::Int, N::Int
)::Matrix{Float64}
    tau = get(irt_dict, :TAU, nothing)

    if tau !== nothing && !isempty(tau)
        return hcat(tau...)  # Directly concatenate tau columns if provided
    end

    return calc_tau(score_matrix, r, N)
end

"""
    get_tau_info(irt_dict::Dict{Symbol, Any}, info_matrix::Matrix{Float64}, k::Int, N::Int)::Vector{Float64}

Calculates or retrieves the tau_info values based on the information matrix.
"""
function get_tau_info(
        irt_dict::Dict{Symbol, Any}, info_matrix::Matrix{Float64}, N::Int
)::Vector{Float64}
    tau_info = get(irt_dict, :TAU_INFO, nothing)

    if tau_info !== nothing && !isempty(tau_info)
        return Vector{Float64}(tau_info)
    end

    return calc_info_tau(info_matrix, N)
end

end # module IRTDataLoader
