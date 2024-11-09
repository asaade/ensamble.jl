module StatsFunctions

# Export the key functions to be used by other modules

export calc_tau,
       calc_info_tau,
       prob_3pl,
       info_3pl,
       observed_score_continuous,
       lw_dist,
       expected_score_item,
       expected_score_matrix,
       expected_info_matrix,
       calc_expected_scores_reference

using Base.Threads
using DataFrames
using Distributions
using FastGaussQuadrature
using QuadGK
using Random
using Statistics
using StatsFuns
using ThreadsX

# Probability function for the 3PL model
"""
    prob_3pl(a::Float64, b::Float64, c::Float64, θ::Float64; D::Float64 = 1.0)::Float64

TBW
"""
function prob_3pl(a::Float64, b::Float64, c::Float64, θ::Float64; D::Float64 = 1.0)::Float64
    exponent = D * a * (θ - b)
    p = c + (1.0 - c) / (1.0 + exp(-exponent))
    return p
end

"""
    prob_pcm(a::Float64, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0)::Vector{Float64}

Computes the probability of response in each category for the Partial Credit Model (PCM).

# Arguments

  - `a`: Discrimination parameter (scalar).
  - `b`: Step difficulty parameters (vector of length `K`).
  - `θ`: Ability parameter (scalar).
  - `D`: Scaling constant (default is 1.0).

# Returns

  - `prob_category`: Vector of probabilities for each response category (length `K + 1`).
"""
function prob_pcm(
        a::Float64, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0
)::Vector{Float64}
    # Compute increments for cumulative sum
    delta = D * a .* (θ .- b)  # Vector of length K

    # Compute cumulative sums with initial zero for category 0
    sum_terms = cumsum([0.0; delta])  # Vector of length K + 1

    # Compute numerators for each category
    numerators = exp.(sum_terms)

    # Compute denominator (sum of numerators)
    denominator = sum(numerators)

    # Compute probabilities for each category
    prob_category = numerators / denominator

    return prob_category
end

"""
    expected_score_pcm(a::Float64, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0)::Float64

Computes the expected score for the Partial Credit Model (PCM).

# Arguments

  - `a`: Discrimination parameter (scalar).
  - `b`: Step difficulty parameters (vector of length `K`).
  - `θ`: Ability parameter (scalar).
  - `D`: Scaling constant (default is 1.0).

# Returns

  - `expected_score`: Expected score for the item.
"""
function expected_score_pcm(
        a::Float64, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0
)::Float64
    # Compute category probabilities
    prob_category = prob_pcm(a, b, θ; D = D)
    # Scores are typically 0 to K
    scores = 0:length(b)
    # Compute expected score
    expected_score = sum(prob_category .* scores)
    return expected_score
end

"""
    prob_grm(a::Float64, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0)::Vector{Float64}

Computes the probability of response in each category for the Graded Response Model (GRM).

# Arguments

  - `a`: Discrimination parameter (scalar).
  - `b`: Threshold parameters (vector of length `K - 1`).
  - `θ`: Ability parameter (scalar).
  - `D`: Scaling constant (default is 1.0).

# Returns

  - `prob_category`: Vector of probabilities for each response category (length `K`).
"""
function prob_grm(
        a::Float64, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0
)::Vector{Float64}
    # Compute cumulative probabilities (P*)
    z = D * a .* (θ .- b)
    P_star = 1.0 ./ (1.0 .+ exp.(-z))
    P_star = vcat(0.0, P_star, 1.0)  # Add P*_0 = 0 and P*_K = 1

    # Compute category probabilities
    prob_category = diff(P_star)

    return prob_category
end

function prob_item(
        model::String,
        a,
        bs::Vector{Float64},
        c::Union{Nothing, Float64},
        θ::Float64;
        D::Float64 = 1.0
)::Float64
    if model in ("ThreePL", "THREEPL")
        return prob_3pl(a, bs[1], c, θ; D = D)
    elseif model == "PCM"
        return prob_pcm(a, bs, θ; D = D)
    elseif model == "GPCM"
        return prob_gpcm(a, bs, θ; D = D)
    elseif model in ("TwoPL", "TWOPL")
        return prob_3pl(a, bs[1], 1.0, θ; D = D)
    elseif model == "GRM"
        return prob_grm(a, bs, θ; D = D)
    else
        error("Unsupported model: $model")
    end
end

"""
    expected_score_grm(a::Float64, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0)::Float64

Computes the expected score for the Graded Response Model (GRM).

# Arguments

  - `a`: Discrimination parameter (scalar).
  - `b`: Threshold parameters (vector of length `K - 1`).
  - `θ`: Ability parameter (scalar).
  - `D`: Scaling constant (default is 1.0).

# Returns

  - `expected_score`: Expected score for the item.
"""
function expected_score_grm(
        a::Float64, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0
)::Float64
    # Compute category probabilities
    prob_category = prob_grm(a, b, θ; D = D)
    # Scores are typically 0 to K - 1
    scores = 0:length(b)
    # Compute expected score
    expected_score = sum(prob_category .* scores)
    return expected_score
end

"""
    prob_gpcm(a::Vector{Float64}, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0)::Vector{Float64}

Computes the probability of response in each category for the Generalized Partial Credit Model (GPCM).

# Arguments

  - `a`: Vector of discrimination parameters for each category transition (length `K`).
  - `b`: Vector of threshold parameters for each category transition (length `K`).
  - `θ`: Ability parameter (scalar).
  - `D`: Scaling constant (default is 1.0).

# Returns

  - `prob_category`: Vector of probabilities for each response category (length `K + 1`).
"""
function prob_gpcm(
        a::Vector{Float64}, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0
)::Vector{Float64}
    delta_theta = θ .- b
    sum_terms = cumsum(D .* a .* delta_theta)
    sum_terms = vcat(0.0, sum_terms)
    numerators = exp.(sum_terms)
    denominator = sum(numerators)
    prob_category = numerators / denominator
    return prob_category
end

"""
    expected_score_gpcm(a::Vector{Float64}, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0)::Float64

Computes the expected score for the Generalized Partial Credit Model (GPCM).

# Arguments

  - `a`: Vector of discrimination parameters for each category transition (length `K`).
  - `b`: Vector of threshold parameters for each category transition (length `K`).
  - `θ`: Ability parameter (scalar).
  - `D`: Scaling constant (default is 1.0).

# Returns

  - `expected_score`: Expected score for the item.
"""
function expected_score_gpcm(
        a::Vector{Float64}, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0
)::Float64
    # Compute category probabilities
    prob_category = prob_gpcm(a, b, θ; D = D)
    # Scores are tcypically 0 to K
    scores = 0:length(b)
    # Compute expected score
    expected_score = sum(prob_category .* scores)
    return expected_score
end

function expected_score_item(
        model::String,
        a::Float64,
        bs::Vector{Float64},
        c::Union{Nothing, Float64},
        θ::Float64;
        D::Float64 = 1.0
)::Float64
    # Determine the number of categories
    K = length(bs) + 1
    scores = 0:(K - 1)

    # Compute category probabilities based on the model
    if model == "3PL"
        p = prob_3pl(a, bs[1], c, θ; D = D)
        prob_category = [1 - p, p]
        scores = [0, 1]
    elseif model == "PCM"
        prob_category = prob_pcm(a, bs, θ; D = D)
    elseif model == "GPCM"
        prob_category = prob_gpcm(a, bs, θ; D = D)
    elseif model in ("TwoPL", "TWOPL")
        p = prob_3pl(a, bs[1], 0.0, θ; D = D)
        prob_category = [1 - p, p]
        scores = [0, 1]
    elseif model == "GRM"
        prob_category = prob_grm(a, bs, θ; D = D)
    else
        error("Unsupported model: $model")
    end

    # Compute expected score
    expected_score = sum(prob_category .* scores)
    return expected_score
end

function info_2pl(a::Float64, b::Float64, θ::Float64; D::Float64 = 1.0)::Float64
    exponent = D * a * (θ - b)
    p = 1.0 / (1.0 + exp(-exponent))
    q = 1.0 - p
    info = (D * a)^2 * p * q
    return info
end

# Information function for the 3PL model
function info_3pl(
        θ::Float64, b::Vector{Float64}, a::Vector{Float64},
        c::Vector{Float64}; D::Float64 = 1.0
)::Vector{Float64}
    p = prob_3pl.(a, b, c, θ; D = D)
    q = 1.0 .- p
    numerator = (D .* a) .^ 2 .* (p .- c) .^ 2 .* q
    denominator = ((1.0 .- c) .^ 2) .* p
    info = numerator ./ denominator
    return info
end

function info_3pl(a::Float64, b::Float64, c::Float64, θ::Float64; D::Float64 = 1.0)::Float64
    p = prob_3pl(a, b, c, θ; D = D)
    q = 1.0 - p
    numerator = (D * a * (p - c))^2 * q
    denominator = ((1.0 - c)^2) * p
    info = numerator / denominator
    return info
end

function info_grm(a::Float64, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0)::Float64
    # Compute cumulative probabilities (P*)
    z = D * a .* (θ .- b)
    P_star = 1.0 ./ (1.0 .+ exp.(-z))
    P_star = vcat(0.0, P_star, 1.0)  # P*_0 = 0, P*_K = 1

    # Compute derivatives of cumulative probabilities
    P_star_derivative = D * a .* P_star[2:(end - 1)] .* (1.0 .- P_star[2:(end - 1)])

    # Compute category probabilities (P_k)
    P_k = diff(P_star)

    # Compute item information components
    numerator = P_star_derivative .^ 2
    denominator = P_k[1:(end - 1)] .* P_k[2:end]
    info_components = numerator ./ denominator

    # Sum over all categories
    info = sum(info_components)

    return info
end

function info_pcm(a::Float64, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0)::Float64
    # Compute probabilities for each category
    prob = prob_pcm(a, b, θ; D = D)  # Vector of length K + 1

    # Categories from 0 to K
    categories = 0:length(b)  # Vector of length K + 1

    # Expected score E(X)
    expected_score = sum(prob .* categories)

    # Compute score differences
    score_diff = categories .- expected_score

    # Compute item information
    info = (D * a)^2 * sum(prob .* (score_diff .^ 2))

    return info
end

"""
    info_gpcm(a::Vector{Float64}, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0) -> Float64

Calculates the item information for the Generalized Partial Credit Model (GPCM).

# Arguments

  - `a`: Vector of discrimination parameters for each category transition (length `K`).
  - `b`: Vector of threshold parameters for each category transition (length `K`).
  - `θ`: Ability parameter (scalar).
  - `D`: Scaling constant (default is 1.0).

# Returns

  - `info`: The item information at ability level `θ`.
"""
function info_gpcm(
        a::Vector{Float64}, b::Vector{Float64}, θ::Float64; D::Float64 = 1.0
)::Float64
    # Compute probabilities for each category
    prob = prob_gpcm(a, b, θ; D = D)  # Vector of length K + 1

    # Compute cumulative discrimination parameters (η_k)
    eta_k = D * cumsum([0.0; a])  # Vector of length K + 1

    # Compute expected value of η (η̄)
    eta_bar = sum(prob .* eta_k)

    # Compute differences (η_k - η̄)
    eta_diff = eta_k .- eta_bar

    # Compute item information
    info = sum(prob .* (eta_diff .^ 2))

    return info
end

function info_item(
        model::String,
        a,
        bs::Vector{Float64},
        c::Union{Nothing, Float64},
        θ::Float64;
        D::Float64 = 1.0
)::Float64
    if model == "3PL"
        return info_3pl(a, bs[1], c, θ; D = D)
    elseif model == "PCM"
        return info_pcm(a, bs, θ; D = D)
    elseif model == "GPCM"
        return info_gpcm(a, bs, θ; D = D)
    elseif model == "2PL"
        return info_3pl(a, bs[1], 1.0, θ; D = D)
    elseif model == "GRM"
        return info_grm(a, bs, θ; D = D)
    else
        error("Unsupported model: $model")
    end
end

function kahan_sum(data::Vector{Float64})::Float64
    sum = 0.0
    c = 0.0  # A running compensation for lost low-order bits.

    for x in data
        y = x - c        # So far, c is zero in the first iteration.
        t = sum + y      # Add the compensated value.
        c = (t - sum) - y  # (t - sum) recovers the high-order part of y; subtracting y recovers the low-order part.
        sum = t          # New sum includes the low-order bits.
    end

    return sum
end

function expected_score_matrix(
        bank::DataFrame, θ_values::Vector{Float64}; D::Float64 = 1.0
)::Matrix{Float64}
    num_items = nrow(bank)
    num_thetas = length(θ_values)

    # Initialize the matrix to hold expected scores (num_items x num_thetas)
    score_matrix = zeros(Float64, num_items, num_thetas)

    # Parallelize the loop over items
    @threads for idx in 1:num_items
        @inbounds begin
            model = bank[idx, :MODEL_TYPE]
            a = bank[idx, :A]
            c = if model == "3PL"
                bank[idx, :C]
            else
                nothing
            end
            b_columns = filter(col -> occursin(r"^B\d*$|^B$", string(col)), names(bank))
            bs = [bank[idx, col] for col in b_columns if !ismissing(bank[idx, col])]

            # Calculate expected scores for all θ values using the expected_score_item
            expected_scores = map(
                θ -> expected_score_item(model, a, bs, c, θ; D = D), θ_values
            )

            score_matrix[idx, :] .= expected_scores
        end
    end

    return score_matrix
end

function expected_info_matrix(
        bank::DataFrame, θ_values::Vector{Float64}; D::Float64 = 1.0
)::Matrix{Float64}
    num_items = nrow(bank)
    num_thetas = length(θ_values)
    info_matrix = zeros(Float64, num_items, num_thetas)
    b_columns = filter(col -> occursin(r"^B\d*$", string(col)), names(bank))

    @threads for idx in 1:num_items
        model = bank.MODEL_TYPE[idx]
        a = bank.A[idx]
        c = bank.C[idx]

        bs = [bank[idx, col] for col in b_columns]
        bs = filter(!ismissing, bs)
        for (j, θ) in enumerate(θ_values)
            info_matrix[idx, j] = info_item(model, a, bs, c, θ; D = D)
        end
    end

    return info_matrix
end


function calc_tau(
    score_matrix::Matrix{Float64},
    r::Int,
    N::Int;
    num_simulations::Int = 250,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)::Matrix{Float64}
    """
    Calculate the tau matrix for given probabilities.

    # Arguments:
    - `score_matrix`: Probability matrix (Items × K).
    - `r`: Number of powers.
    - `N`: Sample size.
    - `num_simulations`: Number of simulations (default = 250).
    - `rng`: Random number generator (default = Random.GLOBAL_RNG).

    # Returns:
    - Tau matrix (R × K).
    """
    num_items, num_theta_points = size(score_matrix)
    tau = zeros(Float64, r, num_theta_points)

    if N > num_items
        error("Sample size N cannot exceed the number of items in the score_matrix.")
    end

    # Generate unique seeds for reproducibility
    seeds = rand(rng, UInt64, num_simulations)

    # Function to process a single simulation
    function process_simulation(seed)
        local_rng = MersenneTwister(seed)
        sampled_indices = randperm(local_rng, num_items)[1:N]  # Sample N items without replacement
        buffer = score_matrix[sampled_indices, :]  # N × K matrix
        tau_local = zeros(Float64, r, num_theta_points)
        for r_index in 1:r
            tau_local[r_index, :] = sum(buffer .^ r_index; dims = 1)[:]
        end
        return tau_local
    end

    # Process simulations in parallel
    simulations = ThreadsX.map(process_simulation, seeds)

    # Sum over all simulations
    for sim_tau in simulations
        tau .+= sim_tau
    end

    # Average the results
    tau ./= num_simulations

    return tau
end

using Random
using ThreadsX  # Make sure to add the ThreadsX package if not already installed

function calc_info_tau(
    info_matrix::Matrix{Float64},
    N::Int;
    num_batches::Int = 250,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    sample_with_replacement::Bool = false,
)::Vector{Float64}
    """
    Calculate the tau vector for item information functions by simulating sums over random samples.

    # Arguments:
    - `info_matrix`: Information matrix (Items × K), where each row represents an item and each column represents a theta point.
    - `N`: Number of items to sample in each batch (sample size).
    - `num_batches`: Number of batches to simulate (default = 250).
    - `rng`: Random number generator for reproducibility (default = `Random.GLOBAL_RNG`).
    - `sample_with_replacement`: Whether to sample items with replacement (default = `false`).

    # Returns:
    - `tau`: Tau information vector of length `K`, representing the average sum of item information across batches.
    """
    num_items, num_theta_points = size(info_matrix)
    tau = zeros(Float64, num_theta_points)

    # Error handling
    if N > num_items && !sample_with_replacement
        error("Sample size N cannot exceed the number of items when sampling without replacement.")
    end

    # Generate unique seeds for reproducibility
    seeds = rand(rng, UInt64, num_batches)

    # Function to process a single batch
    function process_batch(seed)
        local_rng = MersenneTwister(seed)
        if sample_with_replacement
            sampled_indices = rand(local_rng, 1:num_items, N)
        else
            sampled_indices = randperm(local_rng, num_items)[1:N]
        end
        buffer = info_matrix[sampled_indices, :]  # N × K matrix
        tau_local = sum(buffer; dims = 1)[:]      # Sum over items and flatten to 1D vector
        return tau_local
    end

    # Process batches in parallel
    batch_results = ThreadsX.map(process_batch, seeds)

    # Accumulate the results
    for batch_tau in batch_results
        tau .+= batch_tau
    end

    # Compute the average over batches
    tau ./= num_batches

    return tau
end


"""
    calc_expected_scores_reference(expected_scores::Matrix{Float64}, N::Int; num_forms::Int = 250)

Calculates the mean (`tau_mean`) and variance (`tau_var`) of expected scores for the reference form, either from user-provided values or by simulating test forms from the item bank.

# Arguments:

  - `expected_scores`: Matrix of expected scores (Items × Theta points), where each row corresponds to an item, and each column corresponds to a critical theta point.
  - `N`: Number of items to sample in each simulated form (sample size). If `N == 0`, the function calculates `tau_mean` and `tau_var` directly from `expected_scores`.
  - `num_forms`: Number of simulated test forms (default = 250).

# Returns:

  - `tau_mean`: Vector of mean expected scores for each theta point (size `k`).
  - `tau_var`: Vector of variance of expected scores for each theta point (size `k`).
"""
function calc_expected_scores_reference(
        expected_scores::Matrix{Float64},
        N::Int;
        num_forms::Int = 250
)
    num_items, _ = size(expected_scores)  # Number of items and theta points

    if N == 0
        # Direct calculation from provided expected scores
        tau_mean = vec(mean(expected_scores; dims = 1))
        tau_var = vec(var(expected_scores; dims = 1))
        return tau_mean, tau_var
    end

    # Generate unique seeds for RNGs
    rng = Random.MersenneTwister()
    seeds = rand(rng, UInt64, num_forms)

    function process_form(seed)
        rng = MersenneTwister(seed)
        if N <= num_items
            # Sample N unique items without replacement
            sampled_indices = randperm(rng, num_items)[1:N]
        else
            error("Sample size N cannot exceed the number of items in the expected_scores matrix.")
        end

        # Extract expected scores for sampled items
        sampled_scores = expected_scores[sampled_indices, :]  # Shape: (N, num_theta_points)

        # Calculate mean and variance across sampled items for each theta point
        form_mean = mean(sampled_scores; dims = 1)  # Shape: (1, num_theta_points)
        form_var = var(sampled_scores; dims = 1)    # Shape: (1, num_theta_points)
        return form_mean, form_var
    end

    # Process forms in parallel using ThreadsX
    results = ThreadsX.map(process_form, seeds)

    # Aggregate results
    form_means = vcat([res[1] for res in results]...)
    form_vars = vcat([res[2] for res in results]...)

    # Compute overall mean and variance
    tau_mean = vec(mean(form_means; dims = 1))
    tau_var = vec(mean(form_vars; dims = 1))

    return tau_mean, tau_var
end


"""
    observed_score_continuous(item_params::Matrix{Float64}, ability_dist::Normal;
                                   D::Float64=1.0, num_points::Int=120)::Vector{Float64}

Calculates the observed score distribution using numerical integration for 3PL (dichotomous) items.

# Arguments

  - `item_params`: Matrix of item parameters (a, b, c), size (num_items, 3).
  - `ability_dist`: Ability distribution (e.g., Normal).
  - `num_points`: Number of integration points (default 100).

# Returns

  - Observed score distribution as a vector.
"""
function observed_score_continuous(
        item_params::Matrix{Float64}, ability_dist::Normal; D::Float64 = 1.0, num_points::Int = 120
)::Vector{Float64}
    num_items = size(item_params, 1)

    # Define the integrand function for numerical integration
    function integrand(θ, x)
        score_dist = lw_dist(item_params, θ; D)  # Score distribution for dichotomous items
        if x + 1 > length(score_dist)
            return 0.0  # Avoid bounds error
        end
        return score_dist[x + 1] * pdf(ability_dist, θ)
    end

    # Calculate max possible score for dichotomous items
    max_score = num_items  # Each item contributes 1 score point

    # Initialize the observed distribution vector
    observed_dist = zeros(Float64, max_score + 1)

    # Iterate over possible observed scores (from 0 to max_score)
    for x in 0:max_score
        observed_dist[x + 1] = quadgk(θ -> integrand(θ, x), -Inf, Inf; order = num_points)[1]
    end

    return observed_dist
end

"""
    lw_dist(item_params::Matrix{Float64}, θ::Float64; D::Float64=1.0)::Vector{Float64}

Implementation of Lord and Wingersky Recursion Formula to calculate score distribution for dichotomous items.

# Arguments

  - `item_params`: Matrix of item parameters (a, b, c).
  - `θ`: Ability parameter.

# Returns

  - Score distribution as a vector.
"""
function lw_dist(
        item_params::Matrix{Float64}, θ::Float64; D::Float64 = 1.0)::Vector{Float64}
    num_items = size(item_params, 1)

    # Pre-allocate the result array for probabilities
    res = ones(Float64, num_items + 1)  # Initialize with 1 (for zero correct answers)

    # Loop over items and apply the recursion formula
    for i in 1:num_items
        a, b, c = item_params[i, :]  # Corrected: extract parameters from row `i`

        prob_correct = prob_3pl(a, b, c, θ; D)
        prob_incorrect = 1.0 - prob_correct

        # Create a new array for intermediate results
        prov = zeros(Float64, i + 1)

        # Update the first element
        prov[1] = prob_incorrect * res[1]

        # Apply the recursion formula for each subsequent element
        for j in 2:i
            prov[j] = prob_incorrect * res[j] + prob_correct * res[j - 1]
        end

        # Update the last element
        prov[i + 1] = prob_correct * res[i]

        # Update the result array with the new values
        res[1:(i + 1)] .= prov  # Broadcasting for efficient assignment
    end

    return res
end

end # module
