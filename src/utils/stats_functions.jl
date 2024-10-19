module StatsFunctions

# Export the key functions to be used by other modules
export Probability,
       Information, calc_tau, calc_info_tau,
       observed_score_continuous, lw_dist,
       polytomous_probabilities, calc_info_tau_polytomous,
       calc_tau_polytomous

using Random, Distributions, QuadGK

"""
    Probability(θ::Float64, b::Float64, a::Float64, c::Float64; d::Float64 = 1.0) -> Float64

Calculates the probability of success given the IRT parameters and ability level θ.

# Arguments

  - `θ`: The ability parameter.
  - `b`: The difficulty parameter.
  - `a`: The discrimination parameter.
  - `c`: The guessing parameter.
  - `d`: Scaling constant (default is 1.0).

# Returns

  - Probability of success as `Float64`.
"""
function Probability(θ::Float64, b::Float64, a::Float64, c::Float64;
                     d::Float64=1.0)::Float64
    return c + (1 - c) / (1 + exp(-d * a * (θ - b)))
end

"""
    Probability(θ::Float64, b::Vector{Float64}, a::Vector{Float64}, c::Vector{Float64}; d::Float64 = 1.0) -> Vector{Float64}

Calculates the probability of success for a vector of items at a given ability level.

# Arguments

  - `θ`: The ability parameter.
  - `a`: Vector of discrimination parameters.
  - `b`: Vector of difficulty parameters.
  - `c`: Vector of guessing parameters.
  - `d`: Scaling constant (default is 1.0).

# Returns

  - A vector of probabilities of success for each item.
"""
function Probability(θ::Float64, b::Vector{Float64}, a::Vector{Float64}, c::Vector{Float64};
                     d::Float64=1.0)::Vector{Float64}
    return c .+ (1 .- c) ./ (1 .+ exp.(-d .* a .* (θ .- b)))
end

"""
    Information(θ::Float64, b::Vector{Float64}, a::Vector{Float64}, c::Vector{Float64}; d::Float64 = 1.0) -> Vector{Float64}

Calculates the item information function for a vector of items.

# Arguments

  - `θ`: The ability parameter.
  - `b`: Vector of difficulty parameters.
  - `a`: Vector of discrimination parameters.
  - `c`: Vector of guessing parameters.
  - `d`: Scaling constant (default is 1.0).

# Returns

  - A vector of item information values.
"""
function Information(θ::Float64, b::Vector{Float64}, a::Vector{Float64}, c::Vector{Float64};
                     d::Float64=1.0)::Vector{Float64}
    p = Probability.(θ, b, a, c; d=d)
    q = 1 .- p
    return (d .* a) .^ 2 .* (p .- c) .^ 2 .* q ./ ((1 .- c) .^ 2 .* p)
end

# Scalar version of Information function
function Information(θ::Float64, b::Float64, a::Float64, c::Float64;
                     d::Float64=1.0)::Float64
    p = Probability(θ, b, a, c; d=d)
    q = 1 - p
    return (d * a)^2 * (p - c)^2 * q / ((1 - c)^2 * p)
end

function prob_grm(a::Float64, b::Vector{Float64}, θ::Float64)
    K = length(b) + 1  # Number of categories
    prob = zeros(Float64, K)

    # Compute probabilities for response categories
    for k in 1:(K - 1)
        prob[k] = 1.0 / (1.0 + exp(-a * (θ - b[k])))
    end

    # For the last category, probability is 1
    prob[K] = 1.0

    # Convert to probability of response in category
    prob_category = zeros(Float64, K)
    prob_category[1] = 1 - prob[1]
    for k in 2:(K - 1)
        prob_category[k] = prob[k - 1] - prob[k]
    end
    prob_category[K] = prob[K - 1]

    return prob_category
end

function info_grm(a::Float64, b::Vector{Float64}, θ::Float64)
    K = length(b) + 1  # Number of categories
    prob = prob_grm(a, b, θ)
    info = 0.0

    # Calculate item information
    for k in 1:(K - 1)
        p = prob[k]
        info += a^2 * p * (1 - p)
    end

    return info
end

function prob_pcm(a::Float64, b::Vector{Float64}, θ::Float64)
    K = length(b)  # Number of categories
    numerators = zeros(Float64, K + 1)

    # Compute numerator for each category
    for k in 0:K
        sum_term = sum(a * (θ - b[j]) for j in 1:k)
        numerators[k + 1] = exp(sum_term)
    end

    # Compute denominator (sum of numerators)
    denominator = sum(numerators)

    # Compute probabilities for each category
    prob_category = numerators / denominator
    return prob_category
end

function info_pcm(a::Float64, b::Vector{Float64}, θ::Float64)
    K = length(b)  # Number of categories
    prob = prob_pcm(a, b, θ)
    info = 0.0

    # Calculate item information
    for k in 1:(K + 1)
        p = prob[k]
        log_derivative = a * (k - sum(prob .* (0:K)))  # Gradient of log-probability
        info += p * log_derivative^2
    end

    return info
end

function prob_gpcm(a::Vector{Float64}, b::Vector{Float64}, θ::Float64)
    K = length(b)  # Number of categories
    numerators = zeros(Float64, K + 1)

    # Compute numerator for each category
    for k in 0:K
        sum_term = sum(a[j] * (θ - b[j]) for j in 1:k)
        numerators[k + 1] = exp(sum_term)
    end

    # Compute denominator (sum of numerators)
    denominator = sum(numerators)

    # Compute probabilities for each category
    prob_category = numerators / denominator
    return prob_category
end

"""
The information for GPCM is computed similarly to PCM, but each category
k has its own discrimination parameter, so the information calculation takes this into account:
"""
function info_gpcm(a::Vector{Float64}, b::Vector{Float64}, θ::Float64)
    K = length(b)  # Number of categories
    prob = prob_gpcm(a, b, θ)
    info = 0.0

    # Calculate item information
    for k in 1:(K + 1)
        p = prob[k]
        log_derivative = a[k] * (k - sum(prob .* (0:K)))  # Gradient of log-probability
        info += p * log_derivative^2
    end

    return info
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

function calc_tau(p_matrix::Matrix{Float64}, r::Int, k::Int, N::Int)::Matrix{Float64}
    """
    Calculate the tau matrix for given probabilities.

    # Arguments:
    - `p_matrix`: Probability matrix (Items x K).
    - `r`: Number of powers.
    - `k`: Number of theta points.
    - `N`: Sample size.

    # Returns:
    - Tau matrix (R x K).
    """
    tau = zeros(Float64, r, k)
    total_samples = 500 * N
    sampled_data = p_matrix[rand(1:size(p_matrix, 1), total_samples), :]

    # Sum the sampled probabilities raised to the r-th power in batches of N
    for i in 1:500
        start_idx = (i - 1) * N + 1
        end_idx = i * N
        buffer = sampled_data[start_idx:end_idx, :]

        for r_index in 1:r
            tau[r_index, :] .+= vec(sum(buffer .^ r_index; dims=1))  # Use `vec` to flatten
        end
    end

    return tau / 500.0
end

function calc_info_tau(info_matrix::Matrix{Float64}, k::Int, N::Int)::Vector{Float64}
    """
    Calculate the tau vector for item information functions.

    # Arguments:
    - `info_matrix`: Information matrix (Items x K).
    - `k`: Number of theta points.
    - `N`: Sample size.

    # Returns:
    - Tau information vector (length k).
    """
    tau = zeros(Float64, k)
    total_samples = 500 * N
    sampled_data = info_matrix[rand(1:size(info_matrix, 1), total_samples), :]

    # Sum the sampled information across items in batches of N
    for i in 1:500
        start_idx = (i - 1) * N + 1
        end_idx = i * N
        buffer = sampled_data[start_idx:end_idx, :]

        tau .+= vec(sum(buffer; dims=1))  # Use `vec` to flatten
    end

    return tau / 500.0
end

"""
    observed_score_continuous(item_params::Matrix{Float64}, ability_dist::Normal; num_points::Int=100) -> Vector{Float64}

Calculates the observed score distribution using numerical integration for 3PL (dichotomous) items.

# Arguments

  - `item_params`: Matrix of item parameters (a, b, c), size (num_items, 3).
  - `ability_dist`: Ability distribution (e.g., Normal).
  - `num_points`: Number of integration points (default 100).

# Returns

  - Observed score distribution as a vector.
"""
function observed_score_continuous(item_params::Matrix{Float64}, ability_dist::Normal;
                                   num_points::Int=120)::Vector{Float64}
    num_items = size(item_params, 1)

    # Define the integrand function for numerical integration
    function integrand(θ, x)
        score_dist = lw_dist(item_params, θ)  # Score distribution for dichotomous items
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
        observed_dist[x + 1] = quadgk(θ -> integrand(θ, x), -Inf, Inf; order=num_points)[1]
    end

    return observed_dist
end

"""
    lw_dist(item_params::Matrix{Float64}, θ::Float64) -> Vector{Float64}

Implementation of Lord and Wingersky Recursion Formula to calculate score distribution for dichotomous items.

# Arguments

  - `item_params`: Matrix of item parameters (a, b, c).
  - `θ`: Ability parameter.

# Returns

  - Score distribution as a vector.
"""
function lw_dist(item_params::Matrix{Float64}, θ::Float64)::Vector{Float64}
    num_items = size(item_params, 1)

    # Pre-allocate the result array for probabilities
    res = ones(Float64, num_items + 1)  # Initialize with 1 (for zero correct answers)

    # Loop over items and apply the recursion formula
    for i in 1:num_items
        a, b, c = item_params[i, :]  # Corrected: extract parameters from row `i`

        prob_correct = Probability(θ, b, a, c)
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
