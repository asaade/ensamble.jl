module StatsFunctions

# Export the key functions to be used by other modules
export Probability,
    Information, calc_tau, calc_info_tau,
    observed_score_continuous, lw_dist,
    UnifiedProbability, UnifiedInformation,
    polytomous_probabilities

using Random, Distributions, QuadGK


"""
    UnifiedProbability(item::Dict, θ::Float64) -> Vector{Float64}

Calculates the probability of response for either 3PL or polytomous items,
depending on the item type.

# Arguments
  - `item`: A dictionary containing item parameters and model type.
  - `θ`: The ability parameter.

# Returns
  - A vector of probabilities for each response category (or for success/failure in 3PL).
"""
function UnifiedProbability(item::Dict, θ::Float64)::Vector{Float64}
    model_type = item["POLY_MODEL_TYPE"]

    if model_type == "3PL"
        # Use 3PL probability function
        return [Probability(θ, item["B"], item["A"], item["C"])]
    elseif model_type == "GRM"
        return prob_grm(item["A"], item["B"], θ)
    elseif model_type == "PCM"
        return prob_pcm(item["A"], item["B"], θ)
    elseif model_type == "GPCM"
        return prob_gpcm(item["A"], item["B"], θ)
    else
        error("Unsupported model type: $model_type")
    end
end



"""
    UnifiedInformation(item::Dict, θ::Float64) -> Float64

Calculates the information for either 3PL or polytomous items, depending on the item type.

# Arguments
  - `item`: A dictionary containing item parameters and model type.
  - `θ`: The ability parameter.

# Returns
  - The information value for the given item.
"""
function UnifiedInformation(item::Dict, θ::Float64)::Float64
    model_type = item["POLY_MODEL_TYPE"]

    if model_type == "3PL"
        # Use 3PL information function
        return Information(θ, item["B"], item["A"], item["C"])
    elseif model_type == "GRM"
        return info_grm(item["A"], item["B"], θ)
    elseif model_type == "PCM"
        return info_pcm(item["A"], item["B"], θ)
    elseif model_type == "GPCM"
        return info_gpcm(item["A"], item["B"], θ)
    else
        error("Unsupported model type: $model_type")
    end
end


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
    for k in 1:(K-1)
        prob[k] = 1.0 / (1.0 + exp(-a * (θ - b[k])))
    end

    # For the last category, probability is 1
    prob[K] = 1.0

    # Convert to probability of response in category
    prob_category = zeros(Float64, K)
    prob_category[1] = 1 - prob[1]
    for k in 2:(K-1)
        prob_category[k] = prob[k-1] - prob[k]
    end
    prob_category[K] = prob[K-1]

    return prob_category
end

function info_grm(a::Float64, b::Vector{Float64}, θ::Float64)
    K = length(b) + 1  # Number of categories
    prob = prob_grm(a, b, θ)
    info = 0.0

    # Calculate item information
    for k in 1:(K-1)
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
        numerators[k+1] = exp(sum_term)
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
    for k in 1:(K+1)
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
        numerators[k+1] = exp(sum_term)
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
    for k in 1:(K+1)
        p = prob[k]
        log_derivative = a[k] * (k - sum(prob .* (0:K)))  # Gradient of log-probability
        info += p * log_derivative^2
    end

    return info
end


"""
    calc_tau(P::Matrix{Float64}, R::Int, K::Int, N::Int) -> Matrix{Float64}

Calculates the tau matrix for given parameters and items based on probabilities.

# Arguments

  - `P`: Probability matrix.
  - `R`: Number of powers.
  - `K`: Number of points.
  - `N`: Sample size.

# Returns

  - The tau matrix.
"""
function calc_tau(P::Matrix{Float64}, R::Int, K::Int, N::Int)::Matrix{Float64}
    tau = zeros(Float64, R, K)
    rows = size(P, 1)
    for _ in 1:500
        sampled_data = P[rand(1:rows, N), :]
        for r in 1:R
            tau[r, :] .+= [sum(sampled_data[:, i] .^ r) for i in 1:K]
        end
    end
    return tau / 500.0
end

"""
    calc_info_tau(info::Matrix{Float64}, K::Int, N::Int) -> Vector{Float64}

Calculates the information tau vector for given parameters and items based on information functions.

# Arguments

  - `info`: Information matrix.
  - `K`: Number of items.
  - `N`: Sample size.

# Returns

  - Information tau vector.
"""
function calc_info_tau(info::Matrix{Float64}, K::Int, N::Int)::Vector{Float64}
    tau = zeros(Float64, K)
    rows = size(info, 1)
    for _ in 1:500
        sampled_data = info[rand(1:rows, N), :]
        tau .+= [sum(sampled_data[:, i]) for i in 1:K]
    end
    return tau ./ 500.0
end


function calc_tau_polytomous(P::Array{Float64, 3}, R::Int, K::Int, N::Int)::Matrix{Float64}
    """
    Calculates the tau matrix for polytomous models.

    # Arguments
    - `P`: 3D probability matrix of size (N_items, K_points, Categories). Contains probabilities for each category.
    - `R`: Number of powers.
    - `K`: Number of theta points.
    - `N`: Sample size (number of items to sample).

    # Returns
    - The tau matrix.
    """
    tau = zeros(Float64, R, K)
    N_items, _, categories = size(P)

    for _ in 1:500
        # Sample N items randomly from the probability matrix
        sampled_data = P[rand(1:N_items, N), :, :]

        for r in 1:R
            # Sum over categories for each theta point, then raise to power r
            for k in 1:K
                tau[r, k] += sum(sum(sampled_data[:, k, :] .^ r))
            end
        end
    end
    return tau / 500.0
end


function calc_info_tau_polytomous(info::Array{Float64, 3}, K::Int, N::Int)::Vector{Float64}
    """
    Calculates the information tau vector for polytomous models.

    # Arguments
    - `info`: 3D information matrix of size (N_items, K_points, Categories).
    - `K`: Number of theta points.
    - `N`: Sample size (number of items to sample).

    # Returns
    - The information tau vector.
    """
    tau = zeros(Float64, K)
    N_items, _, categories = size(info)

    for _ in 1:500
        # Sample N items randomly from the information matrix
        sampled_data = info[rand(1:N_items, N), :, :]

        # Sum over categories for each theta point
        for k in 1:K
            tau[k] += sum(sum(sampled_data[:, k, :]))
        end
    end

    return tau / 500.0
end


"""
    lw_dist(item_params::Matrix{Float64}, θ::Float64) -> Vector{Float64}

Implementation of Lord and Wingersky Recursion Formula to calculate score distribution.

# Arguments

  - `item_params`: Matrix of item parameters (a, b, c).
  - `θ`: Ability parameter.

# Returns

  - Score distribution as a vector.
"""
function lw_dist_dichotomous(item_params::Matrix{Float64}, θ::Float64)::Vector{Float64}
    num_items = size(item_params, 2)
    a, b, c = item_params[:, 1]
    prob_correct = Probability(θ, b, a, c)
    res = [1 - prob_correct, prob_correct]

    for i in 2:num_items
        a, b, c = item_params[:, i]
        prob_correct = Probability(θ, b, a, c)
        prob_incorrect = 1 - prob_correct

        prov = zeros(Float64, i + 1)
        prov[1] = prob_incorrect * res[1]

        for j in 2:i
            prov[j] = prob_incorrect * res[j] + prob_correct * res[j - 1]
        end
        prov[i + 1] = prob_correct * res[i]
        res = prov
    end

    return res
end


"""
    Computes the conditional distributions of number-correct (or observed) scores from the probabilities of category responses to items using Lord and Wingersky recursion formula.
"""
function lw_dist(item_params, θ::Float64, model_type, b_thresh)::Vector{Float64}
    num_items = length(item_params)

    # Initialize the first item's probability distribution
    if model_type[1] == "3PL"
        a, b, c = item_params[1]
        prob_correct = Probability(θ, b, a, c)
        res = [1 - prob_correct, prob_correct]  # Dichotomous case
    elseif b_thresh !== nothing && !isempty(b_thresh[1])
        res = polytomous_probabilities(θ, item_params[1], b_thresh[1], model_type[1])  # Polytomous case
    else
        error("Polytomous model with missing or invalid thresholds.")
    end

    # Iterate through remaining items to update the score distribution
    for i in 2:num_items
        if model_type[i] == "3PL"
            a, b, c = item_params[i]
            prob_correct = Probability(θ, b, a, c)
            prob_incorrect = 1 - prob_correct

            # Preallocate new result vector
            new_res = zeros(Float64, length(res) + 1)

            # Update probabilities for the dichotomous item
            new_res[1] = prob_incorrect * res[1]
            for j in 2:length(res)
                new_res[j] = prob_incorrect * res[j] + prob_correct * res[j - 1]
            end
            new_res[end] = prob_correct * res[end]

        elseif b_thresh !== nothing && !isempty(b_thresh[i]) && i <= length(b_thresh)
            prob_matrix = polytomous_probabilities(θ, item_params[i], b_thresh[i], model_type[i])
            num_cat = length(prob_matrix)
            new_res = zeros(Float64, length(res) + num_cat - 1)

            # Update probabilities for the polytomous item
            for j in 1:length(res)
                for k in 1:num_cat
                    new_res[j + k - 1] += res[j] * prob_matrix[k]
                end
            end
        else
            error("Polytomous model with missing thresholds or invalid index.")
        end

        # Update result for next iteration
        res = new_res
    end

    return res
end


"""
    observed_score_continuous(item_params, ability_dist::Normal; num_points::Int = 100, model_type, b_thresh) -> Vector{Float64}

Calculates the observed score distribution using numerical integration for both 3PL and polytomous items.
"""
function observed_score_continuous(item_params,
                                   ability_dist::Normal;
                                   num_points::Int=100,
                                   model_type,
                                   b_thresh)::Vector{Float64}
    # Convert model_type and b_thresh to Vector if they are PooledVector
    model_type = Vector(model_type)  # In case it's a PooledVector
    b_thresh = b_thresh === nothing ? nothing : Vector(b_thresh)

    num_items = length(item_params)

    function integrand(θ, x)
        score_dist = lw_dist(item_params, θ, model_type, b_thresh)
        if x + 1 > length(score_dist)
            return 0.0  # Avoid bounds error
        end
        return score_dist[x + 1] * pdf(ability_dist, θ)
    end

    # Calculate max possible score for mixed format (dichotomous and polytomous)
    max_score = 0
    for i in 1:num_items
        if model_type[i] == "3PL"
            max_score += 1  # Dichotomous items contribute 1 score point
        elseif b_thresh !== nothing && i <= length(b_thresh)
            max_score += length(b_thresh[i])  # Polytomous items contribute based on their thresholds
        elseif model_type[i] != "3PL" && b_thresh === nothing
            error("Missing thresholds for polytomous items.")
        end
    end

    # Initialize the observed distribution vector
    observed_dist = zeros(Float64, max_score + 1)

    # Iterate over possible observed scores (from 0 to max_score)
    for x in 0:max_score
        observed_dist[x + 1] = quadgk(θ -> integrand(θ, x), -Inf, Inf; order=num_points)[1]
    end

    return observed_dist
end


"""
    polytomous_probabilities(θ::Float64, params::Vector{Float64}, thresholds::Vector{Float64}, model_type::AbstractString) -> Vector{Float64}

Calculates the category probabilities for polytomous items (GRM, PCM, GPCM) given θ.

# Arguments
  - `θ`: Ability parameter.
  - `params`: Vector of item parameters (e.g., a, b for GRM).
  - `thresholds`: Vector of thresholds for polytomous models.
  - `model_type`: Polytomous model type (e.g., "GRM", "PCM", "GPCM").

# Returns
  - Vector of category probabilities.
"""
function polytomous_probabilities(θ::Float64, params::Vector{Float64}, thresholds::Vector{Float64}, model_type::AbstractString)::Vector{Float64}
    if model_type == "GRM"
        return prob_grm(params[1], thresholds, θ)
    elseif model_type == "PCM"
        return prob_pcm(params[1], thresholds, θ)
    elseif model_type == "GPCM"
        return prob_gpcm(params[1], thresholds, θ)
    else
        throw(ArgumentError("Unsupported polytomous model type: $model_type"))
    end
end


"""
    polytomous_information(θ::Float64, params::Vector{Float64}, thresholds::Vector{Float64}, model_type::AbstractString) -> Vector{Float64}

Calculates the category information for polytomous items (GRM, PCM, GPCM) given θ.

# Arguments
  - `θ`: Ability parameter.
  - `params`: Vector of item parameters (e.g., a, b for GRM).
  - `thresholds`: Vector of thresholds for polytomous models.
  - `model_type`: Polytomous model type (e.g., "GRM", "PCM", "GPCM").

# Returns
  - Vector of category information.
"""
function polytomous_information(θ::Float64, params::Vector{Float64}, thresholds::Vector{Float64}, model_type::AbstractString)::Vector{Float64}
    if model_type == "GRM"
        return info_grm(params[1], thresholds, θ)
    elseif model_type == "PCM"
        return info_pcm(params[1], thresholds, θ)
    elseif model_type == "GPCM"
        return info_gpcm(params[1], thresholds, θ)
    else
        throw(ArgumentError("Unsupported polytomous model type: $model_type"))
    end
end


end # module
