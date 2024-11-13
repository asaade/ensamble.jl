# -----------------------------------------------------------------------------
# Module for IRT Utility Functions (Improved)
# -----------------------------------------------------------------------------

module StatsFunctions

export calc_tau,
       calc_info_tau,
       prob_3pl, prob_item,
       info_3pl,
       observed_score_continuous,
       lw_dist,
       observed_score_continuous_log,
       lw_dist_log,
       expected_score_item,
       expected_score_matrix,
       expected_info_matrix,
       calc_scores_reference
export SUPPORTED_DICHOTOMOUS_MODELS
export SUPPORTED_POLYTOMOUS_MODELS
export SUPPORTED_MODELS

using Random, StatsBase, DataFrames, Distributions, QuadGK, Base.Threads, ThreadsX,
      FastGaussQuadrature

# Constants
const SUPPORTED_DICHOTOMOUS_MODELS = ["1PL", "2PL", "3PL", "RASCH"]
const SUPPORTED_POLYTOMOUS_MODELS = ["PCM", "GPCM", "GRM"]
const SUPPORTED_MODELS = vcat(SUPPORTED_DICHOTOMOUS_MODELS, SUPPORTED_POLYTOMOUS_MODELS)

# -----------------------------------------------------------------------------
# Probability Functions for Item Response Models
# -----------------------------------------------------------------------------

function prob_3pl(a, b, c, θ; D = 1.0)
    @assert 0.0<=c<=1.0 "Parameter `c` must be between 0 and 1."
    exponent = D * a * (θ - b)
    p = c + (1 - c) / (1 + exp(-exponent))
    return p
end

function prob_pcm(a, b, θ; D = 1.0)
    delta = D * a .* (θ .- b)
    sum_terms = cumsum([0.0; delta])
    numerators = exp.(sum_terms)
    prob_category = numerators / sum(numerators)
    return prob_category
end

function prob_grm(a, bs, θ; D = 1.0)
    z = D * a .* (θ .- bs)
    P_star = 1.0 ./ (1.0 .+ exp.(-z))
    P_star = vcat(1.0, P_star, 0.0)
    prob_category = P_star[1:(end - 1)] .- P_star[2:end]
    return prob_category
end

function prob_gpcm(a, b, θ; D = 1.0)
    # `a` is a scalar discrimination parameter
    delta_theta = θ .- b
    sum_terms = cumsum(D * a * delta_theta)
    sum_terms = vcat(0.0, sum_terms)
    numerators = exp.(sum_terms)
    prob_category = numerators / sum(numerators)
    return prob_category
end

function prob_item(model::String, a::Float64, bs::Vector{Float64},
        c::Union{Nothing, Float64}, θ::Float64; D::Float64 = 1.0)
    if model in SUPPORTED_DICHOTOMOUS_MODELS
        # Use the first element of bs for dichotomous models
        @assert length(bs)==1 "Expected `bs` to be a single-element vector for dichotomous model."
        return [prob_3pl(a, bs[1], c, θ; D = D)]  # Return as a single-element vector
    elseif model == "PCM"
        return prob_pcm(a, bs, θ; D = D)  # Returns a vector of probabilities
    elseif model == "GPCM"
        return prob_gpcm(a, bs, θ; D = D)  # Returns a vector of probabilities
    elseif model == "GRM"
        return prob_grm(a, bs, θ; D = D)  # Returns a vector of probabilities
    else
        error("Unsupported model: $model")
    end
end

# -----------------------------------------------------------------------------
# Expected Score Functions
# -----------------------------------------------------------------------------

function expected_score_pcm(a, b, θ; D = 1.0)
    prob_category = prob_pcm(a, b, θ; D = D)
    scores = 0:length(b)
    expected_score = sum(prob_category .* scores)
    return expected_score
end

function expected_score_grm(a, b, θ; D = 1.0)
    prob_category = prob_grm(a, b, θ; D = D)
    scores = 0:length(b)
    expected_score = sum(prob_category .* scores)
    return expected_score
end

function expected_score_gpcm(a, b, θ; D = 1.0)
    prob_category = prob_gpcm(a, b, θ; D = D)
    scores = 0:length(b)
    expected_score = sum(prob_category .* scores)
    return expected_score
end

function expected_score_item(model, a, bs, c, θ; D = 1.0)
    if model in SUPPORTED_DICHOTOMOUS_MODELS
        expected_score = prob_3pl(a, bs[1], c, θ; D = D)
    elseif model == "PCM"
        expected_score = expected_score_pcm(a, bs, θ; D = D)
    elseif model == "GPCM"
        expected_score = expected_score_gpcm(a, bs, θ; D = D)
    elseif model == "GRM"
        expected_score = expected_score_grm(a, bs, θ; D = D)
    else
        error("Unsupported model: $model")
    end
    return expected_score
end

# -----------------------------------------------------------------------------
# Information Functions
# -----------------------------------------------------------------------------

function info_3pl(a, b, c, θ; D = 1.0)
    p = prob_3pl(a, b, c, θ; D = D)
    q = 1.0 - p
    numerator = (D * a * (p - c))^2 * q
    denominator = ((1.0 - c)^2) * p
    info = numerator / denominator
    return info
end

function info_grm(a, b, θ; D = 1.0)
    z = D * a .* (θ .- b)
    P_star = 1.0 ./ (1.0 .+ exp.(-z))
    P_star = vcat(0.0, P_star, 1.0)
    P_star_derivative = D * a .* P_star[2:(end - 1)] .* (1.0 .- P_star[2:(end - 1)])
    P_k = diff(P_star)
    numerator = P_star_derivative .^ 2
    denominator = P_k[1:(end - 1)] .* P_k[2:end]
    info_components = numerator ./ denominator
    info = sum(info_components)
    return info
end

function info_pcm(a, b, θ; D = 1.0)
    prob = prob_pcm(a, b, θ; D = D)
    categories = 0:length(b)
    expected_score = sum(prob .* categories)
    score_diff = categories .- expected_score
    info = (D * a)^2 * sum(prob .* (score_diff .^ 2))
    return info
end

function info_gpcm(a, b, θ; D = 1.0)
    prob = prob_gpcm(a, b, θ; D = D)
    eta_k = D * cumsum([0.0; a])
    eta_bar = sum(prob .* eta_k)
    eta_diff = eta_k .- eta_bar
    info = sum(prob .* (eta_diff .^ 2))
    return info
end

info_item(model, a, bs, c, θ; D = 1.0) =
    if model in SUPPORTED_DICHOTOMOUS_MODELS
        return info_3pl(a, bs[1], c, θ; D = D)
    elseif model == "PCM"
        return info_pcm(a, bs, θ; D = D)
    elseif model == "GPCM"
        return info_gpcm(a, bs, θ; D = D)
    elseif model == "GRM"
        return info_grm(a, bs, θ; D = D)
    else
        error("Unsupported model: $model")
    end

# -----------------------------------------------------------------------------
# Expected Score and Information Matrices
# -----------------------------------------------------------------------------

function expected_score_matrix(bank::DataFrame, θ_values::Vector{Float64}; D = 1.0)
    num_items = nrow(bank)
    num_thetas = length(θ_values)
    score_matrix = zeros(num_items, num_thetas)

    @threads for idx in 1:num_items
        model = bank[idx, :MODEL]
        a = bank[idx, :A]
        c = if model == "3PL"
            bank[idx, :C]
        else
            nothing
        end
        b_columns = filter(col -> occursin(r"^B\d*$|^B$", string(col)), names(bank))
        bs = [bank[idx, col] for col in b_columns if !ismissing(bank[idx, col])]

        expected_scores = map(θ -> expected_score_item(model, a, bs, c, θ; D = D), θ_values)
        score_matrix[idx, :] .= expected_scores
    end
    return score_matrix
end

function expected_info_matrix(bank::DataFrame, θ_values; D = 1.0)
    num_items = nrow(bank)
    num_thetas = length(θ_values)
    info_matrix = zeros(num_items, num_thetas)
    b_columns = filter(col -> occursin(r"^B\d*$", string(col)), names(bank))

    @threads for idx in 1:num_items
        model = bank.MODEL[idx]
        a = bank.A[idx]
        c = bank.C[idx]
        bs = [bank[idx, col] for col in b_columns if !ismissing(bank[idx, col])]

        for (j, θ) in enumerate(θ_values)
            info_matrix[idx, j] = info_item(model, a, bs, c, θ; D = D)
        end
    end
    return info_matrix
end

# -----------------------------------------------------------------------------
# Statistical Calculations for Test Assembly
# -----------------------------------------------------------------------------

function calc_tau(score_matrix, r, N; num_simulations = 1000, rng = Random.GLOBAL_RNG)
    num_items, num_theta_points = size(score_matrix)
    tau = zeros(r, num_theta_points)

    if N > num_items
        error("Sample size N cannot exceed the number of items in the score_matrix.")
    end

    seeds = rand(rng, UInt64, num_simulations)

    function process_simulation(seed)
        local_rng = MersenneTwister(seed)
        sampled_indices = randperm(local_rng, num_items)[1:N]
        buffer = score_matrix[sampled_indices, :]
        tau_local = zeros(r, num_theta_points)
        for r_index in 1:r
            tau_local[r_index, :] = sum(buffer .^ r_index; dims = 1)[:]
        end
        return tau_local
    end

    simulations = ThreadsX.map(process_simulation, seeds)

    for sim_tau in simulations
        tau .+= sim_tau
    end

    tau ./= num_simulations
    return tau
end

function calc_info_tau(info_matrix, N; num_batches = 1000,
        rng = Random.GLOBAL_RNG, sample_with_replacement = false)
    num_items, num_theta_points = size(info_matrix)
    tau = zeros(num_theta_points)

    if N > num_items && !sample_with_replacement
        error("Sample size N cannot exceed the number of items when sampling without replacement.")
    end

    seeds = rand(rng, UInt64, num_batches)

    function process_batch(seed)
        local_rng = MersenneTwister(seed)
        sampled_indices = if sample_with_replacement
            rand(local_rng, 1:num_items, N)
        else
            randperm(local_rng, num_items)[1:N]
        end
        buffer = info_matrix[sampled_indices, :]
        tau_local = sum(buffer; dims = 1)[:]
        return tau_local
    end

    batch_results = ThreadsX.map(process_batch, seeds)

    for batch_tau in batch_results
        tau .+= batch_tau
    end

    tau ./= num_batches
    return tau
end

function calc_scores_reference(expected_scores, N; num_forms = 1000)
    # Get the number of items and theta points
    num_items, _ = size(expected_scores)

    item_score_means = map(x -> mean(trim(x, prop = 0.1)), eachcol(expected_scores))

    # Initialize a random number generator and seeds for reproducibility
    rng = MersenneTwister()
    seeds = rand(rng, UInt64, num_forms)

    function process_form(seed)
        local_rng = MersenneTwister(seed)
        # Randomly select N items from the available items
        if N <= num_items
            sampled_indices = randperm(local_rng, num_items)[1:N]
        else
            error("Sample size N cannot exceed the number of items.")
        end
        # Compute the sum of the sampled scores for each theta point (aggregated scores)
        sampled_scores = expected_scores[sampled_indices, :]
        aggregated_scores = sum(sampled_scores; dims = 1)  # Sum across selected items
        # Compute the variance of the sampled scores for each theta point
        return aggregated_scores
    end

    # Perform simulations across the number of forms
    results = ThreadsX.map(process_form, seeds)
    aggregated_scores_per_form = vcat([res[1] for res in results]...)

    # Calculate tau_mean as the average aggregated scores across forms
    tau_mean = vec(mean(aggregated_scores_per_form; dims = 1))

    return tau_mean, item_score_means
end

num_categories(bs::Vector{Float64}) = length(bs) + 1

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
function lw_dist(
        item_params::Vector{Tuple{Float64, Vector{Float64}, Float64, String}},
        θ::Float64;
        D::Float64 = 1.0
)
    num_items = length(item_params)

    if num_items == 0
        return [1.0]
    end

    max_score = sum(
        num_categories(bs) - 1
    for (a, bs, c, model) in item_params
    )

    # Initialize result vector
    res = zeros(max_score + 1)
    res[1] = 1.0

    # Process each item
    for (a, bs, c, model) in item_params
        # Get response probabilities for current item
        response_probs = prob_item(model, a, bs, c, θ; D = D)

        # Handle dichotomous items
        if length(response_probs) == 1
            p_correct = response_probs[1]
            p_incorrect = 1.0 - p_correct
            response_probs = [p_incorrect, p_correct]
            possible_scores = [0, 1]
        else
            possible_scores = 0:(length(response_probs) - 1)
        end

        # Initialize temporary vector
        prov = zeros(length(res))

        # Update score probabilities
        for score in 0:max_score
            curr_prob = res[score + 1]
            if curr_prob < 1e-12
                continue
            end

            for (score_inc, prob_inc) in zip(possible_scores, response_probs)
                new_score = score + score_inc
                if new_score <= max_score
                    prov[new_score + 1] += curr_prob * prob_inc
                end
            end
        end

        # Update res
        res .= prov
    end

    return res
end

"""
    observed_score_continuous(item_params, ability_dist; D = 1.0, num_points = 21)

Computes the observed score distribution for a test given a set of item parameters
and an ability distribution, using the Lord-Wingersky recursion and numerical integration.

# Arguments

  - `item_params::Vector{Tuple{Float64, Vector{Float64}, Float64, String}}`:
    A vector of tuples representing parameters for each item, including discrimination (`a`),
    difficulty parameters (`bs`), guessing parameter (`c`), and model type (`model`).
  - `ability_dist::ContinuousUnivariateDistribution`:
    The distribution of abilities (e.g., `Normal(0, 1)`).
  - `D::Float64=1.0`:
    Scaling constant for IRT models (typically set to 1 or 1.7).
  - `num_points::Int=21`:
    Number of quadrature points used for integration over the ability distribution.

# Returns

  - `Vector{Float64}`: The observed score distribution as a vector.
"""
function observed_score_continuous(
        item_params::Vector{Tuple{Float64, Vector{Float64}, Float64, String}},
        ability_dist::ContinuousUnivariateDistribution;
        D::Float64 = 1.0,
        num_points::Int = 101
)
    num_items = length(item_params)

    if num_items == 0
        return [1.0]
    end
    if !(ability_dist isa ContinuousUnivariateDistribution)
        error("ability_dist must be a continuous univariate distribution.")
    end

    # Calculate maximum possible score
    function num_categories(bs::Vector{Float64}, model)
        if model in SUPPORTED_DICHOTOMOUS_MODELS
            return 2
        elseif model in SUPPORTED_POLYTOMOUS_MODELS
            return length(bs) + 1
        else
            error("Unsupported model: $model")
        end
    end

    max_score = sum(
        num_categories(bs, model) - 1
    for (a, bs, c, model) in item_params
    )

    observed_dist = zeros(max_score + 1)

    # Use Gauss-Hermite quadrature for Normal distributions
    if ability_dist isa Normal
        μ = mean(ability_dist)
        σ = std(ability_dist)
        # Obtain quadrature points and weights
        nodes, weights = gausshermite(num_points)
        θs = sqrt(2) * σ * nodes .+ μ
        ws = weights ./ sqrt(pi)

        for (θ, w) in zip(θs, ws)
            score_dist = lw_dist(item_params, θ; D = D)
            observed_dist .+= w * score_dist
        end
    else
        # For non-Normal distributions, use adaptive quadrature
        function integrand(θ)
            score_dist = lw_dist(item_params, θ; D = D)
            return score_dist * pdf(ability_dist, θ)
        end
        observed_dist, _ = quadgk(θ -> integrand(θ), -Inf, Inf; atol = 1e-8, rtol = 1e-6)
    end

    # Normalize the observed distribution to sum to 1
    total_prob = sum(observed_dist)
    if total_prob > 0.0
        observed_dist ./= total_prob
    end

    return observed_dist
end

function lw_dist_log(
        item_params::Vector{Tuple{Float64, Vector{Float64}, Float64, String}},
        θ::Float64;
        D::Float64 = 1.0
)
    num_items = length(item_params)

    if num_items == 0
        return [0.0]  # log(1.0) = 0.0
    end

    # Initialize result vector in log domain
    max_score = sum(
        num_categories(bs) - 1
    for (a, bs, c, model) in item_params
    )
    res = fill(-Inf, max_score + 1)
    res[1] = 0.0  # log(1.0)

    # Process each item
    for (a, bs, c, model) in item_params
        # Get response probabilities for current item
        response_probs = prob_item(model, a, bs, c, θ; D = D)

        # Handle dichotomous items
        if length(response_probs) == 1
            p_correct = response_probs[1]
            p_incorrect = 1.0 - p_correct
            response_probs = [p_incorrect, p_correct]
            possible_scores = [0, 1]
        else
            possible_scores = 0:(length(response_probs) - 1)
        end

        # Convert probabilities to log domain
        log_response_probs = log.(response_probs)

        # Initialize temporary vector in log domain
        prov = fill(-Inf, length(res))

        # Update score probabilities
        for score in 0:max_score
            curr_log_prob = res[score + 1]
            # Removed the condition that skips when curr_log_prob == -Inf

            for (score_inc, log_prob_inc) in zip(possible_scores, log_response_probs)
                new_score = score + score_inc
                if new_score <= max_score
                    prov[new_score + 1] = logsumexp((
                        prov[new_score + 1], curr_log_prob + log_prob_inc))
                end
            end
        end

        # Update res
        res .= prov
    end

    # Convert back to probability domain
    res_probs = exp.(res)
    return res_probs
end

# Helper function for log-sum-exp
function logsumexp(log_probs::Tuple{Float64, Float64})
    a, b = log_probs
    if a == -Inf
        return b
    elseif b == -Inf
        return a
    else
        max_log_prob = max(a, b)
        return max_log_prob + log(exp(a - max_log_prob) + exp(b - max_log_prob))
    end
end

function observed_score_continuous_log(
        item_params::Vector{Tuple{Float64, Vector{Float64}, Float64, String}},
        ability_dist::ContinuousUnivariateDistribution;
        D::Float64 = 1.0,
        num_points::Int = 41
)
    num_items = length(item_params)

    if num_items == 0
        return [1.0]
    end
    if !(ability_dist isa ContinuousUnivariateDistribution)
        error("ability_dist must be a continuous univariate distribution.")
    end

    # Calculate maximum possible score
    max_score = sum(
        num_categories(bs) - 1
    for (a, bs, c, model) in item_params
    )

    observed_dist = zeros(max_score + 1)

    # Use Gauss-Hermite quadrature for Normal distributions
    if ability_dist isa Normal
        μ = mean(ability_dist)
        σ = std(ability_dist)
        # Obtain quadrature points and weights
        nodes, weights = gausshermite(num_points)
        θs = sqrt(2) * σ * nodes .+ μ
        ws = weights ./ sqrt(pi)

        for (θ, w) in zip(θs, ws)
            score_dist = lw_dist_log(item_params, θ; D = D)
            observed_dist .+= w * score_dist
        end
    else
        # For non-Normal distributions, use adaptive quadrature
        function integrand(θ)
            score_dist = lw_dist_log(item_params, θ; D = D)
            return score_dist * pdf(ability_dist, θ)
        end
        observed_dist, _ = quadgk(θ -> integrand(θ), -Inf, Inf; atol = 1e-8, rtol = 1e-6)
    end

    # Normalize the observed distribution to sum to 1
    total_prob = sum(observed_dist)
    if total_prob > 0.0
        observed_dist ./= total_prob
    end

    return observed_dist
end

end # module IRTUtils
