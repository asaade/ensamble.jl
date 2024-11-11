# -----------------------------------------------------------------------------
# Module for IRT Utility Functions (Improved)
# -----------------------------------------------------------------------------

module StatsFunctions

export calc_tau,
       calc_info_tau,
       prob_3pl,
       info_3pl,
       observed_score_continuous,
       lw_dist,
       expected_score_item,
       expected_score_matrix,
       expected_info_matrix,
       calc_scores_reference


using Random, StatsBase, DataFrames, Distributions, QuadGK, Base.Threads, ThreadsX

# -----------------------------------------------------------------------------
# Probability Functions for Item Response Models
# -----------------------------------------------------------------------------

function prob_3pl(a, b, c, θ; D = 1.0)
    @assert 0.0 <= c <= 1.0 "Parameter `c` must be between 0 and 1."
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

function prob_grm(a, b, θ; D = 1.0)
    z = D * a .* (θ .- b)
    P_star = 1.0 ./ (1.0 .+ exp.(-z))
    P_star = vcat(0.0, P_star, 1.0)
    prob_category = diff(P_star)
    return prob_category
end

function prob_gpcm(a, b, θ; D = 1.0)
    @assert length(a) == length(b) "Vectors `a` and `b` must have the same length."
    delta_theta = θ .- b
    sum_terms = cumsum(D .* a .* delta_theta)
    sum_terms = vcat(0.0, sum_terms)
    numerators = exp.(sum_terms)
    prob_category = numerators / sum(numerators)
    return prob_category
end

function prob_item(model, a, bs, c, θ; D = 1.0)
    if model == "3PL"
        return prob_3pl(a, bs[1], c, θ; D = D)
    elseif model == "PCM"
        return prob_pcm(a, bs, θ; D = D)
    elseif model == "GPCM"
        return prob_gpcm(a, bs, θ; D = D)
    elseif model == "2PL"
        return prob_3pl(a, bs[1], 0.0, θ; D = D)
    elseif model == "GRM"
        return prob_grm(a, bs, θ; D = D)
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
    if model == "3PL"
        p = prob_3pl(a, bs[1], c, θ; D = D)
        expected_score = p
    elseif model == "PCM"
        expected_score = expected_score_pcm(a, bs, θ; D = D)
    elseif model == "GPCM"
        expected_score = expected_score_gpcm(a, bs, θ; D = D)
    elseif model == "2PL"
        p = prob_3pl(a, bs[1], 0.0, θ; D = D)
        expected_score = p
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

function info_item(model, a, bs, c, θ; D = 1.0)
    if model == "3PL"
        return info_3pl(a, bs[1], c, θ; D = D)
    elseif model == "PCM"
        return info_pcm(a, bs, θ; D = D)
    elseif model == "GPCM"
        return info_gpcm(a, bs, θ; D = D)
    elseif model == "2PL"
        return info_3pl(a, bs[1], 0.0, θ; D = D)
    elseif model == "GRM"
        return info_grm(a, bs, θ; D = D)
    else
        error("Unsupported model: $model")
    end
end

# -----------------------------------------------------------------------------
# Expected Score and Information Matrices
# -----------------------------------------------------------------------------

function expected_score_matrix(bank::DataFrame, θ_values::Vector{Float64}; D = 1.0)
    num_items = nrow(bank)
    num_thetas = length(θ_values)
    score_matrix = zeros(num_items, num_thetas)

    @threads for idx in 1:num_items
        model = bank[idx, :MODEL_TYPE]
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
        model = bank.MODEL_TYPE[idx]
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

function calc_info_tau(info_matrix, N; num_batches = 1000, rng = Random.GLOBAL_RNG, sample_with_replacement = false)
    num_items, num_theta_points = size(info_matrix)
    tau = zeros(num_theta_points)

    if N > num_items && !sample_with_replacement
        error("Sample size N cannot exceed the number of items when sampling without replacement.")
    end

    seeds = rand(rng, UInt64, num_batches)

    function process_batch(seed)
        local_rng = MersenneTwister(seed)
        sampled_indices = sample_with_replacement ?
            rand(local_rng, 1:num_items, N) :
            randperm(local_rng, num_items)[1:N]
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
    num_items, num_theta_points = size(expected_scores)

    item_score_means = map(x -> mean(trim(x, prop=0.1)),  eachcol(expected_scores))

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







# function normalize_expected_scores_zscore(expected_scores)
#     # Calculate the global mean and standard deviation
#     mean_val = mean(expected_scores)
#     std_val = std(expected_scores)

#     # Avoid division by zero if std_val is 0
#     if std_val > 0
#         normalized_scores = (expected_scores .- mean_val) ./ std_val
#     else
#         normalized_scores = fill(0.0, size(expected_scores))  # Assign zero if all values are the same
#     end

#     return normalized_scores
# end




# function calc_normalized_scores_reference(expected_scores, N; num_forms = 250)
#     # Normalize and scale the expected scores
#     num_items, _ = size(expected_scores)
#     normalized_expected_scores = normalize_expected_scores_zscore(expected_scores)

#     if N == 0
#         # Compute tau_mean and tau_var directly from normalized scores
#         raw_tau_mean = vec(mean(normalized_expected_scores; dims = 1))
#         raw_tau_var = vec(var(normalized_expected_scores; dims = 1))
#         # Joint normalization of tau_mean and tau_var
#         min_val = min(minimum(raw_tau_mean), minimum(raw_tau_var))
#         max_val = max(maximum(raw_tau_mean), maximum(raw_tau_var))
#         range_val = max_val - min_val

#         # Normalize tau_mean and tau_var using the same scale
#         if range_val > 0
#             tau_mean = (raw_tau_mean .- min_val) ./ range_val * N
#             tau_var = (raw_tau_var .- min_val) ./ range_val * N
#         else
#             tau_mean = fill(0.5 * N, length(raw_tau_mean))
#             tau_var = fill(0.5 * N, length(raw_tau_var))
#         end
#         return tau_mean, tau_var
#     end

#     rng = MersenneTwister()
#     seeds = rand(rng, UInt64, num_forms)

#     function process_form(seed)
#         local_rng = MersenneTwister(seed)
#         sampled_indices = randperm(local_rng, num_items)[1:N]
#         sampled_scores = normalized_expected_scores[sampled_indices, :]
#         form_mean = mean(sampled_scores; dims = 1)
#         form_var = var(sampled_scores; dims = 1)
#         return form_mean, form_var
#     end

#     results = ThreadsX.map(process_form, seeds)
#     form_means = vcat([res[1] for res in results]...)
#     form_vars = vcat([res[2] for res in results]...)
#     raw_tau_mean = vec(mean(form_means; dims = 1))
#     raw_tau_var = vec(mean(form_vars; dims = 1))

#     # Joint normalization of tau_mean and tau_var
#     min_val = min(minimum(raw_tau_mean), minimum(raw_tau_var))
#     max_val = max(maximum(raw_tau_mean), maximum(raw_tau_var))
#     range_val = max_val - min_val

#     # Normalize tau_mean and tau_var using the same scale
#     if range_val > 0
#         tau_mean = (raw_tau_mean .- min_val) ./ range_val * N
#         tau_var = (raw_tau_var .- min_val) ./ range_val * N
#     else
#         tau_mean = fill(0.5 * N, length(raw_tau_mean))
#         tau_var = fill(0.5 * N, length(raw_tau_var))
#     end

#     return tau_mean, tau_var
# end





# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

function lw_dist(item_params, θ; D = 1.0)
    num_items = size(item_params, 1)
    res = ones(num_items + 1)  # Initialize result vector

    for i in 1:num_items
        a, b, c = item_params[i, :]
        prob_correct = prob_3pl(a, b, c, θ; D)
        prob_incorrect = 1.0 - prob_correct
        prov = zeros(i + 1)  # Temporary vector for storing calculations

        prov[1] = prob_incorrect * res[1]
        for j in 2:i
            prov[j] = prob_incorrect * res[j] + prob_correct * res[j - 1]
        end
        prov[i + 1] = prob_correct * res[i]
        res[1:(i + 1)] .= prov
    end
    return res
end

function observed_score_continuous(item_params, ability_dist::ContinuousUnivariateDistribution; D = 1.0, num_points = 120)
    num_items = size(item_params, 1)

    function integrand(θ, x)
        score_dist = lw_dist(item_params, θ; D)
        if x + 1 > length(score_dist)
            return 0.0
        end
        return score_dist[x + 1] * pdf(ability_dist, θ)
    end

    max_score = num_items
    observed_dist = zeros(max_score + 1)

    for x in 0:max_score
        observed_dist[x + 1] = quadgk(θ -> integrand(θ, x), -Inf, Inf; order = num_points)[1]
    end
    return observed_dist
end

end # module IRTUtils
