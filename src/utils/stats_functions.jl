module StatsFunctions

# Export the key functions to be used by other modules
export Probability,
       Information, calc_tau, calc_info_tau,
       observed_score_distribution_continuous, lw_dist

using Random, Distributions, QuadGK, Logging

"""
    Probability(θ::Float64, b::Float64, a::Float64, c::Float64; d::Float64 = 1.0) -> Float64

Calculates the probability of success given the IRT parameters and ability level θ.

# Arguments

  - `θ`: The ability parameter.
  - `a`: The discrimination parameter.
  - `b`: The difficulty parameter.
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
  - `a`: Vector of discrimination parameters.
  - `b`: Vector of difficulty parameters.
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

"""
    lw_dist(item_params::Matrix{Float64}, θ::Float64) -> Vector{Float64}

Implementation of Lord and Wingersky Recursion Formula to calculate score distribution.

# Arguments

  - `item_params`: Matrix of item parameters (a, b, c).
  - `θ`: Ability parameter.

# Returns

  - Score distribution as a vector.
"""
function lw_dist(item_params::Matrix{Float64}, θ::Float64)::Vector{Float64}
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
    observed_score_distribution(item_params::Matrix{Float64}, num_examinees::Int) -> Vector{Float64}

Calculates the observed score distribution for a group of examinees.

# Arguments

  - `item_params`: Matrix of item parameters.
  - `num_examinees`: Number of examinees.

# Returns

  - The observed score distribution as a vector.
"""
function observed_score_distribution(item_params::Matrix{Float64},
                                     num_examinees::Int)::Vector{Float64}
    abilities = simulate_abilities(num_examinees)
    max_score = size(item_params, 2)
    cumulative_distribution = zeros(Float64, max_score + 1)

    for θ in abilities
        score_dist = lw_dist(item_params, θ)
        cumulative_distribution .= cumulative_distribution .+ score_dist
    end

    return cumulative_distribution ./ num_examinees
end

"""
    observed_score_distribution_continuous(item_params::Matrix{Float64}, ability_dist::Normal; num_points::Int = 100) -> Vector{Float64}

Calculates the observed score distribution using numerical integration.

# Arguments

  - `item_params`: Matrix of item parameters.
  - `ability_dist`: Ability distribution (e.g., Normal(0, 1)).
  - `num_points`: Number of points for numerical integration.

# Returns

  - The observed score distribution as a vector.
"""
function observed_score_distribution_continuous(item_params::Matrix{Float64},
                                                ability_dist::Normal;
                                                num_points::Int=100)::Vector{Float64}
    num_items = size(item_params, 2)

    function integrand(θ, x)
        score_dist = lw_dist(item_params, θ)
        return score_dist[x + 1] * pdf(ability_dist, θ)
    end

    observed_dist = zeros(Float64, num_items + 1)
    for x in 0:num_items
        observed_dist[x + 1] = quadgk(θ -> integrand(θ, x), -Inf, Inf; order=num_points)[1]
    end

    return observed_dist
end

end # module
