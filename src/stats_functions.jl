using Random, Distributions, QuadGK

"""
    Pr.(theta_k, B, A, C; d = 1.0)

Calculates the probability using the given parameters,
for one or many parameters or ability points (vectors)

# Arguments

  - `theta_k`: The ability parameter.
  - `A`: The discrimination parameter.
  - `B`: The difficulty parameter.
  - `C`: The guessing parameter.
  - `d`: The scaling constant (default is 1.0).

# Returns

  - The calculated probability.
"""
function Pr(θ::Float64, b::Float64, a::Float64, c::Float64; d::Float64 = 1.0)
    return c + (1 - c) / (1 + exp(-d * a * (θ - b)))
end

# Overload to handle scalar inputs for b, a, and c
function Pr(θ::Float64, b::AbstractVector, a::AbstractVector, c::AbstractVector;
            d::Float64 = 1.0)
    return c .+ (1 .- c) ./ (1 .+ exp.(-d .* a .* (θ .- b)))
end

"""
    item_information(theta_k, B, A, C; D = 1.0)

Calculates the item information using the given parameters.

# Arguments

  - `theta_k`: The ability parameter.
  - `A`: The discrimination parameter.
  - `B`: The difficulty parameter.
  - `C`: The guessing parameter.
  - `D`: The scaling constant (default is 1.0).

# Returns

  - The item information rounded to 4 decimal places.
"""
function item_information(θ::Float64, b::AbstractVector, a::AbstractVector,
                          c::AbstractVector; d::Float64 = 1.0)
    p = Pr.(θ, b, a, c; d = d)
    q = 1 .- p
    return (d .* a) .^ 2 .* (p .- c) .^ 2 .* q ./ ((1 .- c) .^ 2 .* p)
end

function item_information(θ::Float64, b::Float64, a::Float64, c::Float64; d::Float64 = 1.0)
    p = Pr.(θ, b, a, c; d = d)
    q = 1 - p
    return (d * a)^2 * (p - c)^2 * q / ((1 - c)^2 * p)
end

"""
    calc_tau(P, R::Int, K::Int, N::Int, items)

Calculates the tau matrix for given parameters and items.

# Arguments

  - `P`: The probability matrix.
  - `R`: The number of powers.
  - `K`: The number of points.
  - `N`: The sample size.
  - `items`: The items matrix.

# Returns

  - The tau matrix.
"""
function calc_tau(P, R::Int, K::Int, N::Int, items)
    tau = zeros((R, K))
    rows, _ = size(items)
    for _ in 1:500
        datos = P[rand(1:rows, N), :]
        for r in 1:R
            tau[r, :] .+= [sum(datos[:, i] .^ r) for i in 1:K]
        end
    end
    return tau / 500.0
end

"""
    calc_info_tau(info, K::Int, N::Int, items)

Calculates the information tau vector for given parameters and items.

# Arguments

  - `info`: The information matrix.
  - `K`: The number of items.
  - `N`: The sample size.
  - `items`: The items matrix.

# Returns

  - The information tau vector rounded to 4 decimal places.
"""
function calc_info_tau(info, K::Int, N::Int)
    tau = zeros(K)
    rows, _ = size(info)
    for _ in 1:500
        datos = info[rand(1:rows, N), :]
        tau .+= [sum(datos[:, i]) for i in 1:K]
    end
    return tau ./ 500.0
end

# Lord and Wingersky Recursion Formula implementation in Julia
function lw_dist(item_params::Matrix{Float64}, θ::Float64)
    num_items = size(item_params, 2)

    # Initialize the probability distributions for the first item
    a, b, c = item_params[:, 1]
    prob_correct = Pr.(θ, b, a, c)
    res = [1 - prob_correct, prob_correct]

    # Iterate through remaining items and apply recursion formula
    for i in 2:num_items
        a, b, c = item_params[:, i]
        prob_correct = Pr.(θ, b, a, c)
        prob_incorrect = 1 - prob_correct

        # Initialize provisional result for current item
        prov = zeros(Float64, i + 1)

        # Apply recursion formula
        prov[1] = prob_incorrect * res[1]
        for j in 2:i
            prov[j] = prob_incorrect * res[j] + prob_correct * res[j - 1]
        end
        prov[i + 1] = prob_correct * res[i]

        # Update result with provisional values
        res = prov
    end

    return res
end

# # Example usage
# item_params = [2.0 1.5 1.2; # Discrimination parameters (a)
#                0.0 0.5 1.0; # Difficulty parameters (b)
#                0.2 0.2 0.2] # Guessing parameters (c)

# θ = 1.0
# score_distribution = lw_dist(item_params, θ)
# println("Score distribution: ", score_distribution)

# Simulate a group of students with abilities drawn from a normal distribution N(0, 1)
function simulate_abilities(num_examinees::Int, mean::Float64 = 0.0, stddev::Float64 = 1.0)
    dist = Normal(mean, stddev)
    return rand(dist, num_examinees)
end

# Calculate the observed score distribution for a group of examinees
function observed_score_distribution(item_params::Matrix{Float64}, num_examinees::Int)
    # Simulate abilities for the group of students
    abilities = simulate_abilities(num_examinees)

    # Initialize the score distribution
    max_score = size(item_params, 2)
    cumulative_distribution = zeros(Float64, max_score + 1)

    # Aggregate results for all examinees
    for θ in abilities
        score_dist = lw_dist(item_params, θ)
        cumulative_distribution .= cumulative_distribution .+ score_dist
    end

    # Normalize to get the distribution of scores
    cumulative_distribution ./= num_examinees
    return cumulative_distribution
end

# # Example usage
# item_params = [2.0 1.5 1.2; # Discrimination parameters (a)
#                0.0 0.5 1.0; # Difficulty parameters (b)
#                0.2 0.2 0.2] # Guessing parameters (c)

# num_examinees = 10000
# observed_dist = observed_score_distribution(item_params, num_examinees)
# println("Observed score distribution: ", observed_dist)

# Function to calculate the observed score distribution using numerical integration
function observed_score_distribution_continuous(item_params::Matrix{Float64},
                                                ability_dist::Normal; num_points::Int = 100)
    num_items = size(item_params, 2)

    # Function to calculate f(x|θ) * ψ(θ)
    function integrand(θ, x)
        score_dist = lw_dist(item_params, θ)
        return score_dist[x + 1] * pdf(ability_dist, θ)
    end

    # Integrate for each score using Gaussian quadrature
    observed_dist = zeros(Float64, num_items + 1)
    for x in 0:num_items
        observed_dist[x + 1] = quadgk(θ -> integrand(θ, x), -Inf, Inf; order = num_points)[1]
    end

    return observed_dist
end

euclidian_distance(x::Vector{Float64}, y::Vector{Float64}) = (sqrt.(x .- y)) .^ 0.5

euclidian_distance(x::Number, y::Number) = sqrt.((x .- y) .^ 2)

function delta(q1::Vector{T}, q2::Vector{T}) where {T <: Number}
    items = size(q1, 1)
    v = sqrt.([(q1[i] - q1[j]) .^ 2 for i in 1:items for j in 1:items] .+
              [(q2[i] - q2[j]) .^ 2 for i in 1:items for j in 1:items])
    return reshape(v, items, items)
end

# # Example usage
# item_params = [2.0 1.5 1.2; # Discrimination parameters (a)
#                0.0 0.5 1.0; # Difficulty parameters (b)
#                0.2 0.2 0.2] # Guessing parameters (c)

# ability_dist = Normal(0.0, 1.0)
# observed_dist = observed_score_distribution_continuous(item_params, ability_dist)
# println("Observed score distribution: ", observed_dist)

# item_params = [0.60, 1.20, 1.00, 1.40, 1.00;
#                −1.70, −1.00, 0.80, 1.30, 1.40;
#                0.20, 0.20, 0.25, 0.25, 0.20]

# a = [0.60, 1.20, 1.00, 1.40, 1.00]
# b = [−1.70, −1.00, 0.80, 1.30, 1.40]
# c = [0.20, 0.20, 0.25, 0.25, 0.20]

# export Pr, item_information, calc_tau, calc_info_tau, lw_dist, observed_score_distribution_continuous

# end
