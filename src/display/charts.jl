module Charts

export plot_results, save_to_csv, simulate_scores, simulate_scores_log

using CSV, DataFrames, Plots, StatsPlots, Distributions, Measures
using Base.Threads
using StatsBase
using Base.Threads

using ..Configuration, ..Utils

"""
    irt_params(bank, items) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}

Extracts IRT parameters (a, b, c) for selected items.
"""
function irt_params(bank::DataFrame, items::Vector)
    idx = bank.ID .∈ Ref(skipmissing(items))
    a = bank[idx, :A]
    b = bank[idx, :B]
    c = bank[idx, :C]

    return a, b, c
end

"""
    irt_params(bank, idx) -> Tuple{Float64, Union{Float64, Vector{Float64}}, Union{Float64, Nothing}, String}

Extracts IRT parameters (discrimination `a`, difficulty `bs`, guessing `c`, and model type)
for a specified item index from an item bank DataFrame.

# Arguments
- `bank::DataFrame`: A DataFrame containing IRT item parameters.
- `idx::Int`: Index of the item to extract parameters for.

# Returns
- `Tuple{Float64, Union{Float64, Vector{Float64}}, Union{Float64, Nothing}, String}`: A tuple containing
  the discrimination (`a`), difficulty parameters (`bs` as `Float64` or `Vector{Float64}`), guessing parameter (`c`),
  and the model type as a string.
"""
function irt_params2(bank::DataFrame, idx::Int)
    model = bank[idx, :MODEL]
    a = if model == "RASCH"
        1.0
    else
        bank[idx, :A]
    end

    # Set c to 0.0 for non-3PL models to ensure consistent type handling as Float64
    c = if model == "3PL"
        bank[idx, :C]
    else
        0.0  # Set to 0.0 for all other models
    end

    # Extracting difficulty parameters (bs)
    b_columns = filter(col -> occursin(r"^B\d*$|^B$", string(col)), names(bank))
    bs_values = [bank[idx, col] for col in b_columns if !ismissing(bank[idx, col])]

    # Ensure bs is always a vector, even for dichotomous models
    bs = if model in SUPPORTED_DICHOTOMOUS_MODELS && length(bs_values) == 1
        [bs_values[1]]  # Wrap single value in a vector
    else
        bs_values  # Already a vector for polytomous models
    end

    return a, bs, c, model
end



"""
    simulate_scores(bank, results, dist; D = 1.0) -> Matrix{Union{Missing, Float64}}

Simulates test scores based on a given ability distribution and an item bank, applying
the Lord-Wingersky recursion formula to compute observed score distributions.

# Arguments
- `bank::DataFrame`: A DataFrame containing item parameters (including model type, discrimination, etc.).
- `results::DataFrame`: A DataFrame where each column corresponds to a form (test version)
  and rows represent item IDs.
- `dist::Distribution=Normal(0, 1)`: The ability distribution for simulated examinees (default is standard normal).
- `D::Float64=1.0`: Scaling constant for IRT models (typically 1 or 1.7).

# Returns
- `Matrix{Union{Missing, Float64}}`: A matrix where each column represents simulated scores
  for a corresponding test form, with rows representing different possible total scores.
"""
function simulate_scores(
    bank::DataFrame,
    results::DataFrame,
    dist::Distribution = Normal(0, 1);
    D = 1.0
)
    n_forms = size(results, 2)
    total_scores_list = Vector{Vector{Float64}}()
    max_total_score = 0

    # Create a mapping from item IDs to indices in bank
    id_to_index = Dict(id => idx for (idx, id) in enumerate(bank.ID))

    for form in 1:n_forms
        # Extract item IDs for the current form
        items = skipmissing(results[:, form])

        # Get indices of items in bank.ID
        items_idx = [id_to_index[item_id] for item_id in items if haskey(id_to_index, item_id)]

        # Extract parameters for selected items
        params = [irt_params2(bank, idx) for idx in items_idx]

        # Prepare the item_params list for lw_dist
        item_params = [Tuple(param) for param in params]

        # Calculate observed score distribution
        total_scores = observed_score_continuous(item_params, dist; D = D)

        # Store total_scores in list
        push!(total_scores_list, total_scores)

        # Update max_total_score
        max_total_score = max(max_total_score, length(total_scores) - 1)
    end

    # Now, create sim_matrix with size (max_total_score + 1, n_forms)
    sim_matrix = zeros(max_total_score + 1, n_forms)

    # For each form, pad total_scores to length max_total_score + 1, and store in sim_matrix
    for (form_idx, total_scores) in enumerate(total_scores_list)
        padded_scores = vcat(total_scores, zeros(max_total_score + 1 - length(total_scores)))
        sim_matrix[:, form_idx] = padded_scores
    end

    return sim_matrix
end


function simulate_scores_log(
    bank::DataFrame,
    results::DataFrame,
    dist::Distribution = Normal(0, 1);
    D = 1.0
)
    n_forms = size(results, 2)
    total_scores_list = Vector{Vector{Float64}}()
    max_total_score = 0

    # Create a mapping from item IDs to indices in bank
    id_to_index = Dict(id => idx for (idx, id) in enumerate(bank.ID))

    for form in 1:n_forms
        # Extract item IDs for the current form
        items = skipmissing(results[:, form])

        # Get indices of items in bank.ID
        items_idx = [id_to_index[item_id] for item_id in items if haskey(id_to_index, item_id)]

        # Extract parameters for selected items
        params = [irt_params2(bank, idx) for idx in items_idx]

        # Prepare the item_params list for lw_dist_log
        item_params = [Tuple(param) for param in params]

        # Calculate observed score distribution
        total_scores = observed_score_continuous_log(item_params, dist; D = D)

        # Store total_scores in list
        push!(total_scores_list, total_scores)

        # Update max_total_score
        max_total_score = max(max_total_score, length(total_scores) - 1)
    end

    # Now, create sim_matrix with size (max_total_score + 1, n_forms)
    sim_matrix = zeros(max_total_score + 1, n_forms)

    # For each form, pad total_scores to length max_total_score + 1, and store in sim_matrix
    for (form_idx, total_scores) in enumerate(total_scores_list)
        padded_scores = vcat(total_scores, zeros(max_total_score + 1 - length(total_scores)))
        sim_matrix[:, form_idx] = padded_scores
    end

    return sim_matrix
end



"""
    char_curves(parms, results, theta_range, r=1) -> DataFrame

Generates characteristic curve data for all forms using dichotomous items.
"""
function char_curves(
        parms::Parameters,
        results::DataFrame,
        theta_range::Union{AbstractVector, AbstractRange},
        r::Int = 1
)
    bank = parms.bank
    n_forms = size(results, 2)
    n_thetas = length(theta_range)

    curves_matrix = Matrix{Float64}(undef, n_thetas, n_forms)

    # Loop over each form
    for i in 1:n_forms
        selected = results[:, i]
        a, b, c = irt_params(bank, selected)
        scores = zeros(Float64, n_thetas)

        # Recalculate probabilities dynamically for each theta value
        for (j, theta) in enumerate(theta_range)
            for k in eachindex(selected)
                prob = prob_3pl(a[k], b[k], c[k], theta; D = parms.D)
                scores[j] += prob^r  # Summing the probability for correct response
            end
        end

        curves_matrix[:, i] = scores
    end

    curves = DataFrame(curves_matrix, Symbol.(names(results)))
    return round.(curves, digits = 2)
end

"""
    score_curves(parms::Parameters, results::DataFrame,
                     theta_range::Union{AbstractVector, AbstractRange})::DataFrame

Generates expected scrore curve data for all forms using dichotomous items.
"""
function expected_score_curves(
        parms::Parameters, results::DataFrame, theta_range::Union{
            AbstractVector, AbstractRange}
)::DataFrame
    bank = parms.bank
    theta::Vector{Float64} = collect(theta_range)
    n_forms = size(results, 2)  # Number of forms (columns in results)
    n_thetas = length(theta)  # Number of theta points

    # Pre-allocate the curves matrix (theta points x forms)
    curves_matrix = Matrix{Float64}(undef, n_thetas, n_forms)

    # Loop over each form in parallel
    @threads for i in 1:n_forms
        selected = results[:, i]  # Select the i-th column
        idx = bank.ID .∈ Ref(skipmissing(selected))  # Filter selected items from bank

        # Compute the expected score matrix for the selected items
        scores = expected_score_matrix(bank[idx, :], theta; parms.D)

        # Store the sum of the scores for this form in the curves matrix
        curves_matrix[:, i] = sum(scores; dims = 1)  # Sum along items (rows)
    end

    # Convert the curves matrix into a DataFrame and round the results
    curves = DataFrame(curves_matrix, Symbol.(names(results)))
    return round.(curves, digits = 2)
end

"""
    info_curves(parms, results, theta_range) -> DataFrame

Generates information curve data for all forms using dichotomous items.
"""
function info_curves(
        parms::Parameters, results::DataFrame, theta_range::Union{
            AbstractVector, AbstractRange}
)
    bank = parms.bank
    n_forms = size(results, 2)
    n_thetas = length(theta_range)

    info_matrix = Matrix{Float64}(undef, n_thetas, n_forms)

    # Loop over each form
    for i in 1:n_forms
        selected = results[:, i]
        a, b, c = irt_params(bank, selected)
        info = zeros(Float64, n_thetas)

        # Recalculate information dynamically for each theta value
        for (j, theta) in enumerate(theta_range)
            for k in eachindex(selected)
                info[j] += info_3pl(a[k], b[k], c[k], theta; D = parms.D)
            end
        end

        info_matrix[:, i] = info
    end

    info_data = DataFrame(info_matrix, Symbol.(names(results)))
    return round.(info_data, digits = 2)
end

function expected_info_curves(
        parms::Parameters, results::DataFrame, theta_range::Union{
            AbstractVector, AbstractRange}
)
    bank = parms.bank
    theta::Vector{Float64} = collect(theta_range)
    n_forms = size(results, 2)  # Number of forms (columns in results)
    n_thetas = length(theta)  # Number of theta points

    # Pre-allocate the curves matrix (theta points x forms)
    curves_matrix = Matrix{Float64}(undef, n_thetas, n_forms)

    # Loop over each form in parallel
    @threads for i in 1:n_forms
        selected = results[:, i]  # Select the i-th column
        idx = bank.ID .∈ Ref(skipmissing(selected))  # Filter selected items from bank

        # Compute the expected information matrix for the selected items
        info = expected_info_matrix(bank[idx, :], theta; parms.D)

        # Store the sum of the information for this form in the curves matrix
        curves_matrix[:, i] = sum(info; dims = 1)  # Sum along items (rows)
    end

    # Convert the curves matrix into a DataFrame and round the results
    curves = DataFrame(curves_matrix, Symbol.(names(results)))
    return round.(curves, digits = 2)
end

"""
    save_to_csv(data, file)

Saves a DataFrame to a CSV file.
"""
save_to_csv(data::DataFrame, file::String) = CSV.write(file, data)

"""
    plot_results(parms, conf, results, theta_range=-3.0:0.1:3.0, plot_file="results/combined_plot.pdf") -> DataFrame

Generates and plots characteristic curves, information curves, and simulated scores.
"""
function plot_results(
        parms::Parameters,
        conf::Config,
        results::DataFrame,
        theta_range::Union{AbstractVector, AbstractRange} = -4.0:0.1:4.0,
        plot_file::String = "results/combined_plot.pdf"
)::DataFrame
    # Generate characteristic and information curves
    char_data, info_data = make_curves(parms, results, theta_range)

    # Generate simulation data
    sim_data = DataFrame(simulate_scores(parms.bank, results), :auto)

    # Combine all plots
    combined = combine_plots(parms, theta_range, char_data, info_data, sim_data)

    # Save results and plots
    save_all(char_data, theta_range, conf, combined, plot_file)

    return char_data
end

"""
    make_curves(parms, results, theta_range) -> Tuple{DataFrame, DataFrame}

Generates both characteristic and information curves.
"""
function make_curves(
        parms::Parameters, results::DataFrame, theta_range::Union{
            AbstractVector, AbstractRange}
)
    char_data = expected_score_curves(parms, results, theta_range)
    info_data = expected_info_curves(parms, results, theta_range)
    return char_data, info_data
end

"""
    combine_plots(parms, theta_range, char_data, info_data, sim_data) -> Plot

Combines characteristic, information, and simulation plots.
"""
function combine_plots(
        parms::Parameters,
        theta_range::Union{AbstractVector, AbstractRange},
        char_data::DataFrame,
        info_data::DataFrame,
        sim_data::DataFrame
)
    theme(:dao)
    gr(; size = (950, 850), legend = :topright)

    # Plot characteristic curves with improved labels and styles
    p1 = @df char_data plot(
        theta_range,
        cols(),
        title = "Test Characteristic Curves",
        xlabel = "Ability (θ)",
        ylabel = "Expected Score",
        linewidth = 2,
        label = "",
        grid = :both,
        legend = :topright,
        xticks = :auto,
        yticks = :auto,
        color = :viridis,
        ylims = (0, parms.max_items)
    )
    (parms.method in ["TCC", "TCC2", "MIXED"]) &&
        scatter!(parms.theta, parms.tau[1, :]; label = "", markersize = 4)

    # Add information curves on the same graph using dual axes (right axis for info curves)
    p2 = @df info_data plot(
        theta_range,
        cols(),
        title = "Test Information Curves",
        xlabel = "Ability (θ)",
        ylabel = "Information",
        linewidth = 2,
        label = "",
        grid = :both,
        color = :plasma
    )

    # Overlay a reference line at θ = 0
    vline!([0]; color = :gray, linestyle = :dash, label = "")

    # # Plot simulated observed scores with score distribution (optional)
    max_total_score = size(sim_data, 1) - 1
    scores = 0:max_total_score
    p3 = @df sim_data plot(
        scores,
        cols(),
        title = "Simulated Observed Scores",
        xlabel = "Score",
        ylabel = "Frequency",
        linewidth = 2,
        label = "",
        grid = :both,
        color = :inferno
    )

    # # Combine plots into a single layout with consistent sizes
    # plot(p1, p2, p3; layout=(2, 2), size=(950, 850), margin=8mm)
    return plot(p1, p2, p3; layout = (2, 2), size = (950, 850), margin = 8mm)
end

"""
    save_all(curves, theta_range, conf, plot, plot_file)

Saves the curves to a CSV and the plot to a file.
"""
function save_all(
        curves::DataFrame,
        theta_range::Union{AbstractVector, AbstractRange},
        conf::Config,
        plot::AbstractPlot,
        plot_file::String
)
    insertcols!(curves, 1, :THETA => collect(theta_range))
    save_to_csv(curves, conf.tcc_file)

    savefig(plot, plot_file)

    println("TCC data saved to: ", conf.tcc_file)
    return println("Charts saved to: ", plot_file)
end

end
