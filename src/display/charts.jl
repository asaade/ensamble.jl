module Charts

export plot_results, save_to_csv, simulate_scores

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
    simulate_scores(parms, results, dist=Normal(0, 1)) -> DataFrame

Simulates test scores based on ability distribution for 3PL items.
"""
function simulate_scores(
        parms::Parameters, results::DataFrame, dist::Distribution = Normal(0, 1)
)
    bank = parms.bank
    n_items, n_forms = size(results)
    n_items += 1  # Add one extra row to account for the padded zero score

    # Preallocate a matrix to store the results for all forms
    sim_matrix = Matrix{Union{Missing, Float64}}(missing, n_items, n_forms)

    @threads for form in 1:n_forms
        selected_items = results[:, form]

        # Extract IRT parameters (a, b, c) for the selected items (dichotomous 3PL model)
        a, b, c = irt_params(bank, selected_items)

        # Prepare the item_params matrix for lw_dist (parameters: a, b, c for each item)
        params = Matrix(hcat(a, b, c))  # Corrected: no transpose needed

        # Apply Lord-Wingersky recursion to calculate the score distribution
        total_scores = observed_score_continuous(params, dist; parms.D)

        # Pad the results if needed and store in the simulation matrix
        sim_matrix[:, form] = vcat(
            total_scores, fill(missing, n_items - length(total_scores))
        )
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
    sim_data = DataFrame(simulate_scores(parms, results), :auto)

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
    p3 = @df sim_data plot(
        1:size(sim_data, 1),
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
