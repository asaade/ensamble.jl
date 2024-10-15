module Charts

export plot_results, save_to_csv

using CSV, DataFrames, Plots, StatsPlots, Distributions, Measures
using Base.Threads
using StatsBase

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
    b_thresholds = bank[idx, :B_THRESHOLDS]
    model_type = bank[idx, :MODEL_TYPE]
    return a, b, c, b_thresholds, model_type
end


"""
    simulate_scores(parms, results, dist=Normal(0, 1)) -> DataFrame

Simulates test scores based on ability distribution for both dichotomous and polytomous items.
"""
function simulate_scores(parms::Parameters, results::DataFrame, dist::Distribution=Normal(0, 1))
    bank = parms.bank
    n_items, n_forms = size(results)
    n_items += 1  # Account for padded 0 score
    n_forms = size(results, 2)

    # Preallocate a matrix to store results (thread-safe)
    sim_matrix = Matrix{Union{Missing, Float64}}(missing, n_items, n_forms)

    @threads for i in 1:n_forms
        selected = results[:, i]
        a, b, c, b_thresh, model_type = irt_params(bank, selected)

        # Initialize observed score distribution
        total_items = 0
        for i in 1:length(model_type)
            if model_type[i] == "3PL"
                total_items += 1  # Dichotomous item contributes 1 score point
            else
                total_items += length(b_thresh[i])  # Polytomous item contributes length of b_thresh
            end
        end

        observed_dist = zeros(Float64, total_items + 1)  # Account for 0-based scores

        # Prepare the parameter matrices for dichotomous items
        dich_items_idx = findall(model_type .== "3PL")
        poly_items_idx = findall(model_type .!= "3PL")

        if !isempty(dich_items_idx)
            # Prepare dichotomous items and simulate using continuous Lord-Wingersky method
            dich_params = hcat(a[dich_items_idx], b[dich_items_idx], c[dich_items_idx])
            dich_params = Matrix(dich_params)  # Ensure regular matrix
            dich_scores = observed_score_continuous(dich_params, dist; model_type=fill("3PL", length(dich_items_idx)), b_thresh=nothing)
        else
            dich_scores = zeros(Float64, length(observed_dist))  # No dichotomous items
        end

        if !isempty(poly_items_idx)
            # Simulate scores for polytomous items using polytomous response model
            poly_scores = observed_score_continuous_polytomous(
                θ -> polytomous_probabilities(a[poly_items_idx], b_thresh[poly_items_idx], model_type[poly_items_idx], θ),
                dist
            )
        else
            poly_scores = zeros(Float64, length(observed_dist))  # No polytomous items
        end

        # Combine dichotomous and polytomous scores
        total_scores = dich_scores .+ poly_scores

        # Thread-local padded array for any missing values
        padded = vcat(total_scores, fill(missing, n_items - length(total_scores)))
        sim_matrix[:, i] = padded  # Thread-safe writing to matrix
    end

    # Convert the matrix to a DataFrame after the threaded loop
    sim_data = DataFrame(sim_matrix, Symbol.(names(results)))
    return sim_data
end



"""
    char_curves(parms, results, theta_range, r=1) -> DataFrame

Generates characteristic curve data for all forms.
"""
function char_curves(parms::Parameters, results::DataFrame, theta_range::Union{AbstractVector, AbstractRange}, r::Int=1)
    bank = parms.bank
    n_forms = size(results, 2)
    n_thetas = length(theta_range)

    curves_matrix = Matrix{Float64}(undef, n_thetas, n_forms)

    # @threads
    for i in 1:n_forms
        selected = results[:, i]
        a, b, c, b_thresh, model_type = irt_params(bank, selected)
        scores = zeros(Float64, n_thetas)

        for (j, theta) in enumerate(theta_range)
            for k in 1:length(selected)
                if model_type[k] == "3PL"
                    prob = Probability(theta, b[k], a[k], c[k])
                elseif model_type[k] == "GRM"
                    prob = prob_grm(a[k], b_thresh[k], theta)
                elseif model_type[k] == "PCM"
                    prob = prob_pcm(a[k], b_thresh[k], theta)
                elseif model_type[k] == "GPCM"
                    prob = prob_gpcm(a[k], b_thresh[k], theta)
                else
                    continue
                end
                scores[j] += sum(prob .^ r)  # Summing the probabilities for each category
            end
        end

        curves_matrix[:, i] = scores
    end

    curves = DataFrame(curves_matrix, Symbol.(names(results)))
    return round.(curves, digits=2)
end


"""
    info_curves(parms, results, theta_range) -> DataFrame

Generates information curve data for all forms.
"""
function info_curves(parms::Parameters, results::DataFrame, theta_range::Union{AbstractVector, AbstractRange})
    bank = parms.bank
    n_thetas = length(theta_range)
    n_forms = size(results, 2)

    info_matrix = Matrix{Float64}(undef, n_thetas, n_forms)

    #  @threads
    for i in 1:n_forms
        selected = results[:, i]
        a, b, c, b_thresh, model_type = irt_params(bank, selected)
        info = zeros(Float64, n_thetas)

        for (j, theta) in enumerate(theta_range)
            for k in 1:length(selected)
                if model_type[k] == "3PL"
                    info[j] += Information(theta, b[k], a[k], c[k])
                elseif model_type[k] == "GRM"
                    info[j] += info_grm(a[k], b_thresh[k], theta)
                elseif model_type[k] == "PCM"
                    info[j] += info_pcm(a[k], b_thresh[k], theta)
                elseif model_type[k] == "GPCM"
                    info[j] += info_gpcm(a[k], b_thresh[k], theta)
                else
                    continue
                end
            end
        end

        info_matrix[:, i] = info
    end

    info_data = DataFrame(info_matrix, Symbol.(names(results)))
    return round.(info_data, digits=2)
end


"""
    save_to_csv(data, file)

Saves a DataFrame to a CSV file.
"""
function save_to_csv(data::DataFrame, file::String)
    CSV.write(file, data)
end


"""
    plot_results(parms, conf, results, theta_range=-3.0:0.1:3.0, plot_file="results/combined_plot.png") -> DataFrame

Generates and plots characteristic curves, information curves, and simulated scores.
"""
function plot_results(parms::Parameters, conf::Config, results::DataFrame,
                      theta_range::Union{AbstractVector, AbstractRange} = -4.0:0.1:4.0,
                      plot_file::String = "results/combined_plot.png")::DataFrame

    # Generate characteristic and information curves
    char_data, info_data = make_curves(parms, results, theta_range)

    # Generate simulation data
    # sim_data = make_sims(parms, results)

    # Combine all plots
    combined = combine_plots(parms, theta_range, char_data, info_data) # , sim_data)

    # Save results and plots
    save_all(char_data, theta_range, conf, combined, plot_file)

    return char_data
end


"""
    make_curves(parms, results, theta_range) -> Tuple{DataFrame, DataFrame}

Generates both characteristic and information curves.
"""
function make_curves(parms::Parameters, results::DataFrame, theta_range::Union{AbstractVector, AbstractRange})
    char_data = char_curves(parms, results, theta_range)
    info_data = info_curves(parms, results, theta_range)
    return char_data, info_data
end

"""
    make_sims(parms, results) -> DataFrame

Generates a single simulation based on a Normal(0, 1) distribution.
"""
function make_sims(parms::Parameters, results::DataFrame; distr = Normal(0, 1))
    return simulate_scores(parms, results, distr)
end

"""
    combine_plots(parms, theta_range, char_data, info_data, sim_data) -> Plot

Combines characteristic, information, and simulation plots.
"""
function combine_plots(parms::Parameters, theta_range::Union{AbstractVector, AbstractRange},
                       char_data::DataFrame, info_data::DataFrame)
                       ## sim_data::DataFrame)

    theme(:default)
    gr(size=(950, 850), legend=:topright)

    # Plot characteristic curves with improved labels and styles
    p1 = @df char_data plot(theta_range, cols(),
                            title="Test Characteristic Curves",
                            xlabel="Ability (θ)", ylabel="Expected Score",
                            linewidth=2, label="", grid=:both, legend=:topright,
                            xticks=:auto, yticks=:auto, color=:viridis,
                            ylims = (0, parms.max_items))
    parms.method == "TCC" && scatter!(parms.theta, parms.tau[1, :]; label="", markersize=3)

    # Add information curves on the same graph using dual axes (right axis for info curves)
    p2 = @df info_data plot(theta_range, cols(),
                            title="Test Information Curves",
                            xlabel="Ability (θ)", ylabel="Information",
                            linewidth=2, label="", grid=:both, color=:plasma)

    # Overlay a reference line at θ = 0
    vline!([0], color=:gray, linestyle=:dash, label="")

    # # Plot simulated observed scores with score distribution (optional)
    # p3 = @df sim_data plot(1:size(sim_data, 1), cols(),
    #                        title="Simulated Observed Scores", xlabel="Items", ylabel="Percentage",
    #                        linewidth=2, label="", grid=:both, color=:inferno)

    # # Combine plots into a single layout with consistent sizes
    # plot(p1, p2, p3; layout=(2, 2), size=(950, 850), margin=8mm)
    plot(p1, p2; layout=(2, 1), size=(450, 850), margin=8mm)
end


"""
    save_all(curves, theta_range, conf, plot, plot_file)

Saves the curves to a CSV and the plot to a file.
"""
function save_all(curves::DataFrame, theta_range::Union{AbstractVector, AbstractRange}, conf::Config, plot::AbstractPlot, plot_file::String)
    insertcols!(curves, 1, :THETA => collect(theta_range))
    save_to_csv(curves, conf.tcc_file)

    savefig(plot, plot_file)

    println("TCC data saved to: ", conf.tcc_file)
    println("Charts saved to: ", plot_file)
end

end
