module Charts

export plot_results, write_results_to_file

using CSV
using DataFrames
using Plots
using StatsPlots
using Distributions
using Measures  # For margin handling
using ..Configuration
using ..Utils

"""
    fetch_irt_parms(bank::DataFrame, selected_items::Vector{String}) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}

Extracts the IRT item parameters (a, b, c) from the `bank` DataFrame for the selected items.

# Arguments

  - `bank::DataFrame`: DataFrame containing the item bank with columns `A`, `B`, and `C`.
  - `selected_items::Vector{String}`: Vector of item IDs for which parameters are extracted.

# Returns

  - A tuple `(a, b, c)` where `a`, `b`, and `c` are vectors containing the respective IRT parameters.
"""
function fetch_irt_parms(bank::DataFrame, selected_items::Vector{String})
    selected_items_idx = bank.ID .∈ Ref(skipmissing(selected_items))
    a = bank[selected_items_idx, :A]
    b = bank[selected_items_idx, :B]
    c = bank[selected_items_idx, :C]
    return a, b, c
end

"""
    simulate_observed_scores(parms::Parameters, results::DataFrame, ability_dist::Distribution=Normal(0.0, 1.0)) -> DataFrame

Simulates observed test scores for test takers based on the proposed ability distribution using
the Lord and Wingersky Recursion Formula.

# Arguments

  - `parms::Parameters`: Struct containing the system's parameters in the item bank.
  - `results::DataFrame`: DataFrame containing the selected items for each form.
  - `ability_dist::Distribution`: Ability distribution of the test takers (default: Normal(0.0, 1.0)).

# Returns

  - A `DataFrame` containing the simulated observed scores for each form.
"""
function simulate_observed_scores(parms::Parameters, results::DataFrame,
                                  ability_dist::Distribution=Normal(0.0, 1.0))
    bank = parms.bank
    observed_dist = DataFrame()
    max_length, num_forms = size(results)
    max_length += 1
    column_names = names(results)

    for i in 1:num_forms
        selected_items = collect(skipmissing(results[:, i]))
        a, b, c = fetch_irt_parms(bank, selected_items)

        # Simulate observed scores based on item parameters
        item_params::Matrix{Float64} = hcat(a, b, c)'  # Transpose for matrix structure
        dist = observed_score_distribution_continuous(item_params, ability_dist)
        padded_dist = vcat(dist, fill(missing, max_length - length(dist)))
        observed_dist[!, column_names[i]] = padded_dist
    end

    return observed_dist
end

"""
    generate_characteristic_curves(parms::Parameters, results::DataFrame, theta_range::AbstractVector, r::Int=1) -> DataFrame

Generates characteristic curves for each form based on the selected items and theta values.

# Arguments

  - `parms::Parameters`: Struct containing the system's parameters in the item bank.
  - `results::DataFrame`: DataFrame containing the selected items for each form.
  - `theta_range::AbstractVector`: Range of theta values (ability levels) to generate curves for.
  - `r::Int`: Exponent applied to the probabilities (default: 1).

# Returns

  - A `DataFrame` containing the characteristic curves for each form.
"""
function generate_characteristic_curves(parms::Parameters, results::DataFrame,
                                        theta_range::AbstractVector, r::Int=1)
    bank = parms.bank
    num_forms = size(results, 2)
    curves = DataFrame()

    for i in 1:num_forms
        selected_items = collect(skipmissing(results[:, i]))
        a, b, c = fetch_irt_parms(bank, selected_items)
        scores = map(theta -> sum(Probability.(theta, b, a, c) .^ r), theta_range)
        curves[!, names(results)[i]] = round.(scores, digits=2)
    end

    return curves
end

"""
    generate_information_curves(parms::Parameters, results::DataFrame, theta_range::AbstractVector) -> DataFrame

Generates information curves for each form based on the selected items and theta values.

# Arguments

  - `parms::Parameters`: Struct containing the system's parameters in the item bank.
  - `results::DataFrame`: DataFrame containing the selected items for each form.
  - `theta_range::AbstractVector`: Range of theta values (ability levels) to generate information curves for.

# Returns

  - A `DataFrame` containing the information curves for each form.
"""
function generate_information_curves(parms::Parameters, results::DataFrame,
                                     theta_range::AbstractVector)
    bank = parms.bank
    num_forms = size(results, 2)
    curves = DataFrame()

    for i in 1:num_forms
        selected_items = collect(skipmissing(results[:, i]))
        a, b, c = fetch_irt_parms(bank, selected_items)
        information = map(theta -> sum(Information.(theta, b, a, c)), theta_range)
        curves[!, names(results)[i]] = round.(information, digits=2)
    end

    return curves
end

"""
    write_results_to_file(curve_data::DataFrame, output_file::String)

Writes the DataFrame containing the characteristic or information curves to a CSV file.

# Arguments

  - `curve_data::DataFrame`: DataFrame containing the curves to be saved.
  - `output_file::String`: Path to the output CSV file.
"""
function write_results_to_file(curve_data::DataFrame, output_file::String)
    CSV.write(output_file, curve_data)
    return nothing
end

"""
    plot_results(parms::Parameters, conf::Config, results::DataFrame,
                 theta_range::AbstractVector=-3.0:0.1:3.0, plot_file::String="results/combined_plot.png") -> DataFrame

Generates and plots characteristic curves, information curves, and simulated observed scores,
and saves the combined plot and results to files.

# Arguments

  - `parms::Parameters`: Struct containing the system's parameters in the item bank.
  - `conf::Config`: Configuration struct containing file paths for saving results.
  - `results::DataFrame`: DataFrame containing the selected items for each form.
  - `theta_range::AbstractVector`: Range of theta values (default: -3.0:0.1:3.0) for generating the curves.
  - `plot_file::String`: Path to save the combined plot (default: "results/combined_plot.png").

# Returns

  - A `DataFrame` containing the characteristic curves for each form.
"""
function plot_results(parms::Parameters, conf::Config, results::DataFrame,
                      theta_range::AbstractVector=-3.0:0.1:3.0,
                      plot_file::String="results/combined_plot.png")::DataFrame

    # Step 1: Generate curves
    characteristic_curves, information_curves = generate_curves(parms, results, theta_range)

    # Step 2: Handle simulations based on TIC3 method logic
    simulation_data1, simulation_data2, simulation_data3 = generate_simulations(parms,
                                                                                results)

    # Step 3: Generate and combine plots
    combined_plot = create_combined_plot(parms, theta_range, characteristic_curves,
                                         information_curves, simulation_data1,
                                         simulation_data2, simulation_data3)

    # Step 4: Save results and plots to files
    save_results(characteristic_curves, theta_range, conf, combined_plot, plot_file)

    return characteristic_curves
end

"""
    generate_curves(parms::Parameters, results::DataFrame, theta_range::AbstractVector) -> Tuple{DataFrame, DataFrame}

Generates both characteristic curves and information curves.
"""
function generate_curves(parms::Parameters, results::DataFrame, theta_range::AbstractVector)
    characteristic_curves = generate_characteristic_curves(parms, results, theta_range)
    information_curves = generate_information_curves(parms, results, theta_range)
    return characteristic_curves, information_curves
end

"""
    generate_simulations(parms::Parameters, results::DataFrame) -> Tuple{DataFrame, DataFrame, DataFrame}

Handles simulation data generation based on the TIC3 method logic.
"""
function generate_simulations(parms::Parameters, results::DataFrame)
    if parms.method != "TIC3"
        simulation_data1 = simulate_observed_scores(parms, results, Normal(0.0, 1.0))
        simulation_data2 = simulate_observed_scores(parms, results, Normal(-1.0, 1.0))
        simulation_data3 = simulate_observed_scores(parms, results, Normal(1.0, 0.7))
    else
        simulation_data1 = simulate_observed_scores(parms, DataFrame(; data=results[:, 1]),
                                                    Normal(-1.0, 0.8))
        simulation_data2 = simulate_observed_scores(parms, DataFrame(; data=results[:, 2]),
                                                    Normal(0.0, 1.0))
        simulation_data3 = simulate_observed_scores(parms, DataFrame(; data=results[:, 3]),
                                                    Normal(1.0, 0.7))
    end
    return simulation_data1, simulation_data2, simulation_data3
end

"""
    create_combined_plot(parms::Parameters, theta_range::AbstractVector, characteristic_curves::DataFrame, information_curves::DataFrame,
                         simulation_data1::DataFrame, simulation_data2::DataFrame, simulation_data3::DataFrame) -> Plot

Creates and combines the characteristic, information, and simulation plots into a single layout.
"""
function create_combined_plot(parms::Parameters, theta_range::AbstractVector,
                              characteristic_curves::DataFrame,
                              information_curves::DataFrame,
                              simulation_data1::DataFrame, simulation_data2::DataFrame,
                              simulation_data3::DataFrame)

    # Load plotting libraries
    theme(:default)
    gr(; size=(950, 850))

    # Plot characteristic curves
    p1 = @df characteristic_curves plot(theta_range, cols(), title="Characteristic Curves",
                                        xlabel="θ", ylabel="Score", linewidth=2, label="")
    if parms.method == "TCC"
        p1 = scatter!(parms.theta, parms.tau[1, :]; label="", markersize=3)
    end

    # Plot information curves
    p2 = @df information_curves plot(theta_range, cols(), title="Information Curves",
                                     xlabel="θ", ylabel="Information", linewidth=2,
                                     label="")

    # Plot observed score simulations
    p3 = @df simulation_data1 plot(1:size(simulation_data1, 1), cols(),
                                   title="Observed Scores (Simulations)", xlabel="Items",
                                   ylabel="Percentage", linewidth=2, label="")
    p3 = @df simulation_data2 plot!(1:size(simulation_data2, 1), cols(), linewidth=2,
                                    label="")
    p3 = @df simulation_data3 plot!(1:size(simulation_data3, 1), cols(), linewidth=2,
                                    label="")

    # Combine plots into a single layout
    combined_plot = plot(p1, p2, p3; layout=(2, 2), size=(950, 850), margin=8mm)

    return combined_plot
end

"""
    save_results(characteristic_curves::DataFrame, theta_range::AbstractVector, conf::Config, combined_plot::Plot, plot_file::String)

Saves the characteristic curves to a file and the combined plot to a PNG.
"""
function save_results(characteristic_curves::DataFrame, theta_range::AbstractVector,
                      conf::Config, combined_plot::AbstractPlot, plot_file::String)
    # Add theta range to characteristic curves and save
    insertcols!(characteristic_curves, 1, :THETA => collect(theta_range))
    write_results_to_file(characteristic_curves, conf.tcc_file)

    # Save the combined plot
    savefig(combined_plot, plot_file)

    println("TCC data saved to: ", conf.tcc_file)
    println("Charts saved to: ", plot_file)
    return nothing
end

end
