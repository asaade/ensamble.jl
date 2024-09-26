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

# Function to extract a, b, c parameters from the bank based on selected items
function fetch_irt_parms(bank::DataFrame, selected_items::Vector{String})
    selected_items_idx = bank.ID .∈ Ref(skipmissing(selected_items))
    a = bank[selected_items_idx, :A]
    b = bank[selected_items_idx, :B]
    c = bank[selected_items_idx, :C]
    return a, b, c
end

# Simulate results based on ability distribution
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

        # Simulate the observed scores based on item parameters
        item_params::Matrix{Float64} = hcat(a, b, c)'  # Transpose for matrix structure
        dist = observed_score_distribution_continuous(item_params, ability_dist)
        padded_dist = vcat(dist, fill(missing, max_length - length(dist)))
        observed_dist[!, column_names[i]] = padded_dist
    end

    return observed_dist
end

# Generate characteristic curves for each form
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

# Generate information curves for each form
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

# Function to write results to a CSV file
function write_results_to_file(curve_data::DataFrame, output_file::String)
    CSV.write(output_file, curve_data)
    return nothing
end

# Main function to generate and plot characteristic curves, information curves, and simulation data
function plot_results(parms::Parameters, conf::Config, results::DataFrame,
                      theta_range::AbstractVector=-3.0:0.1:3.0,
                      plot_file::String="results/combined_plot.pdf")::DataFrame
    # Generate plot data
    characteristic_curves = generate_characteristic_curves(parms, results, theta_range)
    information_curves = generate_information_curves(parms, results, theta_range)

    # Handle special logic for TIC3 method
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

    # Set up a light theme with bright colors
    theme(:default)
    gr(; size=(750, 1200))  # Adjust size ratio to (750, 1200)

    # Create subplots with individual titles
    p1 = @df characteristic_curves plot(theta_range, cols(),
                                        title="Characteristic Curves",
                                        xlabel="θ", ylabel="Score", linewidth=2,
                                        label="",
                                        grid=(:on, :lightgray, :solid, 1, 0.9),
                                        tickfontsize=12, titlefontsize=16)

    if parms.method == "TCC"
        p1 = scatter!(parms.theta, parms.tau[1, :]; label="", markersize=5)
    end

    p2 = @df information_curves plot(theta_range, cols(), title="Information Curves",
                                     xlabel="θ", ylabel="Information", linewidth=2,
                                     label="", grid=(:on, :lightgray, :solid, 1, 0.9),
                                     tickfontsize=12, titlefontsize=16)

    # Only one simulation chart with all variations
    p3 = @df simulation_data1 plot(1:size(simulation_data1, 1), cols(),
                                   title="Observed Scores (Simulations)",
                                   xlabel="Items", ylabel="Percentage", linewidth=2,
                                   label="", grid=(:on, :lightgray, :solid, 1, 0.9),
                                   tickfontsize=12, titlefontsize=16)

    # Add the other simulations to the same plot
    p3 = @df simulation_data2 plot!(1:size(simulation_data2, 1), cols(), linewidth=2,
                                    label="")
    p3 = @df simulation_data3 plot!(1:size(simulation_data3, 1), cols(), linewidth=2,
                                    label="")

    # Combine plots into a 3-row layout with increased margins and both individual and general titles
    combined_plot = plot(p1, p2, p3; layout=(3, 1), size=(750, 1200),
                         # title="Combined IRT Analysis",
                         margin=15mm)

    # Write results to file
    insertcols!(characteristic_curves, 1, :THETA => collect(theta_range))
    write_results_to_file(characteristic_curves, conf.tcc_file)
    println("TCC data saved to: ", conf.tcc_file)
    # Save the combined plot
    savefig(combined_plot, plot_file)
    println("Charts saved to: ", plot_file)
    return characteristic_curves
end

end
