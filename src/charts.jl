using Plots
using StatsPlots
using DataFrames

include("types.jl")
include("stats_functions.jl")

# Function to extract a, b, c parameters from the bank based on selected items
function get_item_parameters(bank::DataFrame, selected_items::Vector{Union{Missing, String}})
    # items = in(skipmissing(selected_items)).(bank.CLAVE)
    items = bank.CLAVE .∈ Ref(skipmissing(selected_items))
    return get_item_parameters(bank, items)
end

function get_item_parameters(bank::DataFrame, selected_items::BitVector)
    a = bank[selected_items, :A]
    b = bank[selected_items, :B]
    c = bank[selected_items, :C]
    return a, b, c
end

function get_item_parameters(bank::DataFrame, selected_items::Vector{Int})
    a = bank[selected_items, :A]
    b = bank[selected_items, :B]
    c = bank[selected_items, :C]
    return a, b, c
end

# Generate results simulation based on ability distribution
function generate_results_simulation(parameters::Params, results::DataFrame, ability_dist::Distribution = Normal(0.0, 1.0))
    bank = parameters.bank
    observed_dist = DataFrame()
    max_length, num_versions = size(results)
    max_length += 1
    column_names = names(results)

    for i in 1:num_versions
        selected_items = results[:, i]
        a, b, c = get_item_parameters(bank, selected_items)

        item_params::Matrix{Float64} = hcat(a, b, c)'
        dist = observed_score_distribution_continuous(item_params, ability_dist)
        padded_dist = vcat(dist, fill(missing, max_length - length(dist)))
        observed_dist[!, column_names[i]] = padded_dist
    end

    return observed_dist
end

# Generate characteristic curves for each version
function generate_characteristic_curves(parameters, results::DataFrame, theta_range::AbstractVector, r::Int=1)
    bank = parameters.bank
    num_versions = size(results, 2)
    curves = DataFrame()

    for i in 1:num_versions
        selected_items = results[:, i]
        a, b, c = get_item_parameters(bank, selected_items)
        scores = map(theta -> sum(Pr.(theta, b, a, c) .^ r), theta_range)
        curves[!, names(results)[i]] = round.(scores, digits=2)
    end

    return curves
end

# Generate characteristic curves for each version
function generate_information_curves(parameters, results::DataFrame, theta_range::AbstractVector)
    bank = parameters.bank
    num_versions = size(results, 2)
    curves = DataFrame()

    for i in 1:num_versions
        selected_items = results[:, i]
        a, b, c = get_item_parameters(bank, selected_items)
        information = map(theta -> sum(item_information.(theta, b, a, c)), theta_range)
        curves[!, names(results)[i]] = round.(information, digits=2)
    end

    return curves
end

# Function to write results to file
function write_results_to_file(curve_data, output_file="data/tcc.csv")
    println("Writing results to file: ", output_file)
    CSV.write(output_file, curve_data)
end


# Generate characteristic curves and observed score distribution plots
function plot_characteristic_curves_and_simulation(parameters, results::DataFrame, theta_range::AbstractVector = -3.0:0.1:3.0, plot_file::String = "data/combined_plot.png")
    # Generate the plot data
    characteristic_curves = generate_characteristic_curves(parameters, results, theta_range)
    information_curves = generate_information_curves(parameters, results, theta_range)
    simulation_data = generate_results_simulation(parameters, results, Normal(0.0, 1.0))

    # Set up the plot aesthetics
    theme(:mute)
    gr(size=(900, 750))

    # Create subplots
    p1 = @df characteristic_curves plot(theta_range, cols(),
                                        title="Characteristic Curves",
                                        xlabel="Theta", ylabel="Score",
                                        linewidth=2, label="",
                                        grid=(:on, :olivedrab, :dot, 1, 0.9),
                                        tickfontsize=12)
    if parameters.method == "TCC"
        p1 = scatter!(parameters.theta,
                      parameters.tau[1, :],
                      label="")
    end

    p2 = @df information_curves plot(theta_range, cols(),
                                     title="Information Curves",
                                     xlabel="Theta", ylabel="Information",
                                     linewidth=2, label="",
                                     grid=(:on, :olivedrab, :dot, 1, 0.9),
                                     tickfontsize=12)

    p3 = @df simulation_data plot(1:size(simulation_data, 1), cols(),
                                  title="Observed Score Distribution",
                                  xlabel="Item", ylabel="Score",
                                  linewidth=2, label="",
                                  grid=(:on, :olivedrab, :dot, 1, 0.9),
                                  tickfontsize=12)

    # Combine the plots into a single image with subplots
    combined_plot = plot(p1, p2, p3, layout=(2, 2), size=(900, 750))

    write_results_to_file(hcat(theta_range, characteristic_curves))

    # Save the combined plot
    savefig(combined_plot, plot_file)
end
