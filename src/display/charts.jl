using DataFrames
using Plots
using StatsPlots
using Distributions
using .Configuration


# Function to extract a, b, c parms from the bank based on selected items
function fetch_irt_parms(bank::DataFrame,
                         selected_items::Vector{String})
    selected_items = bank.ID .∈ Ref(skipmissing(selected_items))
    a = bank[selected_items, :A]
    b = bank[selected_items, :B]
    c = bank[selected_items, :C]
    return a, b, c
end

# Generate results simulation based on ability distribution
function simulate_observed_scores(parms::Parameters,
                                  results::DataFrame,
                                  ability_dist::Distribution = Normal(0.0, 1.0))
    bank = parms.bank
    observed_dist = DataFrame()
    max_length, num_forms = size(results)
    max_length += 1
    column_names = names(results)

    for i in 1:num_forms
        selected_items = results[:, i]
        a, b, c = fetch_irt_parms(bank, selected_items)

        item_params::Matrix{Float64} = hcat(a, b, c)'
        dist = observed_score_distribution_continuous(item_params, ability_dist)
        padded_dist = vcat(dist, fill(missing, max_length - length(dist)))
        observed_dist[!, column_names[i]] = padded_dist
    end

    return observed_dist
end

# Generate characteristic curves for each form
function generate_characteristic_curves(parms::Parameters,
                                        results::DataFrame,
                                        theta_range::AbstractVector,
                                        r::Int = 1)
    bank = parms.bank
    num_forms = size(results, 2)
    curves = DataFrame()

    for i in 1:num_forms
        selected_items = results[:, i]
        a, b, c = fetch_irt_parms(bank, selected_items)
        scores = map(theta -> sum(Probability.(theta, b, a, c) .^ r), theta_range)
        curves[!, names(results)[i]] = round.(scores, digits = 2)
    end

    return curves
end

# Generate characteristic curves for each form
function generate_information_curves(parms,
                                     results::DataFrame,
                                     theta_range::AbstractVector)
    bank = parms.bank
    num_forms = size(results, 2)
    curves = DataFrame()

    for i in 1:num_forms
        selected_items = results[:, i]
        a, b, c = fetch_irt_parms(bank, selected_items)
        information = map(theta -> sum(Information.(theta, b, a, c)), theta_range)
        curves[!, names(results)[i]] = round.(information, digits = 2)
    end

    return curves
end

# Function to write results to file
function write_results_to_file(curve_data, output_file = "results/tcc.csv")
    # println("Writing results to file: ", output_file)
    return CSV.write(output_file, curve_data)
end

# Generate characteristic curves and observed score distribution plots
function plot_results_and_simulation(parms,
                                     results::DataFrame,
                                     theta_range::AbstractVector = -3.0:0.1:3.0,
                                     plot_file::String = "results/combined_plot.pdf")
    # Generate the plot data
    characteristic_curves = generate_characteristic_curves(parms, results, theta_range)
    information_curves = generate_information_curves(parms, results, theta_range)
    if parms.method != "TIC3"
        simulation_data1 = simulate_observed_scores(parms, results, Normal(0.0, 1.0))
        simulation_data2 = simulate_observed_scores(parms, results, Normal(-1.0, 1.0))
        simulation_data3 = simulate_observed_scores(parms, results, Normal(1.0, 0.7))
    elseif parms.method == "TIC3"
        simulation_data1 = simulate_observed_scores(parms, DataFrame(data = results[:, 1]),
                                                    Normal(-1.0, 0.8))
        simulation_data2 = simulate_observed_scores(parms, DataFrame(data = results[:, 2]),
                                                    Normal(0.0, 1.0))
        simulation_data3 = simulate_observed_scores(parms, DataFrame(data = results[:, 3]),
                                                    Normal(1.0, 0.7))
    end

    # Set up the plot aesthetics
    theme(:dark)
    gr(; size = (1200, 900))

    forms = size(results, 2)

    # Create subplots
    p1 = @df characteristic_curves plot(theta_range,
                                        cols(),
                                        title = "Characteristic Curves for $forms forms",
                                        xlabel = "θ",
                                        ylabel = "Score",
                                        linewidth = 2,
                                        label = "",
                                        grid = (:on, :olivedrab, :dot, 1, 0.9),
                                        tickfontsize = 12,
                                        titlefontsize = 16)

    if parms.method == "TCC"
        p1 = scatter!(parms.theta, parms.tau[1, :]; label = "", markersize = 5)
    end

    p2 = @df information_curves plot(theta_range,
                                     cols(),
                                     title = "Information Curves",
                                     xlabel = "θ",
                                     ylabel = "Information",
                                     linewidth = 2,
                                     label = "",
                                     grid = (:on, :olivedrab, :dot, 1, 0.9),
                                     tickfontsize = 12,
                                     titlefontsize = 16)

    p3 = @df simulation_data1 plot(1:size(simulation_data1, 1),
                                   cols(),
                                   title = "Expected Observed Scores\nfor N(-1, 1), N(0, 1), N(1, 0.7)",
                                   xlabel = "Items",
                                   ylabel = "Percentage",
                                   linewidth = 2,
                                   label = "",
                                   grid = (:on, :olivedrab, :dot, 1, 0.9),
                                   tickfontsize = 12,
                                   titlefontsize = 16)

    p3 = @df simulation_data2 plot!(1:size(simulation_data2, 1),
                                    cols(),
                                    linewidth = 2,
                                    label = "")

    p3 = @df simulation_data3 plot!(1:size(simulation_data3, 1),
                                    cols(),
                                    linewidth = 2,
                                    label = "")

    # Combine the plots into a single image with subplots
    combined_plot = plot(p1, p2, p3; layout = (2, 2), size = (1200, 900))

    write_results_to_file(hcat(theta_range, characteristic_curves))

    # Save the combined plot
    savefig(combined_plot, plot_file)
    return combined_plot
end
