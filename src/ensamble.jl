module Ensamble

export assemble_tests, load_irt_data

using Revise
# Import necessary packages
using JuMP
using DataFrames
using Logging
using PrettyTables

include("constants.jl")

# Include local modules (this loads the code from each file)
include("utils/utils.jl")
include("config/configuration.jl")
include("display/display_results.jl")
include("model/model_initializer.jl")

# Track local files with Revise
# Revise.track("src/utils/utils.jl")
# Revise.track("src/config/configuration.jl")
# Revise.track("src/display/display_results.jl")
# Revise.track("src/model/model_initializer.jl")

# Use the included modules
using .Utils
using .Configuration
using .DisplayResults
using .ModelInitializer
# Set global log level
Logging.global_logger(Logging.ConsoleLogger(stderr, Logging.Info))

"""
    configure_system(config_file::String)::Tuple{Dict, Parameters}

Load system configuration from a TOML file and return the configuration and parameters.
"""
function configure_system(config_file::String)
    println(LOADING_CONFIGURATION_MESSAGE)
    return Configuration.configure(config_file)
end

"""
    run_optimization(model::Model)::Bool

Run the optimization solver and check if the model is solved and feasible.
"""
function run_optimization(model::Model)::Bool
    @info RUNNING_OPTIMIZATION_MESSAGE
    optimize!(model)
    return is_solved_and_feasible(model)
end

"""
    remove_used_items!(parms::Parameters, items_used::Vector{Int})::Parameters

Remove the items used in forms from the bank and update probabilities or
information for the remaining items based on the method used.
"""
function remove_used_items!(parms::Parameters, items_used::Vector{Int})::Parameters
    remaining = setdiff(1:length(parms.bank.ID), items_used)
    parms.bank = parms.bank[remaining, :]

    if parms.method in ["TCC", "TCC2"]
        parms.score_matrix = parms.score_matrix[remaining, :]
    elseif parms.method in ["TIC", "TIC2", "TIC3"]
        parms.info_matrix = parms.info_matrix[remaining, :]
    elseif parms.method == "MIXED"
        parms.score_matrix = parms.score_matrix[remaining, :]
        parms.info_matrix = parms.info_matrix[remaining, :]
    else
        error("Unknown $(parms.method) optimization method used.")
    end

    return parms
end

"""
    generate_unique_column_name(results_df::DataFrame)::String

Generate a unique column name to avoid naming conflicts in the results DataFrame.
"""
function generate_unique_column_name(results_df::DataFrame)::String
    i = 1
    while "Form_$i" in names(results_df)
        i += 1
    end
    return "Form_$i"
end

"""
    process_and_store_results!(model::Model, parms::Parameters, results_df::DataFrame)

Process the optimization results at each iteration and store them in a DataFrame.
Removes used items from the working copy of the bank to avoid its use in subsequent forms.
"""
function process_and_store_results!(
        model::Model, parms::Parameters, results_df::DataFrame
)::DataFrame
    solver_matrix = value.(model[:x])
    item_codes = parms.bank.ID
    items = 1:length(item_codes)
    items_used = Int[]
    max_items = parms.max_items

    for f in 1:(parms.num_forms)
        selected_items = items[solver_matrix[:, f] .> 0.9]
        codes_in_form = item_codes[selected_items]
        form_length = length(codes_in_form)
        missing_rows = max_items - form_length

        padded_codes_vector = if missing_rows > 0
            vcat(codes_in_form, fill(MISSING_VALUE_FILLER, missing_rows))
        else
            codes_in_form
        end
        results_df[!, generate_unique_column_name(results_df)] = padded_codes_vector

        # Directly append the selected items to items_used
        append!(items_used, selected_items)
    end

    # Ensure uniqueness and sort the items after collecting them
    items_used = sort(unique(items_used))

    parms.bank[items_used, :ITEM_USE] .+= 1
    items_used = items_used[parms.bank[items_used, :ITEM_USE] .>= parms.max_item_use]

    remove_used_items!(parms, items_used)
    return results_df
end

"""
    handle_anchor_items(parms::Parameters, orig_parms::Parameters)::Parameters

Update the item bank used for the next iteration
by removing the old anchor items and adding the new anchor items iterating on anchor forms.
Also retrives the original score_matrix or info_matrix for these items, depending on the objective in use.
"""
function handle_anchor_items(parms::Parameters, orig_parms::Parameters)::Parameters
    # Ensure anchor tests are available and cycle through them
    if parms.anchor_tests > 0
        # Cycle to the next anchor test
        total_anchors = orig_parms.anchor_tests
        if total_anchors == 0
            error("No anchor tests available to cycle through.")
        end

        parms.anchor_tests = (parms.anchor_tests % total_anchors) + 1

        # Select current anchor items based on the anchor test value
        current_anchor_items = filter(
            row -> !ismissing(row.ANCHOR) && row.ANCHOR == parms.anchor_tests,
            orig_parms.bank
        )
        non_anchor_items = filter(row -> ismissing(row.ANCHOR), parms.bank)

        # Combine non-anchor items and current anchor items
        parms.bank = vcat(non_anchor_items, current_anchor_items)

        # Update parameters based on the method
        if parms.method in ["TCC", "TCC2"]
            parms.score_matrix = orig_parms.score_matrix[parms.bank.INDEX, :]
        elseif parms.method in ["TIC", "TIC2", "TIC3"]
            parms.info_matrix = orig_parms.info_matrix[parms.bank.INDEX, :]
        elseif parms.method == "MIXED"
            parms.score_matrix = orig_parms.score_matrix[parms.bank.INDEX, :]
            parms.info_matrix = orig_parms.info_matrix[parms.bank.INDEX, :]
        else
            error("Unsupported method: $(parms.method)")
        end
    end
    return parms
end

"""
    assemble_tests(config_file::String="path_to_config_file.toml")

Main entry point for assembling tests. Loads configurations, runs the solver,
and processes the results, then generates and saves a report.
"""
function assemble_tests(config_file::String = "data/config.toml")::DataFrame
    config, orig_parms = configure_system(config_file)

    # validate_parameters(orig_parms)

    constraints = read_constraints(config.constraints_file, orig_parms)
    parms = deepcopy(orig_parms)
    results_df = DataFrame()

    if parms.shadow_test_size > 0
        parms.num_forms = 1
    end

    assembled_forms = 0
    tolerances::Vector{Float64} = []

    while parms.f > 0
        parms.num_forms = min(parms.num_forms, parms.f)
        parms.shadow_test_size = max(0, parms.f - parms.num_forms)

        handle_anchor_items(parms, orig_parms)

        model = Model()
        configure_solver!(model, parms, config.solver)
        initialize_model!(model, parms, constraints)

        if run_optimization(model)
            results_df = process_and_store_results!(model, parms, results_df)
            tolerances = vcat(tolerances, round(objective_value(model); digits = 4))
            show_results(model, parms)
            assembled_forms += parms.num_forms
            parms.f -= parms.num_forms
            println("Forms assembled: $assembled_forms")
            println("Forms remaining: $(parms.f)\n\n")

        else
            println(OPTIMIZATION_FAILED_MESSAGE)
            if parms.verbose > 1
                display(check_constraints(model))
                return 1
            end
            parms.f -= 1
        end
    end

    report_data = final_report(orig_parms, results_df, config, tolerances)

    # Generate the report as a string
    report = generate_report(report_data)

    # Optionally, save the report to a file
    write("results/test_assembly_report.txt", report)

    return results_df
end

end  # module Ensamble
