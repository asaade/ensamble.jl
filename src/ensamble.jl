module Ensamble
__precompile__()

export assemble_tests



# Import necessary packages
using JuMP
using DataFrames
using Logging, LoggingExtras
using PrettyTables

include("constants.jl")

# Include external modules
include("utils/utils.jl")
include("config/configuration.jl")
include("display/display_results.jl")
include("model/model_initializer.jl")
using .Utils
using .Configuration
using .DisplayResults
using .ModelInitializer

using Logging: Logging
# Set global log level (e.g., show info, warnings and errors)
Logging.global_logger(Logging.ConsoleLogger(stderr, Logging.Info))

"""
load_configuration(config_file::String)::Tuple{Dict, Parameters}

Load the configuration from a TOML file and return the configuration and parameters.
"""
function configure_system(config_file::String)
    println(LOADING_CONFIGURATION_MESSAGE)
    return Configuration.configure(config_file)
end

# Run the optimization solver
"""
    run_optimization(model::Model)

Run the optimization solver and check if the model is solved and feasible.
"""
function run_optimization(model::Model)::Bool
    @info RUNNING_OPTIMIZATION_MESSAGE
    optimize!(model)
    return is_solved_and_feasible(model)
end

# Remove used items from the bank
"""
    remove_used_items!(parms::Parameters, items_used::Vector{Int})

Remove items used in forms from the bank and update probabilities or
information for the remaining items based on the method used.
"""
function remove_used_items!(parms::Parameters, items_used::Vector{Int})::Parameters
    remaining = setdiff(1:length(parms.bank.ID), items_used)
    parms.bank = parms.bank[remaining, :]

    if parms.method == "TCC"
        parms.p = parms.p[remaining, :]
    elseif parms.method in ["TIC", "TIC2", "TIC3"]
        parms.info = parms.info[remaining, :]
    else
        error("Unknown $(parms.method) optimization method used.")
    end

    return parms
end

# Generate a unique column name for DataFrame
"""
    generate_unique_column_name(results_df::DataFrame)

Generate a unique column name to avoid naming conflicts in the results DataFrame.
"""
function generate_unique_column_name(results_df::DataFrame)::String
    i = 1
    while "Form_$i" in names(results_df)
        i += 1
    end
    return "Form_$i"
end

# Process optimization results and store them
"""
    process_and_store_results!(model::Model, parms::Parameters, results_df::DataFrame)

Process the optimization results at each iteration and store them in a DataFrame.
Used items ARE DELETED from the working copy of the bank to avoid its use in subsequent forms.
"""
function process_and_store_results!(model::Model, parms::Parameters,
                                    results_df::DataFrame)::DataFrame
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
            vcat(codes_in_form,
                 fill(MISSING_VALUE_FILLER, missing_rows))
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

Cycles through anchor tests, updates the bank by removing the old anchor items and adding the new anchor items,
and updates the relevant parameters (`p` or `info`) depending on the method in use.
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

        # Remove old anchor items and add the new anchor test items
        bank_without_anchors = filter(row -> ismissing(row.ANCHOR), parms.bank)
        new_anchors = filter(row -> !ismissing(row.ANCHOR) && row.ANCHOR == parms.anchor_tests, orig_parms.bank)
        parms.bank = vcat(bank_without_anchors, new_anchors)

        # Update parameters based on the method
        if parms.method == "TCC"
            if "INDEX" in names(parms.bank)
                parms.p = orig_parms.p[parms.bank.INDEX, :]
            else
                error("INDEX column missing in the bank for TCC method.")
            end
        elseif parms.method in ["TIC", "TIC2", "TIC3"]
            if "INDEX" in names(parms.bank)
                parms.info = orig_parms.info[parms.bank.INDEX, :]
            else
                error("INDEX column missing in the bank for TIC method.")
            end
        else
            error("Unsupported method: $(parms.method)")
        end
    end
    return parms
end

# Main function to run the optimization process
"""
    assemble_tests(config_file::String="path_to_config_file.toml")

Main entry point for assembling tests. Loads configurations, runs the solver,
and processes the results, then generates and saves a report.
"""
function assemble_tests(config_file::String="data/config.toml")::DataFrame
    config, orig_parms = configure_system(config_file)


    # validate_parameters(orig_parms)

    parms = deepcopy(orig_parms)
    constraints = read_constraints(config.constraints_file, parms)
    results_df = DataFrame()

    if parms.shadow_test > 0
        parms.num_forms = 1
    end

    assembled_forms = 0
    tolerances::Vector{Float64} = []

    while parms.f > 0
        parms.num_forms = min(parms.num_forms, parms.f)
        parms.shadow_test = max(0, parms.f - parms.num_forms)

        handle_anchor_items(parms, orig_parms)

        model = Model()
        configure_solver!(model, parms, config.solver)
        initialize_model!(model, parms, constraints)

        if run_optimization(model)
            results_df = process_and_store_results!(model, parms, results_df)
            tolerances = vcat(tolerances, round(objective_value(model); digits=4))
            display_results(model, parms)
            assembled_forms += parms.num_forms
            parms.f -= parms.num_forms
            println("Forms assembled: $assembled_forms")
            println("Forms remaining: $(parms.f)")

        else
            println(OPTIMIZATION_FAILED_MESSAGE)
            if parms.verbose > 1
                display(check_constraints(model))
                return 1
            end
            parms.f -= 1
        end
    end

    # Assuming you have the required parameters, results, config, and tolerances
    report_data = final_report(orig_parms, results_df, config, tolerances)

    # Generate the report as a string
    report = generate_report(report_data)

    # Optionally, save the report to a file
    write("results/test_assembly_report.txt", report)

    return results_df
end

end  # module Ensamble
