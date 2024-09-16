using DataFrames
using JuMP
using Logging, LoggingExtras

logger = TeeLogger(
    # Current global logger (stderr)
    global_logger(),
    # Accept any messages with level >= Info
    MinLevelLogger(
        FileLogger("logfile.log"),
        Logging.Info
    ),
    # Accept any messages with level >= Debug
    MinLevelLogger(
        FileLogger("debug.log"),
        Logging.Debug,
    ),
)

# Include external modules

include("src/configuration.jl")
using .Configuration

include("src/constants.jl")
include("src/model/constraints.jl")
include("src/model/criteria_parser.jl")
include("src/model/model_initializer.jl")
include("src/model/solvers.jl")
include("src/display/charts.jl")
include("src/display/display_results.jl")

"""
    load_configuration(config_file::String)

Load the configuration and parameters from the specified YAML configuration file.
"""
function load_configuration(config_file::String)
    println(LOADING_CONFIGURATION_MESSAGE)
    return Configuration.configure(config_file)
end

"""
    run_optimization(model::Model)

Run the optimization solver and check if the model is solved and feasible.
"""
function run_optimization(model::Model)
    println(RUNNING_OPTIMIZATION_MESSAGE)
    optimize!(model)
    return is_solved_and_feasible(model)
end

"""
    remove_used_items!(parms::Parameters, items_used)

Remove items used in forms from the bank and update probabilities or
information for the remaining items based on the method used.
"""
function remove_used_items!(parms::Parameters, items_used::Vector{Int})
    remaining = setdiff(1:length(parms.bank.ID), items_used)
    parms.bank = parms.bank[remaining, :]
    if parms.method in ["TCC", "TIC", "TIC2", "TIC3"]
        parms.method in ["TCC"] && (parms.p = parms.p[remaining, :])
        parms.method in ["TIC", "TIC2", "TIC3"] &&
            (parms.info = parms.info[remaining, :])
    end
    return parms
end

"""
    generate_unique_column_name(results_df::DataFrame)

Generate a unique column name to avoid naming conflicts in the results_df DataFrame.
"""
function generate_unique_column_name(results_df::DataFrame)
    i = 1
    while "Form_$i" in names(results_df)
        i += 1
    end
    return "Form_$i"
end

"""
    process_and_store_results!(model::Model, parms::Parameters, results_df::DataFrame)

Process the optimization results_df for each form and store them in a DataFrame.
Used items are removed from the bank for subsequent forms.
"""
function process_and_store_results!(model::Model, parms::Parameters, results_df::DataFrame)
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
        if missing_rows > 0
            padded_codes_vector = vcat(codes_in_form,
                                       fill(MISSING_VALUE_FILLER, missing_rows))
        else
            padded_codes_vector = codes_in_form
        end
        results_df[!, generate_unique_column_name(results_df)] = padded_codes_vector
        items_used = vcat(items_used, selected_items)
    end

    items_used = sort(unique(items_used))
    parms.bank[items_used, :ITEM_USE] .+= 1
    items_used = items_used[parms.bank[items_used, :ITEM_USE] .>= parms.max_item_use]
    remove_used_items!(parms, items_used)
    return results_df
end

"""
    handle_anchor_items(parms::Parameters, old_par::Parameters)

Handle anchor items by adjusting the bank and relevant parms based on the
specified anchor number.
"""
function handle_anchor_items(parms::Parameters, old_par::Parameters)
    if parms.anchor_tests > 0
        parms.anchor_tests = parms.anchor_tests % old_par.anchor_tests + 1
        bank = parms.bank[parms.bank.ANCHOR .== 0, :]
        anchors = old_par.bank[old_par.bank.ANCHOR .== parms.anchor_tests,
                               :]
        parms.bank = vcat(bank, anchors)

        if parms.method in ["TCC"]
            parms.p = old_par.p[parms.bank.INDEX, :]
        elseif parms.method in ["TIC", "TIC2", "TIC3"]
            parms.info = old_par.info[parms.bank.INDEX, :]
        end
    end
end

"""
    main(config_file::String=CONFIG_FILE)

Main function to run the entire optimization process, including loading configurations,
running the solver, processing results_df, and saving the output.
"""
function main(config_file::String = CONFIG_FILE)
    config, old_par = load_configuration(config_file)
    parms = deepcopy(old_par)
    constraints = read_constraints(config.constraints_file, parms)
    results_df = DataFrame()
    if parms.shadow_test > 0
        parms.num_forms = 1
    end

    assembled_forms = 0

    while parms.f > 0
        parms.num_forms = min(parms.num_forms, parms.f)
        parms.shadow_test = max(0, parms.f - parms.num_forms)
        handle_anchor_items(parms, old_par)

        model = Model()
        configure_solver!(model, parms, config.solver)
        initialize_model!(model, parms, constraints)

        if run_optimization(model)
            results_df = process_and_store_results!(model, parms, results_df)
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

    ## Display results_df and save
    return final_report(old_par, results_df, config)
end

# Uncomment to run the main function
# main(CONFIG_FILE)
