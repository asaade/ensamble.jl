using DataFrames
using JuMP

# Include external modules
include("constants.jl")
include("display_results.jl")
include("get_data.jl")
include("model_initializer.jl")
include("solvers.jl")

"""
    load_configuration(config_file::String)

Load the configuration and parameters from the specified YAML configuration file.
"""
function load_configuration(config_file::String)
    println(LOADING_CONFIGURATION_MESSAGE)
    config = load_config(config_file)
    parameters = get_params(config)
    return config, parameters
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
    remove_used_items!(parameters::Params, used_items)

Remove items used in forms from the bank and update probabilities or
information for the remaining items based on the method used.
"""
function remove_used_items!(parameters::Params, used_items)
    remaining = setdiff(1:length(parameters.bank.CLAVE), used_items)
    parameters.bank = parameters.bank[remaining, :]
    if parameters.method in ["TCC", "TIC", "TIC2", "TIC3"]
        parameters.method in ["TCC"] && (parameters.p = parameters.p[remaining, :])
        parameters.method in ["TIC", "TIC2", "TIC3"] &&
            (parameters.info = parameters.info[remaining, :])
    end
    return parameters
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
    process_and_store_results!(model::Model, parameters::Params, results_df::DataFrame)

Process the optimization results_df for each form and store them in a DataFrame.
Used items are removed from the bank for subsequent forms.
"""
function process_and_store_results!(model::Model, parameters::Params, results_df::DataFrame)
    solver_matrix = value.(model[:x])
    item_codes = parameters.bank.CLAVE
    items = 1:length(item_codes)
    used_items = Int[]
    max_items = parameters.max_items

    for form_name in 1:(parameters.num_forms)
        selected_items = items[solver_matrix[:, form_name] .> 0.5]
        item_codes_in_form = item_codes[selected_items]
        form_length = length(item_codes_in_form)
        missing_rows = max_items - form_length
        padded_item_codes = vcat(item_codes_in_form,
                                 fill(MISSING_VALUE_FILLER, missing_rows))

        results_df[!, generate_unique_column_name(results_df)] = padded_item_codes
        used_items = vcat(used_items, selected_items)
    end

    used_items = sort(unique(used_items))
    remove_used_items!(parameters, used_items)
    return results_df
end

"""
    handle_anchor_items(parameters::Params, original_parameters::Params)

Handle anchor items by adjusting the bank and relevant parameters based on the
specified anchor number.
"""
function handle_anchor_items(parameters::Params, original_parameters::Params)
    if parameters.anchor_number > 0
        parameters.anchor_number = mod(parameters.anchor_number,
                                       original_parameters.anchor_number) + 1
        bank = parameters.bank[parameters.bank.ANCHOR .== 0, :]
        anchors = original_parameters.bank[original_parameters.bank.ANCHOR .== parameters.anchor_number,
                                           :]
        parameters.bank = vcat(bank, anchors)

        if parameters.method in ["TCC"]
            parameters.p = original_parameters.p[parameters.bank.INDEX, :]
        elseif parameters.method in ["TIC", "TIC2", "TIC3"]
            parameters.info = original_parameters.info[parameters.bank.INDEX, :]
        end
    end
end

"""
    main(config_file::String=CONFIG_FILE)

Main function to run the entire optimization process, including loading configurations,
running the solver, processing results_df, and saving the output.
"""
function main(config_file::String=CONFIG_FILE)
    config, original_parameters = load_configuration(config_file)
    parameters = deepcopy(original_parameters)
    constraints = read_constraints(config.constraints_file)
    results_df = DataFrame()

    while parameters.f > 0
        parameters.num_forms = min(parameters.num_forms, parameters.f)
        parameters.shadow_test = max(0, parameters.f - parameters.num_forms)
        handle_anchor_items(parameters, original_parameters)

        model = Model()
        configure_solver!(model, parameters, config.solver)
        initialize_model!(model, parameters, original_parameters, constraints)

        if run_optimization(model)
            results_df = process_and_store_results!(model, parameters, results_df)
            display_results(model, parameters)
            parameters.f -= parameters.num_forms
        else
            println(OPTIMIZATION_FAILED_MESSAGE)
            # parameters.verbose > 0 && conflicting_constraints(model)
            parameters.f = 0
        end
    end

    ## Display results_df and save
    return final_report(original_parameters, results_df, config)
end

# Uncomment to run the main function
# main(CONFIG_FILE)
