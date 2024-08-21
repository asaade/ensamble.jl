using DataFrames
using JuMP

# Include external modules
include("get_data.jl")
include("solvers.jl")
include("stats_functions.jl")
include("charts.jl")
include("constants.jl")
include("model_initializer.jl")

# Print functions
function print_title_and_separator(title::String)
    println(title)
    return println(SEPARATOR)
end

function print_optimization_results(model::Model, parameters::Params)
    parameters.verbose > 1 && println(solution_summary(model))
    println(TOLERANCE_LABEL, round(objective_value(model); digits = 4))
    return println(SEPARATOR)
end

"""
    calculate_common_items(results::DataFrame)

Calculate the number of common items between versions and return a matrix
where each entry [i, j] indicates the number of common items between version `i` and version `j`.
"""
function calculate_common_items(results::DataFrame)
    num_forms = size(results, 2)
    common_items_matrix = zeros(Int, num_forms, num_forms)

    for i in 1:num_forms, j in 1:num_forms
        common = in(skipmissing(results[:, i])).(skipmissing(results[:, j]))
        common_items_matrix[i, j] = sum(common)
    end

    return common_items_matrix
end

"""
    display_common_items(results::DataFrame)

Display the matrix of common items between versions.
"""
function display_common_items(results::DataFrame)
    common_items_matrix = calculate_common_items(results)
    print_title_and_separator(COMMON_ITEMS_MATRIX_TITLE)
    display(common_items_matrix)
    println(SEPARATOR)
    return common_items_matrix
end

function display_final_results(parameters::Params, results::DataFrame)
    items_used = length(unique(reduce(vcat, eachcol(results))))
    println(VERSIONS_ASSEMBLED_MESSAGE, size(results, 2))
    println(ITEMS_USED_MESSAGE, items_used)
    println(REMAINING_ITEMS_MESSAGE, length(parameters.bank.CLAVE) - items_used)
    return display_common_items(results)
end

"""
    display_results(model::Model)

Display the results of the optimization, including the tolerance and objective value.
"""
function display_results(model::Model, parameters::Params)
    print_title_and_separator(OPTIMIZATION_RESULTS_TITLE)
    return print_optimization_results(model, parameters)
end

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

Remove items used in versions from the bank and update probabilities or
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
    process_and_store_results!(model::Model, parameters::Params, results::DataFrame)

Process the optimization results for each form and store them in a DataFrame.
Used items are removed from the bank for subsequent forms.
"""
function process_and_store_results!(model::Model, parameters::Params, results::DataFrame)
    solver_matrix = value.(model[:x])
    item_codes = parameters.bank.CLAVE
    items = 1:length(item_codes)
    used_items = Int[]
    max_items = parameters.max_items

    for version_name in 1:(parameters.num_forms)
        selected_items = items[solver_matrix[:, version_name] .> 0.5]
        item_codes_in_version = item_codes[selected_items]
        version_length = length(item_codes_in_version)
        pad = max_items - version_length
        padded_item_codes = vcat(item_codes_in_version, fill(MISSING_VALUE_FILLER, pad))

        results[!, generate_unique_column_name(results)] = padded_item_codes
        used_items = vcat(used_items, selected_items)
    end
    used_items = sort(unique(used_items))
    remove_used_items!(parameters, used_items)
    return results
end

"""
    generate_unique_column_name(results::DataFrame)

Generate a unique column name to avoid naming conflicts in the results DataFrame.
"""
function generate_unique_column_name(results::DataFrame)
    i = 1
    while "Version_$i" in names(results)
        i += 1
    end
    return "Version_$i"
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
    save_forms(parameters::Params, results::DataFrame, config::Config)

Save the forms to a file, marking used items with a checkmark.
"""
function save_forms(parameters::Params, results::DataFrame, config)
    bank = deepcopy(parameters.bank)

    for v in names(results)
        bank[!, Symbol(v)] = map(x -> x == 1 ? CHECKMARK : "",
                                 bank.CLAVE .∈ Ref(skipmissing(results[:, v])))
    end
    write_results_to_file(bank, config.versions_file)
    return write_results_to_file(results, config.results_file)
end

"""
    main(config_file::String=CONFIG_FILE)

Main function to run the entire optimization process, including loading configurations,
running the solver, processing results, and saving the output.
"""
function main(config_file::String = CONFIG_FILE)
    config, original_parameters = load_configuration(config_file)
    parameters = deepcopy(original_parameters)
    constraints = read_constraints(config.constraints_file)
    parameters.num_forms = parameters.shadow_test_size > 0 ? 1 : parameters.f
    results = DataFrame()

    while parameters.f > 0
        parameters.num_forms = min(parameters.num_forms, parameters.f)
        parameters.shadow_test_size = max(0, parameters.f - parameters.num_forms)
        handle_anchor_items(parameters, original_parameters)

        model = Model()
        configure_solver!(model, parameters, config.solver)
        initialize_model!(model, parameters, original_parameters, constraints)

        if run_optimization(model)
            results = process_and_store_results!(model, parameters, results)
            display_results(model, parameters)
            parameters.f -= parameters.num_forms
        else
            println(OPTIMIZATION_FAILED_MESSAGE)
            parameters.f = 0
        end
    end

    # Display common items matrix
    if parameters.verbose > 0
        # display_common_items(results)
        display_final_results(original_parameters, results)
    end

    # Generate plots
    plot_characteristic_curves_and_simulation(original_parameters, results)

    return save_forms(original_parameters, results, config)
end

# Uncomment to run the main function
# main(CONFIG_FILE)
