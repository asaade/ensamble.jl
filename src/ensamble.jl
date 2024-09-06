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
    par = get_params(config)
    return config, par
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
    remove_used_items!(par::Parameters, items_used)

Remove items used in forms from the bank and update probabilities or
information for the remaining items based on the method used.
"""
function remove_used_items!(par::Parameters, items_used)
    remaining = setdiff(1:length(par.bank.CLAVE), items_used)
    par.bank = par.bank[remaining, :]
    if par.method in ["TCC", "TIC", "TIC2", "TIC3"]
        par.method in ["TCC"] && (par.p = par.p[remaining, :])
        par.method in ["TIC", "TIC2", "TIC3"] &&
            (par.info = par.info[remaining, :])
    end
    return par
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
    process_and_store_results!(model::Model, par::Parameters, results_df::DataFrame)

Process the optimization results_df for each form and store them in a DataFrame.
Used items are removed from the bank for subsequent forms.
"""
function process_and_store_results!(model::Model, par::Parameters, results_df::DataFrame)
    solver_matrix = value.(model[:x])
    item_codes = par.bank.CLAVE
    items = 1:length(item_codes)
    items_used = Int[]
    max_items = par.max_items
    bank = par.bank

    for f in 1:(par.num_forms)
        selected_items = items[solver_matrix[:, f] .> 0.9]
        codes_in_form = item_codes[selected_items]
        form_length = length(codes_in_form)
        missing_rows = max_items - form_length
        padded_codes_vector = vcat(codes_in_form,
                                   fill(MISSING_VALUE_FILLER, missing_rows))
        results_df[!, generate_unique_column_name(results_df)] = padded_codes_vector
        items_used = vcat(items_used, selected_items)
    end

    items_used = sort(unique(items_used))
    bank[items_used, :ITEM_USE] .+= 1
    items_used = items_used[bank[items_used, :ITEM_USE] .>= par.max_item_use]
    remove_used_items!(par, items_used)
    return results_df
end

"""
    handle_anchor_items(par::Parameters, old_par::Parameters)

Handle anchor items by adjusting the bank and relevant par based on the
specified anchor number.
"""
function handle_anchor_items(par::Parameters, old_par::Parameters)
    if par.anchor_tests > 0
        par.anchor_tests = par.anchor_tests % old_par.anchor_tests + 1
        bank = par.bank[par.bank.ANCHOR .== 0, :]
        anchors = old_par.bank[old_par.bank.ANCHOR .== par.anchor_tests,
                               :]
        par.bank = vcat(bank, anchors)

        if par.method in ["TCC"]
            par.p = old_par.p[par.bank.INDEX, :]
        elseif par.method in ["TIC", "TIC2", "TIC3"]
            par.info = old_par.info[par.bank.INDEX, :]
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
    par = deepcopy(old_par)
    constraints = read_constraints(config.constraints_file)
    results_df = DataFrame()

    while par.f > 0
        par.num_forms = min(par.num_forms, par.f)
        par.shadow_test = max(0, par.f - par.num_forms)
        handle_anchor_items(par, old_par)

        model = Model()
        configure_solver!(model, par, config.solver)
        initialize_model!(model, par, constraints)

        if run_optimization(model)
            results_df = process_and_store_results!(model, par, results_df)
            display_results(model, par)
            par.f -= par.num_forms
        else
            println(OPTIMIZATION_FAILED_MESSAGE)
            par.f -= 1
        end
    end

    ## Display results_df and save
    return final_report(old_par, results_df, config)
end

# Uncomment to run the main function
# main(CONFIG_FILE)
