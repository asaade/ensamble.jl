

using DataFrames
using JuMP

# Include external modules
include("get_data.jl")
include("solvers.jl")
include("stats_functions.jl")
include("constraint_processing.jl")
include("charts.jl")

const CONFIG_FILE = "data/config.yaml"

# Calculate common items between versions
function calculate_common_items(results::DataFrame)
    num_forms = size(results, 2)
    common_items_matrix = zeros(Int, num_forms, num_forms)

    for i in 1:num_forms, j in 1:num_forms
        common = in(skipmissing(results[:, i])).(skipmissing(results[:, j]))
        common_items_matrix[i, j] = sum(common)
    end

    return common_items_matrix
end

# Display common items matrix
function display_common_items(results::DataFrame)
    common_items_matrix = calculate_common_items(results)
    println("\nCommon Items Matrix:")
    display(common_items_matrix)
    println("=====================================")
    return common_items_matrix
end

# Display optimization results
function display_results(model::Model)
    println("\nOptimization Results:")
    # println(solution_summary(model))
    println("Tolerance: ", round(objective_value(model); digits=4))
    println("=====================================\n")
end


# Load configuration and parameters from file
function load_configuration(config_file::String)
    println("Loading configuration...")
    config = load_config(config_file)
    parameters = get_params(config)
    return config, parameters
end

# Run the optimization solver
function run_optimization(model::Model)
    println("Running optimization...")
    optimize!(model)
    return is_solved_and_feasible(model)
end


"""
    remove_used_items!(parameters::Params, used_items)

Remove items used in versions from bank and update probability and
information in remaing items from parameters
"""
function remove_used_items!(parameters::Params, used_items)
    remaining = setdiff(1:length(parameters.bank.CLAVE), used_items)
    parameters.bank = parameters.bank[remaining, :]
    if parameters.method in ["TCC", "ICC", "ICC2"]
        parameters.method == "TCC" && (parameters.p = parameters.p[remaining, :])
        parameters.method in ["ICC", "ICC2"] && (parameters.info = parameters.info[remaining, :])
    end
    return parameters
end


"""
        process_and_store_results!(model::Model, parameters::Params, results::DataFrame)

Process results after each optimization iteration and store them in a single
matrix. Used items are excluded from future versions.
"""
function process_and_store_results!(model::Model, parameters::Params, results::DataFrame)
    solver_matrix = value.(model[:x])
    item_codes = parameters.bank.CLAVE
    items = 1:length(item_codes)
    used_items = Int[]

    for version_name in 1:parameters.num_forms
        selected_items = items[solver_matrix[:, version_name] .> 0]
        item_codes_in_version = item_codes[selected_items]

        padded_item_codes = vcat(item_codes_in_version, fill(missing, parameters.max_items - length(item_codes_in_version)))
        results[!, generate_unique_column_name(results)] = padded_item_codes
        used_items = vcat(used_items, selected_items)
    end

    used_items = sort(unique(used_items))
    remove_used_items!(parameters, used_items)
    return results
end

# Generate a unique column name to avoid clashes
function generate_unique_column_name(results::DataFrame)
    i = 1
    while "Version_$i" in names(results)
        i += 1
    end
    return "Version_$i"
end


function handle_anchor_items(parameters::Params, original_parameters::Params)
    if parameters.anchor_number > 0
        parameters.anchor_number = mod(parameters.anchor_number, original_parameters.anchor_number) + 1
        bank = parameters.bank[parameters.bank.ANCHOR.==0, :]
        anchors = original_parameters.bank[original_parameters.bank.ANCHOR.==parameters.anchor_number, :]
        parameters.bank = vcat(bank, anchors)

        if parameters.method in ["TCC", "ICC", "ICC2"]
            if parameters.method == "TCC"
                parameters.p = original_parameters.p[parameters.bank.INDEX, :]
            elseif parameters.method in ["ICC", "ICC2"]
                parameters.info = original_parameters.info[parameters.bank.INDEX, :]
            end
        end
    end
end


function save_forms(parameters::Params, results::DataFrame, file_name)
    bank = deepcopy(parameters.bank)

    for v in names(results)
        bank[!, Symbol(v)] = map(x -> x == 1 ? " ✓" : "", bank.CLAVE .∈ Ref(skipmissing(results[:, v])))
    end
    write_results_to_file(bank, file_name)

    # bank = deepcopy(parameters.bank)
    # dfv = view(bank, bank.CLAVE .∈ Ref(skipmissing(results[:, v])), :)
    # @. dfv.VERSION = v
end

# Main function to run the entire process
function main(config_file::String=CONFIG_FILE)
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
        configure_solver!(model, config.solver)
        initialize_model!(model, parameters, constraints)

        if run_optimization(model)
            results = process_and_store_results!(model, parameters, results)
            display_results(model)
            parameters.f -= parameters.num_forms
        else
            println("Optimization failed")
            parameters.f = 0
        end
    end

    # Display common items matrix
    display_common_items(results)

    # Generate plots
    plot_characteristic_curves_and_simulation(original_parameters, results)

    save_forms(original_parameters, results, "data/versiones.csv")
end

# main(CONFIG_FILE)
