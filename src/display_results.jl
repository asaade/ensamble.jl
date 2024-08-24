using DataFrames, JuMP

include("charts.jl")
include("constants.jl")
include("types.jl")

# Print functions
function print_title_and_separator(title::String)
    println(title)
    return println(SEPARATOR)
end

function print_optimization_results(model::Model, parameters::Params)
    parameters.verbose > 1 && println(solution_summary(model))
    println(TOLERANCE_LABEL, round(objective_value(model); digits=4))
    return println(SEPARATOR)
end

"""
    calculate_common_items(results::DataFrame)

Calculate the number of common items between forms and return a matrix
where each entry [i, j] indicates the number of common items between form `i` and form `j`.
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

Display the matrix of common items between forms.
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
    println(FORMS_ASSEMBLED_MESSAGE, size(results, 2))
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
    save_forms(parameters::Params, results::DataFrame, config::Config)

Save the forms to a file, marking used items with a checkmark.
"""
function save_forms(parameters::Params, results::DataFrame, config)
    bank = deepcopy(parameters.bank)

    for v in names(results)
        bank[!, Symbol(v)] = map(x -> x == 1 ? CHECKMARK : "",
                                 bank.CLAVE .∈ Ref(skipmissing(results[:, v])))
    end
    write_results_to_file(bank, config.forms_file)
    return write_results_to_file(results, config.results_file)
end

function final_report(original_parameters::Params, results::DataFrame, config::Config)
    # Display common items matrix
    # display_common_items(results)
    display_final_results(original_parameters, results)

    # Generate plots
    plot_results_and_simulation(original_parameters, results)

    return save_forms(original_parameters, results, config)
end
