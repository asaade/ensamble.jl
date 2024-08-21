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


function final_report(original_parameters::Params, results::DataFrame, config::Config)
    # Display common items matrix
    if original_parameters.verbose > 0
        # display_common_items(results)
        display_final_results(original_parameters, results)
    end

    # Generate plots
    plot_characteristic_curves_and_simulation(original_parameters, results)

    return save_forms(original_parameters, results, config)
end
