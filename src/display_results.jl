using DataFrames, JuMP

include("charts.jl")
include("constants.jl")
include("types.jl")

# Print functions
function print_title_and_separator(title::String)
    println(title)
    return println(SEPARATOR)
end

function print_optimization_results(model::Model, parms::Parameters)
    parms.verbose > 1 && println(solution_summary(model))
    println(TOLERANCE_LABEL, round(objective_value(model); digits = 4))
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
        if i <= j
        end
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

function display_final_results(parms::Parameters, results::DataFrame)
    bank = parms.bank
    items = bank.CLAVE
    anchor_items = bank[bank.ANCHOR .> 0, :CLAVE]
    items_used = unique(reduce(vcat, eachcol(results)))
    anchors_used = anchor_items[anchor_items .∈ Ref(items_used)]
    non_anchor_used = setdiff(items_used, anchors_used)
    println(FORMS_ASSEMBLED_MESSAGE, size(results, 2))
    println(ITEMS_USED_MESSAGE, length(items_used))
    println(NONANCHOR_USED_MESSAGE, length(non_anchor_used))
    println(ANCHOR_USED_MESSAGE, length(anchors_used))
    println(REMAINING_ITEMS_MESSAGE,
            length(items) - length(anchor_items) - length(items_used))
    return display_common_items(results)
end

"""
    display_results(model::Model)

Display the results of the optimization, including the tolerance and objective value.
"""
function display_results(model::Model, parms::Parameters)
    print_title_and_separator(OPTIMIZATION_RESULTS_TITLE)
    return print_optimization_results(model, parms)
end

"""
    save_forms(parms::Parameters, results::DataFrame, config::Config)

Save the forms to a file, marking used items with a checkmark.
"""
function save_forms(parms::Parameters, results::DataFrame, config)
    bank = deepcopy(parms.bank)

    for v in names(results)
        bank[!, Symbol(v)] = map(x -> x == 1 ? CHECKMARK : "",
                                 bank.CLAVE .∈ Ref(skipmissing(results[:, v])))
    end
    write_results_to_file(bank, config.forms_file)
    return write_results_to_file(results, config.results_file)
end

function final_report(original_parms::Parameters, results::DataFrame, config::Config)
    # Display common items matrix
    # display_common_items(results)
    display_final_results(original_parms, results)

    # Generate plots
    plot_results_and_simulation(original_parms, results)

    return save_forms(original_parms, results, config)
end
