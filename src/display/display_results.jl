using JuMP
using DataFrames
using PrettyTables
using .Ensamble.Configuration

"""
    print_title_and_separator(title::String)

Prints a title followed by a separator line for formatting.
"""
function print_title_and_separator(title::String)
    println("\n" * title)
    println(repeat("=", length(title)))
    return nothing
end

"""
    print_optimization_results(model::Model, parms::Parameters)

Prints the optimization results including the objective value and a tolerance summary.
"""
function print_optimization_results(model::Model, parms::Parameters)
    parms.verbose > 1 && println(solution_summary(model))
    println("Value: ", round(objective_value(model); digits=4))
    println("=====================================")
    return nothing
end

"""
    check_empty_results!(results::DataFrame)

Throws an error if the `results` DataFrame is empty.
"""
function check_empty_results!(results::DataFrame)
    isempty(results) && throw(ArgumentError("Results DataFrame is empty."))
end

"""
    check_column_exists!(column_name::String, bank::DataFrame)

Checks if a column exists in the `bank` DataFrame, throws an error if not.
"""
function check_column_exists!(column_name::String, bank::DataFrame)
    if !(column_name in names(bank)) || isempty(column_name)
        throw(ArgumentError("The column '$column_name' does not exist in the bank DataFrame or is invalid."))
    end
end

"""
    display_results(model::Model, parms::Parameters)

Display the optimization results for the current cycle, including tolerance and objective value.
"""
function display_results(model::Model, parms::Parameters)
    print_title_and_separator("Optimization Results:")
    print_optimization_results(model, parms)
    return nothing
end

"""
    collect_anchors(results::DataFrame, bank::DataFrame) -> String

Returns the item distribution summary (total items, anchor, non-anchor) as a table string.
"""
function collect_anchors(results::DataFrame, bank::DataFrame)::String
    check_empty_results!(results)
    check_column_exists!("ANCHOR", bank)

    num_forms = size(results, 2)
    header = ["Form ID", "Total Items", "Anchor Items", "Non-anchor Items"]
    table_data = []

    for i in 1:num_forms
        selected_items = collect(skipmissing(results[:, i]))
        total_items = length(selected_items)
        anchor_items = sum(in(selected_items).(bank[bank.ANCHOR .> 0, :ID]))
        non_anchor_items = total_items - anchor_items
        push!(table_data, [i, total_items, anchor_items, non_anchor_items])
    end

    return pretty_table(String, hcat(table_data...); header=header, alignment=:r)
end

"""
    collect_column_sums(results::DataFrame, bank::DataFrame, column_names) -> String

Returns the sum of values from one or more specified columns in the bank DataFrame for each form.
"""
function collect_column_sums(results::DataFrame, bank::DataFrame, column_names)::String
    check_empty_results!(results)

    num_forms = size(results, 2)
    item_ids = bank[:, :ID]
    column_names = typeof(column_names) in [String, Symbol] ? [column_names] : column_names
    column_names = String.(column_names)

    for column_name in column_names
        check_column_exists!(column_name, bank)
    end

    form_ids = collect(1:num_forms)
    table_data = [form_ids]

    for col in column_names
        column_sums = [sum(bank[findall(in(skipmissing(results[:, i])), item_ids), col]) for i in 1:num_forms]
        push!(table_data, column_sums)
    end

    return pretty_table(String, hcat(table_data...); header=["Form ID"; column_names...], alignment=:r)
end

"""
    collect_category_counts(results::DataFrame, bank::DataFrame, column_name::Union{String, Symbol}; max_categories::Int=10) -> String

Returns a table with counts of items in the specified column grouped by their value (category) for each form.
"""
function collect_category_counts(results::DataFrame, bank::DataFrame, column_name::Union{String, Symbol}; max_categories::Int=10)::String
    check_empty_results!(results)
    check_column_exists!(String(column_name), bank)

    num_forms = size(results, 2)
    categories = sort(unique(collect(skipmissing(bank[!, Symbol(column_name)]))))
    non_missing_bank = filter(row -> !ismissing(row[Symbol(column_name)]), bank)
    header = ["Category"; [string("Form ", i) for i in 1:num_forms]...]
    table_data = []

    for category in categories
        category_counts = []
        for i in 1:num_forms
            selected_items = skipmissing(results[:, i])
            count = sum(non_missing_bank[in(selected_items).(non_missing_bank.ID), Symbol(column_name)] .== category)
            push!(category_counts, count)
        end

        # Include row if there are counts or if categories are below max limit
        if sum(category_counts) > 0 || length(categories) <= max_categories
            push!(table_data, [category; category_counts...])
        end
    end

    return pretty_table(String, hcat(table_data...)'; header=header, alignment=:r)
end

"""
    collect_common_items(results::DataFrame) -> String

Returns a matrix showing the number of common items between each pair of forms.
"""
function collect_common_items(results::DataFrame)::String
    check_empty_results!(results)

    num_forms = size(results, 2)
    common_items_matrix = zeros(Int, num_forms, num_forms)

    for i in 1:num_forms, j in 1:num_forms
        common = in(skipmissing(results[:, i])).(skipmissing(results[:, j]))
        common_items_matrix[i, j] = sum(common)
    end

    header = [""; collect(1:num_forms)...]
    return pretty_table(String, hcat(collect(1:num_forms), common_items_matrix); header=header, alignment=:r)
end

"""
    collect_final_summary(parms::Parameters, results::DataFrame) -> String

Generates a summary of the final assembly, including total forms, items used, anchor items, and remaining items.
"""
function collect_final_summary(parms::Parameters, results::DataFrame)::String
    bank = parms.bank
    items = bank.ID
    anchor_items = bank[bank.ANCHOR .> 0, :ID]
    items_used = unique(skipmissing(reduce(vcat, eachcol(results))))
    anchors_used = anchor_items[anchor_items .∈ Ref(items_used)]
    non_anchor_used = setdiff(items_used, anchors_used)

    # Separate labels and values into two columns for PrettyTables
    labels = [
        LABEL_TOTAL_FORMS,
        LABEL_ITEMS_USED,
        LABEL_NON_ANCHOR_ITEMS,
        LABEL_ANCHOR_ITEMS,
        LABEL_ITEMS_NOT_USED
    ]

    values = [
        size(results, 2),
        length(items_used),
        length(non_anchor_used),
        length(anchors_used),
        length(items) - length(anchor_items) - length(items_used)
    ]

    header = ["Concept", "Count"]
    return pretty_table(String, hcat(labels, string.(values)); header=header, alignment=[:l, :r])
end

"""
    collect_tolerances(tolerances::Vector{Float64}) -> String

Collects the tolerances for each form and returns them as a table string.
"""
function collect_tolerances(tolerances::Vector{Float64})::String
    header = ["Form ID"; "Tolerance"]
    forms_id = [string("Form ", lpad(s, 2)) for s in 1:length(tolerances)]
    table_dict = Dict(k => rpad(v, 6, "0") for (k, v) in zip(forms_id, tolerances))

    return pretty_table(String, table_dict, sortkeys=true; header=header, alignment=:c)
end

"""
    collect_results_tables(parms::Parameters, config::Config, results::DataFrame, tolerances::Vector{Float64}) -> Dict{String, String}

Generates the final report of the test assembly process, gathering tables and summaries into a Dict for later processing.
"""
function collect_results_tables(parms::Parameters, config::Config, results::DataFrame, tolerances::Vector{Float64})::Dict{String, String}
    results_dict = Dict{String, String}()

    results_dict["Summary"] = collect_final_summary(parms, results)
    results_dict["ANCHOR use"] = collect_anchors(results, parms.bank)

    if !isempty(config.report_categories)
        for category in config.report_categories
            results_dict["$category count"] = collect_category_counts(results, parms.bank, category)
        end
    end

    if !isempty(config.report_sums)
        results_dict["Column sums"] = collect_column_sums(results, parms.bank, config.report_sums)
    end

    results_dict["Common items"] = collect_common_items(results)
    results_dict["Optimization tolerances"] = collect_tolerances(tolerances)

    return results_dict
end

"""
    save_forms(parms::Parameters, results::DataFrame, config::Config)

Saves the forms to a file, marking used items with a checkmark.
"""
function save_forms(parms::Parameters, results::DataFrame, config::Config)
    bank = deepcopy(parms.bank)

    for v in names(results)
        bank[!, Symbol(v)] = map(x -> ismissing(x) ? "" : (x == 1 ? v : ""), bank.ID .∈ Ref(skipmissing(results[:, v])))
    end

    write_results_to_file(bank, config.forms_file)
    write_results_to_file(results, config.results_file)

    println("\nSaved Forms and Results:")
    println("========================")
    println("Modified bank saved to: ", config.forms_file)
    println("Resulting forms saved to: ", config.results_file)
    return nothing
end

"""
    final_report(parms::Parameters, results::DataFrame, config::Config, tolerances::Vector{Float64}) -> Dict{String, String}

Generates the final report of the optimization process and returns it as a Dict for further use.
"""
function final_report(parms::Parameters, results::DataFrame, config::Config, tolerances::Vector{Float64})::Dict{String, String}
    report_tables = collect_results_tables(parms, config, results, tolerances)

    save_forms(parms, results, config)
    plot_results(parms, config, results)

    return report_tables
end
