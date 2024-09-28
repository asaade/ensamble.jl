module DisplayResults

export final_report, generate_report, display_results

using JuMP
using DataFrames
using Dates
using PrettyTables

using ..Configuration
using ..Utils

include("charts.jl")
using .Charts

# Constants for labels used across the reporting functions
const LABEL_TOTAL_FORMS = "Total forms assembled:"
const LABEL_ITEMS_USED = "Total items used (includes anchor):"
const LABEL_NON_ANCHOR_ITEMS = "Non-anchor items used:"
const LABEL_ANCHOR_ITEMS = "Anchor items used:"
const LABEL_ITEMS_NOT_USED = "Items not used (without anchor):"

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
    return isempty(results) && throw(ArgumentError("Results DataFrame is empty."))
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

Returns the item distribution summary (total items, anchor, non-anchor) as a transposed table string.
Forms will appear in the rows, and item types will be in the columns.
"""
function collect_anchors(results::DataFrame, bank::DataFrame)::String
    check_empty_results!(results)
    check_column_exists!("ANCHOR", bank)

    num_forms = size(results, 2)
    header = ["Form ID", "Total", "Anchor", "Non-anchor"]
    table_data = []

    for i in 1:num_forms
        selected_items = collect(skipmissing(results[:, i]))
        total_items = length(selected_items)
        anchor_items = sum(in(selected_items).(bank[bank.ANCHOR .> 0, :ID]))
        non_anchor_items = total_items - anchor_items
        push!(table_data, [i, total_items, anchor_items, non_anchor_items])
    end

    # Convert the table data to a matrix and transpose it
    table_matrix = hcat(table_data...)'

    # Return the PrettyTable formatted string with the transposed matrix
    return pretty_table(String, table_matrix; header=header, alignment=:r)
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
        column_sums = [sum(bank[findall(in(skipmissing(results[:, i])), item_ids), col])
                       for i in 1:num_forms]
        push!(table_data, column_sums)
    end

    return pretty_table(String, hcat(table_data...); header=["Form ID"; column_names...],
                        alignment=:r)
end

"""
    collect_category_counts(results::DataFrame, bank::DataFrame, column_name::Union{String, Symbol}; max_categories::Int=10) -> String

Generates a table with counts of items in the specified column grouped by their value (category) for each form.
If the number of categories exceeds `max_categories`, the table is transposed with forms as columns.
"""
function collect_category_counts(results::DataFrame, bank::DataFrame,
                                 column_name::Union{String, Symbol};
                                 max_categories::Int=10)::String
    check_empty_results!(results)
    check_column_exists!(String(column_name), bank)

    num_forms = size(results, 2)
    categories = sort(unique(collect(skipmissing(bank[!, Symbol(column_name)]))))
    non_missing_bank = filter(row -> !ismissing(row[Symbol(column_name)]), bank)

    # Prepare the table data with form IDs as rows and category counts as columns
    table_data = []
    for i in 1:num_forms
        selected_items = skipmissing(results[:, i])
        form_counts = [sum(non_missing_bank[in(selected_items).(non_missing_bank.ID),
                                            Symbol(column_name)] .== category)
                       for category in categories]
        push!(table_data, [i; form_counts...])
    end

    if length(categories) <= max_categories
        # Case 1: Number of categories is manageable (categories as columns)
        header = ["Form ID"; categories...]
        table_matrix = hcat(table_data...)
        return pretty_table(String, table_matrix'; header=header, alignment=:r)

    else
        # Case 2: Transpose the table when the number of categories is large (forms as columns)
        header = ["Category"; [string("Form ", i) for i in 1:num_forms]...]

        # Construct the transposed table
        # We need to transpose both the categories and the counts
        table_matrix = hcat([categories]..., [row[2:end] for row in table_data]...)

        return pretty_table(String, table_matrix; header=header, alignment=:r)
    end
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
    return pretty_table(String, hcat(collect(1:num_forms), common_items_matrix);
                        header=header, alignment=:r)
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
    labels = [LABEL_TOTAL_FORMS,
              LABEL_ITEMS_USED,
              LABEL_NON_ANCHOR_ITEMS,
              LABEL_ANCHOR_ITEMS,
              LABEL_ITEMS_NOT_USED]

    values = [size(results, 2),
              length(items_used),
              length(non_anchor_used),
              length(anchors_used),
              length(items) - length(anchor_items) - length(items_used)]

    header = ["Concept", "Count"]
    return pretty_table(String, hcat(labels, string.(values)); header=header,
                        alignment=[:l, :r])
end

"""
    collect_tolerances(tolerances::Vector{Float64}) -> String

Collects the tolerances for each form and returns them as a table string.
"""
function collect_tolerances(tolerances::Vector{Float64})::String
    header = ["Form ID"; "Tolerance"]
    forms_id = [string("Form ", lpad(s, 2)) for s in 1:length(tolerances)]
    table_dict = Dict(k => rpad(v, 6, "0") for (k, v) in zip(forms_id, tolerances))

    return pretty_table(String, table_dict; sortkeys=true, header=header, alignment=:c)
end

"""
    collect_results_tables(parms::Parameters, config::Config, results::DataFrame, tolerances::Vector{Float64}) -> Dict{String, String}

Generates the final report of the test assembly process, gathering tables and summaries into a Dict for later processing.
"""
function collect_results_tables(parms::Parameters, config::Config, results::DataFrame,
                                tolerances::Vector{Float64})::Dict{String, String}
    results_dict = Dict{String, String}()

    results_dict["Summary"] = collect_final_summary(parms, results)
    results_dict["ANCHOR use"] = collect_anchors(results, parms.bank)

    if !isempty(config.report_categories)
        for category in config.report_categories
            results_dict["$category count"] = collect_category_counts(results, parms.bank,
                                                                      category)
        end
    end

    if !isempty(config.report_sums)
        results_dict["Column sums"] = collect_column_sums(results, parms.bank,
                                                          config.report_sums)
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
        bank[!, Symbol(v)] = map(x -> ismissing(x) ? "" : (x == 1 ? v : ""),
                                 bank.ID .∈ Ref(skipmissing(results[:, v])))
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
function final_report(parms::Parameters, results::DataFrame, config::Config,
                      tolerances::Vector{Float64})::Dict{String, String}
    report_tables = collect_results_tables(parms, config, results, tolerances)

    save_forms(parms, results, config)
    plot_results(parms, config, results)

    return report_tables
end

"""
    generate_report(report_data::Dict{String, String}) -> String

Generates a formatted report using the results from final_report, with sections for summary,
anchor usage, category counts, column sums, common items, and tolerances.
"""
function generate_report(report_data::Dict{String, String})::String
    report = ""

    # Title Section
    report *= "Report on Automatic Test Assembly Results\n"
    report *= "Generated using Ensamble.jl\n"
    report *= "Date: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))\n\n"

    # Add Summary
    report *= "Summary of Final Results\n"
    report *= report_data["Summary"]
    report *= "\n\n"

    # Add Optimization Tolerances
    report *= "Optimization Tolerances\n"
    report *= report_data["Optimization tolerances"]
    report *= "\n\n"

    # Add Common Items Matrix
    report *= "Common Items Matrix\n"
    report *= report_data["Common items"]
    report *= "\n\n"

    # Add Anchor Usage
    report *= "Anchor Item Usage\n"
    report *= report_data["ANCHOR use"]
    report *= "\n\n"

    # Add Category Counts (handle multiple categories dynamically)
    category_keys = filter(key -> endswith(key, "count"), keys(report_data))
    for category_key in category_keys
        report *= "Category Counts for $(replace(category_key, " count" => ""))\n"
        report *= report_data[category_key]
        report *= "\n\n"
    end

    # Add Column Sums (if present)
    if haskey(report_data, "Column sums")
        report *= "Column Sums\n"
        report *= report_data["Column sums"]
        report *= "\n\n"
    end

    return report
end

end
