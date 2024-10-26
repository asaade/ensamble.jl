module DisplayResults

export final_report, generate_report, show_results, simulate_scores

using JuMP, DataFrames, Dates, PrettyTables, DataFrames
using ..Configuration, ..Utils
include("charts.jl")
using .Charts

# Constants for labels used in reporting
const LABEL_FORMS = "Total forms:"
const LABEL_ITEMS = "Total items (with anchor):"
const LABEL_NON_ANCHOR = "Non-anchor items:"
const LABEL_ANCHOR = "Anchor items:"
const LABEL_UNUSED = "Unused items (non-anchor):"

"""
    title_with_separator(title)

Prints a title with a separator.
"""
function title_with_separator(title::String)
    println("\n" * title)
    return println(repeat("=", length(title)))
end

"""
    show_opt_results(model, parms)

Displays the optimization results, including objective value.
"""
function show_opt_results(model::Model, parms::Parameters)
    parms.verbose > 1 && println(solution_summary(model))
    println("Objective: ", round(objective_value(model); digits = 4))
    return println("=====================================")
end

"""
    check_empty!(df)

Throws an error if the DataFrame is empty.
"""
check_empty!(df::DataFrame) = isempty(df) && throw(ArgumentError("Results are empty."))

"""
    check_column!(col_name, df)

Checks if a column exists in the DataFrame.
"""
function check_column!(col_name::String, df::DataFrame)
    return col_name in names(df) ||
           throw(ArgumentError("Column '$col_name' does not exist."))
end

"""
    show_results(model, parms)

Displays optimization results, including tolerance and objective value.
"""
function show_results(model::Model, parms::Parameters)
    title_with_separator("Optimization Results")
    return show_opt_results(model, parms)
end

"""
    anchor_summary(results, bank) -> String

Returns a table summarizing item distribution (total, anchor, non-anchor) for each form.
"""
function anchor_summary(results::DataFrame, bank::DataFrame)::String
    check_empty!(results)
    check_column!("ANCHOR", bank)

    num_forms = size(results, 2)
    header = ["Form", "Total", "Anchor", "Non-anchor"]
    table_data = []

    for i in 1:num_forms
        selected_items = collect(skipmissing(results[:, i]))
        total_items = length(selected_items)
        anchor_items = sum(in(selected_items).(bank[bank.ANCHOR .!== missing, :ID]))
        non_anchor_items = total_items - anchor_items
        push!(table_data, [i, total_items, anchor_items, non_anchor_items])
    end

    table_matrix = hcat(table_data...)'
    return pretty_table(String, table_matrix; header = header, alignment = :r)
end

"""
    col_sums(results, bank, cols) -> String

Returns a table showing the sum of values from specified columns for each form.
"""
function col_sums(results::DataFrame, bank::DataFrame, cols)::String
    check_empty!(results)

    num_forms = size(results, 2)
    item_ids = bank[:, :ID]
    cols = String.(cols isa AbstractString ? [cols] : cols)

    for col in cols
        check_column!(col, bank)
    end

    form_ids = collect(1:num_forms)
    table_data = [form_ids]

    for col in cols
        col_sum = [sum(bank[findall(in(skipmissing(results[:, i])), item_ids), col])
                   for
                   i in 1:num_forms]
        push!(table_data, col_sum)
    end

    return pretty_table(
        String, hcat(table_data...); header = ["Form"; cols...], alignment = :r)
end

"""
    cat_counts(results, bank, col; max_cats=10) -> String

Returns a table showing the counts of items grouped by category in the specified column for each form.
"""
function cat_counts(
        results::DataFrame, bank::DataFrame, col::Union{String, Symbol}; max_cats::Int = 10
)::String
    check_empty!(results)
    check_column!(String(col), bank)

    num_forms = size(results, 2)
    categories = sort(unique(collect(skipmissing(bank[!, Symbol(col)]))))
    non_missing_bank = filter(row -> !ismissing(row[Symbol(col)]), bank)

    table_data = []
    for i in 1:num_forms
        selected_items = skipmissing(results[:, i])
        form_counts = [sum(
                           non_missing_bank[
                           in(selected_items).(non_missing_bank.ID), Symbol(col)] .==
                           category,
                       ) for category in categories]
        push!(table_data, [i; form_counts...])
    end

    if length(categories) <= max_cats
        header = ["Form"; categories...]
        table_matrix = hcat(table_data...)
        pretty_table(String, table_matrix'; header = header, alignment = :r)
    else
        header = ["Category"; [string("Form ", i) for i in 1:num_forms]...]
        table_matrix = hcat([categories]..., [row[2:end] for row in table_data]...)
        valid_rows = [any(row[2:end] .!= 0) for row in eachrow(table_matrix)]
        pretty_table(String, table_matrix[valid_rows, :]; header = header, alignment = :r)
    end
end

"""
    common_items(results) -> String

Returns a matrix showing common items between each pair of forms.
"""
function common_items(results::DataFrame)::String
    check_empty!(results)

    num_forms = size(results, 2)
    common_matrix = zeros(Int, num_forms, num_forms)

    for i in 1:num_forms, j in 1:num_forms
        common = in(skipmissing(results[:, i])).(skipmissing(results[:, j]))
        common_matrix[i, j] = sum(common)
    end

    header = [""; collect(1:num_forms)...]
    return pretty_table(
        String, hcat(collect(1:num_forms), common_matrix); header = header, alignment = :r
    )
end

"""
    final_summary(parms, results) -> String

Returns a table summarizing the final assembly, including total forms, items used, and remaining items.
"""
function final_summary(parms::Parameters, results::DataFrame)::String
    bank = parms.bank
    items = bank.ID
    anchor_items = bank[bank.ANCHOR .!== missing, :ID]
    used_items = unique(skipmissing(reduce(vcat, eachcol(results))))
    used_anchors = anchor_items[anchor_items .∈ Ref(used_items)]
    used_non_anchors = setdiff(used_items, used_anchors)

    labels = [LABEL_FORMS, LABEL_ITEMS, LABEL_NON_ANCHOR, LABEL_ANCHOR, LABEL_UNUSED]
    values = [
        size(results, 2),
        length(used_items),
        length(used_non_anchors),
        length(used_anchors),
        length(items) - length(used_items)
    ]

    return pretty_table(
        String,
        hcat(labels, string.(values));
        header = ["Concept", "Count"],
        alignment = [:l, :r]
    )
end

"""
    tolerances_table(tols) -> String

Returns a table of tolerances for each form.
"""
function tolerances_table(tols::Vector{Float64})::String
    header = ["Form", "Tolerance"]
    form_ids = [i for i in eachindex(tols)]
    table_data = Dict(k => rpad(v, 6, "0") for (k, v) in zip(form_ids, tols))

    return pretty_table(
        String, table_data; header = header, sortkeys = true, alignment = :c)
end

"""
    gather_tables(parms, config, results, tols) -> Dict{String, String}

Gathers summary tables into a dictionary for the final report.
"""
function gather_tables(
        parms::Parameters, config::Config, results::DataFrame, tols::Vector{Float64}
)::Dict{String, String}
    tables = Dict{String, String}()

    tables["Summary"] = final_summary(parms, results)
    tables["Anchor"] = anchor_summary(results, parms.bank)

    if !isempty(config.report_categories)
        for cat in config.report_categories
            tables["$cat counts"] = cat_counts(results, parms.bank, cat)
        end
    end

    if !isempty(config.report_sums)
        tables["Column sums"] = col_sums(results, parms.bank, config.report_sums)
    end

    tables["Common items"] = common_items(results)
    tables["Tolerances"] = tolerances_table(tols)

    return tables
end

"""
    save_forms(parms, results, config)

Saves the forms and results to files.
"""
function save_forms(parms::Parameters, results::DataFrame, config::Config)
    bank = deepcopy(parms.bank)

    for v in names(results)
        bank[!, Symbol(v)] = map(
            x -> ismissing(x) ? "" : (x == 1 ? v : ""),
            bank.ID .∈ Ref(skipmissing(results[:, v]))
        )
    end

    save_to_csv(bank, config.forms_file)
    save_to_csv(results, config.results_file)

    println("\nSaved Forms and Results")
    println("=======================")
    println("Modified bank saved to: ", config.forms_file)
    return println("Results saved to: ", config.results_file)
end

"""
    report(parms, results, config, tols) -> Dict{String, String}

Generates the final report and returns it as a dictionary.
"""
function final_report(
        parms::Parameters, results::DataFrame, config::Config, tols::Vector{Float64}
)::Dict{String, String}
    report_data = gather_tables(parms, config, results, tols)
    save_forms(parms, results, config)
    plot_results(parms, config, results)
    return report_data
end

"""
    generate_report(report_data) -> String

Generates a formatted report with sections for summary, anchor usage,
counts and sum for selected columns, and form optimization tolerances.
"""
function generate_report(report_data::Dict{String, String})::String
    report = "Report on Test Assembly Results\n"
    report *= "Generated by Ensamble.jl\n"
    report *= "Date: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))\n\n"

    report *= "Summary\n"
    report *= report_data["Summary"] * "\n\n"

    report *= "Tolerances\n"
    report *= report_data["Tolerances"] * "\n\n"

    report *= "Common Items\n"
    report *= report_data["Common items"] * "\n\n"

    report *= "Anchor Items\n"
    report *= report_data["Anchor"] * "\n\n"

    for cat_key in filter(key -> endswith(key, "counts"), keys(report_data))
        report *= "Category Counts for $(replace(cat_key, " counts" => ""))\n"
        report *= report_data[cat_key] * "\n\n"
    end

    if haskey(report_data, "Column sums")
        report *= "Column Sums\n"
        report *= report_data["Column sums"] * "\n\n"
    end

    return report
end

end
