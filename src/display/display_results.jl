using JuMP
using DataFrames
using PrettyTables
using .Ensamble.Configuration


function print_optimization_results(model::Model, parms::Parameters)
    parms.verbose > 1 && println(solution_summary(model))
    println(TOLERANCE_LABEL, round(objective_value(model); digits=4))
    println(SEPARATOR)
    return nothing
end

"""
    print_title_and_separator(title::String)

Prints a title and a separator line beneath it for formatting purposes.
"""
function print_title_and_separator(title::String)
    println("\n" * title)
    println(repeat("=", length(title)))
    return nothing
end

"""
    display_results(model::Model)

Display the results of each optimization cycle, including the tolerance and objective value.
"""
function display_results(model::Model, parms::Parameters)
    print_title_and_separator(OPTIMIZATION_RESULTS_TITLE)
    print_optimization_results(model, parms)
    return nothing
end

"""
    display_item_distribution(results::DataFrame, bank::DataFrame)

Displays a summary of the item distribution across forms, including total items, anchor items,
and non-anchor items for each form.
Checks that the `results` DataFrame is not empty and validates the `bank` column `ANCHOR`.
"""
function display_item_distribution(results::DataFrame, bank::DataFrame)
    if isempty(results)
        throw(ArgumentError("Results DataFrame is empty."))
    end

    if !("ANCHOR" in names(bank))
        throw(ArgumentError("The column 'ANCHOR' does not exist in the bank DataFrame."))
    end

    num_forms = size(results, 2)

    # Prepare the header
    header = ["Form ID", "Total Items", "Anchor Items", "Non-anchor Items"]

    # Prepare the table data
    table_data = []
    for i in 1:num_forms
        selected_items = collect(skipmissing(results[:, i]))
        total_items = length(selected_items)
        anchor_items = sum(in(selected_items).(bank[bank.ANCHOR .> 0, :ID]))
        non_anchor_items = total_items - anchor_items

        # Append the row
        push!(table_data, [i, total_items, anchor_items, non_anchor_items])
    end

    # Convert to a matrix and display the table using PrettyTables
    table_matrix = hcat(table_data...)

    # Display the table with a title
    pretty_table(table_matrix'; header=header, title="Item Distribution",
                 alignment=:r)
    return nothing
end

"""
    display_column_sums(results::DataFrame, bank::DataFrame, column_names)

Displays the sum of values from one or more specified columns in the bank DataFrame for each form
in the results DataFrame.
Throws an exception if the `results` DataFrame is empty or if specified column names are invalid.
"""
function display_column_sums(results::DataFrame, bank::DataFrame, column_names)
    if isempty(results)
        throw(ArgumentError("Results DataFrame is empty."))
    end

    num_forms = size(results, 2)  # Number of forms
    item_ids = bank[:, :ID]  # Access the ID column

    # Normalize the input to wrap a single column in a vector if needed
    column_names = typeof(column_names) in [String, Symbol] ? [column_names] : column_names

    # Convert column names to strings for easier matching
    column_names = String.(column_names)

    # Check that the columns exist in the bank DataFrame
    for column_name in column_names
        if !(column_name in names(bank)) || isempty(column_name)
            throw(ArgumentError("The column '$column_name' does not exist in the bank DataFrame or is invalid."))
        end
    end

    # Prepare table data
    form_ids = collect(1:num_forms)
    table_data = [form_ids]

    # Gather the sums for each form
    for col in column_names
        column_sums = [sum(bank[findall(in(skipmissing(results[:, i])), item_ids), col])
                       for i in 1:num_forms]
        push!(table_data, column_sums)
    end

    # Convert to a matrix
    table_matrix = hcat(table_data...)

    # Set up the header
    header = ["Form ID"; column_names...]

    # Display the table using PrettyTables
    pretty_table(table_matrix; header=header,
                 title="Sum of values in columns $column_names",
                 alignment=:r)
    return nothing
end

"""
    display_category_counts(results::DataFrame, bank::DataFrame, column_name::Union{String, Symbol}; max_categories::Int=10)

Displays a table with counts of items in the specified column grouped by their value (category) for each form.
Throws an exception if the `results` DataFrame or column names are invalid or missing.
If the number of categories exceeds `max_categories` (default is 10), rows with only zero values are removed.
Handles cases where the specified column contains `missing` values.
"""
function display_category_counts(results::DataFrame, bank::DataFrame,
                                 column_name::Union{String, Symbol};
                                 max_categories::Int=10)
    if isempty(results)
        throw(ArgumentError("Results DataFrame is empty."))
    end

    num_forms = size(results, 2)  # Number of forms

    # Ensure the column exists in the bank DataFrame
    if !(String(column_name) in names(bank)) || isempty(column_name)
        throw(ArgumentError("The column '$column_name' does not exist in the bank DataFrame or is invalid."))
    end

    # Get unique categories, excluding missing values
    categories = sort(unique(collect(skipmissing(bank[!, Symbol(column_name)]))))

    # Filter rows where the column is not missing
    non_missing_bank = filter(row -> !ismissing(row[Symbol(column_name)]), bank)

    # Prepare the header
    header = ["Category"; [string("Form ", i) for i in 1:num_forms]...]

    # Prepare the table data
    table_data = []
    for category in categories
        category_counts = []
        for i in 1:num_forms
            selected_items = collect(skipmissing(results[:, i]))  # Non-missing items for the form
            count = sum(non_missing_bank[in(selected_items).(non_missing_bank.ID),
                                         Symbol(column_name)] .== category)
            push!(category_counts, count)
        end

        # Include the row if it has non-zero counts or if categories <= max_categories
        if sum(category_counts) > 0 || length(categories) <= max_categories
            push!(table_data, [category; category_counts...])
        end
    end

    # Convert table data to matrix
    table_matrix = reduce(hcat, table_data)

    # Display the table
    pretty_table(table_matrix'; header=header,
                 title="Items by Category in $column_name", alignment=:r)
    return nothing
end

"""
    calculate_common_items(results::DataFrame)

Calculates the number of common items between forms, returning a matrix where
each entry [i, j] represents the number of common items between form `i` and form `j`.
Throws an exception if the `results` DataFrame is missing or empty.
"""
function calculate_common_items(results::DataFrame)
    if isempty(results)
        throw(ArgumentError("Results DataFrame is empty."))
    end

    num_forms = size(results, 2)
    common_items_matrix = zeros(Int, num_forms, num_forms)

    for i in 1:num_forms, j in 1:num_forms
        # Handle missing values before comparing items between forms
        common = in(skipmissing(results[:, i])).(skipmissing(results[:, j]))
        common_items_matrix[i, j] = sum(common)
    end

    return common_items_matrix
end

"""
    display_common_items(results::DataFrame)

Displays a matrix showing the number of common items between each pair of forms.
Throws an exception if the `results` DataFrame is missing or empty.
"""
function display_common_items(results::DataFrame)
    if isempty(results)
        throw(ArgumentError("Results DataFrame is empty."))
    end

    common_items_matrix = calculate_common_items(results)
    num_forms = size(results, 2)

    # Prepare the header
    header = [""; collect(1:num_forms)...]

    # Add row numbers (Form IDs) to the matrix
    table_matrix = hcat(collect(1:num_forms), common_items_matrix)

    # Display the table with a title
    pretty_table(table_matrix; header=header, title="Items shared by Forms",
                 alignment=:r)

    return common_items_matrix
end

function display_final_results(parms::Parameters, config::Config, results::DataFrame)
    bank = parms.bank
    items = bank.ID
    anchor_items = bank[bank.ANCHOR .> 0, :ID]
    items_used = unique(skipmissing(reduce(vcat, eachcol(results))))
    anchors_used = anchor_items[anchor_items .∈ Ref(items_used)]
    non_anchor_used = setdiff(items_used, anchors_used)

    # General information summary
    print_title_and_separator("Summary of Forms and Items")
    println("Total forms assembled: ", size(results, 2))
    println("Total items used (includes anchor): ", length(items_used))
    println("Non-anchor items used: ", length(non_anchor_used))
    println("Anchor items used: ", length(anchors_used))
    println("Items not used (without anchor): ",
            length(items) - length(anchor_items) - length(items_used))

    # Display item distribution across forms
    display_item_distribution(results, bank)

    # Display item distribution by area across forms
    if !isempty(config.report_categories)
        for category in config.report_categories
            display_category_counts(results, bank, category)
        end
    end

    if !isempty(config.report_sums)
        display_column_sums(results, bank, config.report_sums)
    end

    # Display common items matrix
    display_common_items(results)

    return nothing
end

"""
    save_forms(parms::Parameters, results::DataFrame, config::Config)

Save the forms to a file, marking used items with a checkmark.
"""
function save_forms(parms::Parameters, results::DataFrame, config)
    bank = deepcopy(parms.bank)

    for v in names(results)
        # Handle missing values when checking if an item is used
        bank[!, Symbol(v)] = map(x -> ismissing(x) ? "" : (x == 1 ? v : ""),
                                 bank.ID .∈ Ref(skipmissing(results[:, v])))
    end

    # Save the results
    write_results_to_file(bank, config.forms_file)
    write_results_to_file(results, config.results_file)

    # Display file paths clearly
    println("\nSaved Forms and Results:")
    println("========================")
    println("Forms saved to: ", config.forms_file)
    println("Results saved to: ", config.results_file)
    return nothing
end

function final_report(original_parms::Parameters, results::DataFrame, config::Config)
    # Display common items matrix and summary
    display_final_results(original_parms, config, results)
    # Save the forms and results
    save_forms(original_parms, results, config)
    # Generate plots (this would remain unchanged from your original script)
    plot_results(original_parms, config, results)

    return nothing
end
