using JuMP
using DataFrames
using PrettyTables

using .Ensamble.Configuration

# Print functions
function print_title_and_separator(title::String)
    println("\n" * title)
    println(repeat("=", length(title)))
    return nothing
end

function print_optimization_results(model::Model, parms::Parameters)
    parms.verbose > 1 && println(solution_summary(model))
    println(TOLERANCE_LABEL, round(objective_value(model); digits=4))
    println(SEPARATOR)
    return nothing
end

"""
    display_results(model::Model)

Display the results of the optimization, including the tolerance and objective value.
"""
function display_results(model::Model, parms::Parameters)
    print_title_and_separator(OPTIMIZATION_RESULTS_TITLE)
    print_optimization_results(model, parms)
    return nothing
end

"""
    display_item_distribution(results::DataFrame, bank::DataFrame)

Displays a summary of item distribution across forms, showing total items, anchor, and non-anchor.
"""
function display_item_distribution(results::DataFrame, bank::DataFrame)
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

        # Append row
        push!(table_data, [i, total_items, anchor_items, non_anchor_items])
    end

    # Convert to a matrix and display the table using PrettyTables.jl
    table_matrix = hcat(table_data...)

    # Display the table with title
    pretty_table(table_matrix'; header=header, title="Item Distribution",
                 alignment=:r)
    return nothing
end

"""
    display_column_sums(results::DataFrame, bank::DataFrame, column_names)

Displays a table that shows the sum of values from one or more specified columns of the bank DataFrame for each form in the results DataFrame.
"""
function display_column_sums(results::DataFrame, bank::DataFrame, column_names)
    num_forms = size(results, 2)  # Number of forms (columns in the results)
    item_ids = bank[:, :ID]  # Corrected access to ID column

    # Normalize the input: wrap single column in a vector if needed
    column_names = typeof(column_names) in [String, Symbol] ? [column_names] : column_names

    # Convert all column names to strings for easier matching
    column_names = String.(column_names)

    # Ensure the specified columns exist in the bank DataFrame
    for column_name in column_names
        if !(column_name in names(bank))
            error("The column '$column_name' does not exist in the bank DataFrame.")
        end
    end

    # Prepare the table data column by column
    form_ids = collect(1:num_forms)  # Form IDs as the first column
    table_data = [form_ids]  # Initialize with form IDs

    # Iterate over each column name and gather the sums for each form
    for col in column_names
        column_sums = [sum(bank[findall(in(skipmissing(results[:, i])), item_ids), col])
                       for i in 1:num_forms]
        push!(table_data, column_sums)  # Add each column of sums to the table_data
    end

    # Transpose table_data and convert it to a matrix
    table_matrix = hcat(table_data...)  # Combine columns into a matrix

    # Set up the header
    header = ["Form ID"; column_names...]  # Prepare the header row

    # Print the table using PrettyTables.jl
    pretty_table(table_matrix; header=header,
                 title="Sum of values in columns $column_names",
                 alignment=:r)
    return nothing
end

"""
    display_category_counts(results::DataFrame, bank::DataFrame, column_name::Union{String, Symbol}; max_categories::Int=10)

Displays a table with counts of items in the specified column grouped by their value (category) for each form.
If the total number of categories exceeds `max_categories` (default 10), rows with only zeros will be removed.
Handles cases where the column contains `missing` values.
"""
function display_category_counts(results::DataFrame, bank::DataFrame,
                                 column_name::Union{String, Symbol};
                                 max_categories::Int=10)
    num_forms = size(results, 2)  # Number of forms (columns in the results)

    # Ensure the column exists in the bank DataFrame
    if !(String(column_name) in names(bank))
        error("The column '$column_name' does not exist in the bank DataFrame.")
    end

    # Extract the unique categories from the specified column, excluding missing values
    categories = sort(unique(collect(skipmissing(bank[!, Symbol(column_name)]))))

    # Filter rows where the selected column is not missing
    non_missing_bank = filter(row -> !ismissing(row[Symbol(column_name)]), bank)

    # Prepare the header: Category + Form IDs
    header = ["Category"; [string("Form ", i) for i in 1:num_forms]...]

    # Prepare the table data
    table_data = []

    for category in categories
        # Count the items in each form for the current category
        category_counts = []
        for i in 1:num_forms
            selected_items = collect(skipmissing(results[:, i]))  # Get non-missing items for this form
            # Count how many items in the form belong to the current category
            count = sum(non_missing_bank[in(selected_items).(non_missing_bank.ID),
                                         Symbol(column_name)] .== category)
            push!(category_counts, count)
        end

        # Only append the row if it contains non-zero counts or if the total categories are <= max_categories
        if sum(category_counts) > 0 || length(categories) <= max_categories
            push!(table_data, [category; category_counts...])  # Don't convert category to string here
        end
    end

    # Convert the table_data to a matrix (each element is a row vector)
    table_matrix = reduce(hcat, table_data)

    # Display the table using PrettyTables.jl
    pretty_table(table_matrix'; header=header,
                 title="Items by Category in $column_name", alignment=:r)
    return nothing
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
        # Handle missing values before performing the common item comparison
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
    common_items_matrix = calculate_common_items(results)  # Assume this returns the matrix
    num_forms = size(results, 2)

    # Prepare the header (Form IDs)
    header = [""; collect(1:num_forms)...]  # First column for row names, followed by form IDs

    # Convert the common items matrix to a form that pretty_table can handle
    table_matrix = hcat(collect(1:num_forms), common_items_matrix)  # Add row numbers (Form IDs) to the matrix

    # Display the table with title
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
