using JuMP, DataFrames


# Print functions
function print_title_and_separator(title::String)
    println("\n" * title)
    return println(repeat("=", length(title)))
end

function print_optimization_results(model::Model, parms::Parameters)
    parms.verbose > 1 && println(solution_summary(model))
    println(TOLERANCE_LABEL, round(objective_value(model); digits = 4))
    return println(SEPARATOR)
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
    display_item_distribution(results::DataFrame, bank::DataFrame)

Displays a summary of item distribution across forms, showing total items, anchor, and non-anchor.
"""
function display_item_distribution(results::DataFrame, bank::DataFrame)
    num_forms = size(results, 2)

    println("\nItem Distribution Across Forms:")
    println("===========================================================")
    println("| Form ID  | Total Items | Anchor Items | Non-anchor Items |")
    println("===========================================================")

    for i in 1:num_forms
        selected_items = collect(skipmissing(results[:, i]))
        total_items = length(selected_items)
        anchor_items = sum(in(selected_items).(bank[bank.ANCHOR .> 0, :ID]))
        non_anchor_items = total_items - anchor_items
        println("| Form  $i |     $total_items      |      $anchor_items      |        $non_anchor_items        |")
    end

    println("===========================================================")
end

"""
    display_area_distribution(results::DataFrame, bank::DataFrame)

Displays the distribution of items by area for each form, showing the count of items from each area.
"""
function display_area_distribution(results::DataFrame, bank::DataFrame)
    num_forms = size(results, 2)
    areas = sort(unique(bank.AREA))  # Get the unique areas from the bank

    println("\nItem Distribution by Area Across Forms:")
    println("=================================")
    println("| Form ID  | " * join(["$area" for area in areas], "  | ") * "  |")
    println("=================================")

    for i in 1:num_forms
        selected_items = collect(skipmissing(results[:, i]))
        area_counts = [sum(in(selected_items).(bank[bank.AREA .== area, :ID])) for area in areas]
        println("|  Form $i  | " * join([string(count) for count in area_counts], " | ") * " |")
    end

    println("=================================")
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
    common_items_matrix = calculate_common_items(results)
    print_title_and_separator("Common Items Matrix")
    display(common_items_matrix)
    println(SEPARATOR)
    return common_items_matrix
end

function display_final_results(parms::Parameters, results::DataFrame)
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
    println("Items not used (without anchor): ", length(items) - length(anchor_items) - length(items_used))

    # Display item distribution across forms
    display_item_distribution(results, bank)

    # Display item distribution by area across forms
    display_area_distribution(results, bank)

    # Display common items matrix
    return display_common_items(results)
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
end

function final_report(original_parms::Parameters, results::DataFrame, config::Config)
    # Display common items matrix and summary
    display_final_results(original_parms, results)

    # Generate plots (this would remain unchanged from your original script)
    plot_results(original_parms, config, results)

    # Save the forms and results
    return save_forms(original_parms, results, config)
end
