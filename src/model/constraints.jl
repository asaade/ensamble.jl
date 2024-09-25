using JuMP

# Helper function to handle shadow test adjustments for form count
function get_num_forms(x, shadow_test)
    forms = size(x, 2)
    return shadow_test > 0 ? forms - 1 : forms
end

# ---------------------------------------------------------------------------
# Constraint Functions for Automatic Test Assembly (ATA)
# ---------------------------------------------------------------------------

"""
    constraint_items_per_form(model::Model, parms::Parameters, minItems::Int, maxItems::Int=minItems)

Sets a constraint on the number of items in each form (test). The number of items must be between `minItems`
and `maxItems` for each form.
"""
function constraint_items_per_form(model::Model, parms::Parameters, minItems::Int, maxItems::Int=minItems)
    return constraint_item_count(model::Model, parms::Parameters, trues(size(parms.bank, 1)), minItems::Int, maxItems::Int)
end

"""
    constraint_item_count(model::Model, parms::Parameters, selected::BitVector, minItems::Int, maxItems::Int=minItems)

Combines the item count constraints for both operational forms and the shadow test (if applicable).
Ensures that the number of selected items is within the specified range.
"""
function constraint_item_count(model::Model, parms::Parameters, selected::BitVector, minItems::Int, maxItems::Int=minItems)
    @assert(minItems <= maxItems, "Error in item_count: maxItems < minItems")

    x = model[:x]
    items = collect(1:size(x, 1))[selected]
    forms = get_num_forms(x, parms.shadow_test)  # Get number of operational forms

    # Apply constraints for operational forms
    @constraint(model, [f = 1:forms], sum(x[i, f] for i in items) >= minItems)
    @constraint(model, [f = 1:forms], sum(x[i, f] for i in items) <= maxItems)

    # Apply constraints for shadow test if applicable
    if parms.shadow_test > 0
        shadow_test_col = size(x, 2)  # Last column is shadow test
        @constraint(model, sum(x[i, shadow_test_col] for i in items) >= minItems * parms.shadow_test)
        @constraint(model, sum(x[i, shadow_test_col] for i in items) <= maxItems * parms.shadow_test)
    end

    return model
end


"""
    constraint_item_sum(model::Model, parms::Parameters, vals, minVal, maxVal=minVal)

Combines the item sum constraints for operational forms and the shadow test.
Ensures the sum of item values is between `minVal` and `maxVal` for both operational and shadow forms.
"""
function constraint_item_sum(model::Model, parms::Parameters, vals, minVal, maxVal=minVal)
    @assert(minVal <= maxVal, "Error in item_sum: maxVal < minVal")

    x = model[:x]
    items, forms = size(x)
    forms = get_num_forms(x, parms.shadow_test)  # Get the number of operational forms

    # Ensure `vals` is handled consistently, whether a vector or a matrix
    val = size(vals, 2) == 1 ? vals[:, 1] : vals[:, 2]
    cond = size(vals, 2) == 1 ? trues(length(vals)) : vals[:, 1]

    # Apply sum constraints for operational forms
    @constraint(model, [f = 1:forms], sum(x[i, f] * val[i] for i in 1:items if cond[i]) >= minVal)
    @constraint(model, [f = 1:forms], sum(x[i, f] * val[i] for i in 1:items if cond[i]) <= maxVal)

    # Apply sum constraints for shadow test if applicable
    if parms.shadow_test > 0
        zcol = size(x, 2)  # Last column for shadow test
        @constraint(model, sum(x[i, zcol] * val[i] for i in 1:items if cond[i]) >= minVal * parms.shadow_test)
        @constraint(model, sum(x[i, zcol] * val[i] for i in 1:items if cond[i]) <= maxVal * parms.shadow_test)
    end

    return model
end


"""
    group_items_by_selected(selected::Vector)

Helper function that groups items based on the `selected` vector, ignoring missing values.
Returns a grouped DataFrame.
"""
function group_items_by_selected(selected::Vector)
    data = DataFrame(; selected=selected, index=1:length(selected))
    dropmissing!(data, :selected)
    return groupby(data, :selected; sort=false, skipmissing=true)
end


"""
    constraint_friends_in_form(model::Model, parms::Parameters, selected)

Adds constraints that force friend items (items that should always be together) to be assigned
to the same test form.
"""
function constraint_friends_in_form(model::Model, parms::Parameters, selected)
    x = model[:x]
    forms = get_num_forms(x, parms.shadow_test)

    # Group items by the selected vector
    groups = group_items_by_selected(selected)

    # Apply constraints to ensure that all friend items are together in the same form
    for group in groups
        items = group[!, :index]
        pivot = items[1]  # Choose the first item as the reference
        cnt = length(items)  # Number of items in the group
        @constraint(model, [f = 1:forms], sum(x[i, f] for i in items) == cnt * x[pivot, f])
    end
end


"""
    constraint_enemies_in_form(model::Model, parms::Parameters, selected)

Adds constraints that prevent conflicting (enemy) items from being assigned to the same test form.
"""
function constraint_enemies_in_form(model::Model, parms::Parameters, selected)
    x = model[:x]
    forms = get_num_forms(x, parms.shadow_test)

    # Group items by the selected vector
    groups = group_items_by_selected(selected)

    # Apply constraints to ensure that only one enemy item appears in each form
    for group in groups
        items = group[!, :index]
        @constraint(model, [f = 1:forms], sum(x[i, f] for i in items) <= 1)
    end
end


"""
    constraint_exclude_items(model::Model, parms::Parameters, selected::BitVector)

Ensures that the selected items do not appear in any test form.
Throws an error if any of the selected items are part of the anchor test items.
"""
function constraint_exclude_items(model::Model, parms::Parameters, selected::BitVector)
    x = model[:x]
    forms = size(x, 2)  # Number of forms (columns in x)

    # Collect the indices of the selected items
    selected_items = collect(1:size(x, 1))[selected]

    # Find all anchor items from the bank
    anchor_items = findall(parms.bank.ANCHOR .> 0)

    # Check if any selected items are part of the anchor items
    conflicting_items = intersect(selected_items, anchor_items)
    if !isempty(conflicting_items)
        error("The following selected items are part of the anchor test and cannot be excluded: $conflicting_items")
    end

    # Apply the constraint to prevent these items from appearing in any form
    @constraint(model, [i in selected_items], sum(x[i, f] for f in 1:forms) == 0)

    return model
end


"""
    constraint_include_items(model::Model, parms::Parameters, selected::BitVector)

Forces the inclusion of the selected items in all test forms.
"""
function constraint_include_items(model::Model, selected::BitVector)
    x = model[:x]
    forms = size(x, 2)  # Number of forms (columns in x)

    # Collect the indices of the selected items
    selected_items = collect(1:size(x, 1))[selected]

    # Force the inclusion of each selected item in all test forms
    for i in selected_items
        for f in 1:forms
            JuMP.fix(x[i, f], 1; force=true)
        end
    end
    return model
end



"""
    constraint_add_anchor!(model::Model, parms::Parameters)

Forces the inclusion of anchor items in operational forms, ignoring shadow forms.
"""
function constraint_add_anchor!(model::Model, parms::Parameters)
    if parms.anchor_tests > 0
        x = model[:x]
        forms = get_num_forms(x, parms.shadow_test)  # Number of operational forms

        # Collect anchor items based on the ANCHOR column in the bank DataFrame
        anchor_items = findall(parms.bank.ANCHOR .> 0)

        # Force anchor items to be included in operational forms
        for i in anchor_items
            for f in 1:forms
                JuMP.fix(x[i, f], 1; force=true)
            end
        end
    end
    return model
end


"""
    constraint_max_use(model::Model, parms::Parameters, selected::BitVector, max_use::Int)

Constrains the maximum number of times an item can appear in the test forms, excluding anchor items.
"""
function constraint_max_use(model::Model, parms::Parameters, selected::BitVector, max_use::Int)
    x = model[:x]
    forms = get_num_forms(x, parms.shadow_test)  # Number of operational forms

    # Proceed only if max_use is less than the number of forms
    if max_use < forms
        # Collect the indices of the selected items, excluding anchor items
        selected_items = collect(1:size(x, 1))[selected]
        non_anchor_items = filter(i -> parms.bank.ANCHOR[i] == 0, selected_items)

        # Apply the constraint to limit the maximum usage of non-anchor items across forms
        @constraint(model, max_use[i in non_anchor_items], sum(x[i, f] for f in 1:forms) + parms.bank.ITEM_USE[i] <= max_use)
    end

    return model
end


"""
    constraint_forms_overlap(model::Model, parms::Parameters, minItems::Int, maxItems::Int=minItems)

Constrains the number of overlapping items between test forms. The overlap between two forms must be
between `minItems` and `maxItems`.
"""
function constraint_forms_overlap(model::Model, parms::Parameters, minItems::Int, maxItems::Int=minItems)
    @assert(0 <= minItems <= maxItems, "Error in forms_overlap: maxItems < minItems")
    if parms.shadow_test < 1 && parms.anchor_tests == 0
        x = model[:x]
        num_items, num_forms = size(x)

        @variable(model, z[1:num_items, 1:num_forms, 1:num_forms], Bin)
        items = collect(1:num_items)
        items = items[(parms.bank.ANCHOR .!= parms.anchor_tests)]

        if minItems == maxItems
            @constraint(model, [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms], sum(z[i, t1, t2] for i in items) == maxItems)
        else
            @constraint(model, [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms], sum(z[i, t1, t2] for i in items) <= maxItems)
            @constraint(model, [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms], sum(z[i, t1, t2] for i in items) >= minItems)
        end

        @constraint(model, [i = items], 2 * z[i, t1, t2] <= x[i, t1] + x[i, t2])
        @constraint(model, [i = items], z[i, t1, t2] >= x[i, t1] + x[i, t2] - 1)
    end
end

# ---------------------------------------------------------------------------
# Objective Functions for Test Assembly Optimization
# ---------------------------------------------------------------------------

"""
    objective_match_characteristic_curve!(model::Model, parms::Parameters)

Matches the characteristic curve by constraining the sum of the item parameters raised to a power `r` for each form
and point `k`. Applies stricter weight adjustments for shadow tests, if present.
"""


function objective_match_characteristic_curve!(model::Model, parms::Parameters)
    R, K = 1:(parms.r), 1:(parms.k)
    P, tau = parms.p, parms.tau
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms
    forms -= parms.shadow_test > 0 ? 1 : 0

    # Initialize the weight vector `w[r]` for operational forms
    w = [1.0 for _ in R]

    # Constraints for operational forms
    @constraint(model,
                [f = 1:forms, k = K, r = R],
                sum(P[i, k]^r * x[i, f] for i in 1:items) <= tau[r, k] + (w[r] * y))

    @constraint(model,
                [f = 1:forms, k = K, r = R],
                sum(P[i, k]^r * x[i, f] for i in 1:items) >= tau[r, k] - (w[r] * y))

    # Constraints for shadow tests (if applicable)
    if parms.shadow_test > 0
        shadow_test = parms.shadow_test
        # Stricter weights for the shadow test
        w_shadow = [1.0, 0.8, 0.7, 0.75]

        @constraint(model, [k = K, r = R],
                    sum(P[i, k]^r * x[i, zcol] for i in 1:items) <=
                    ((tau[r, k] + (w_shadow[r] * y)) * shadow_test))

        @constraint(model, [k = K, r = R],
                    sum(P[i, k]^r * x[i, zcol] for i in 1:items) >=
                    ((tau[r, k] - (w_shadow[r] * y)) * shadow_test))
    end

    return model
end


"""
    objective_match_information_curve!(model::Model, parms::Parameters)

Constrains the information curve, ensuring that the sum of `info[i, k] * x[i, f]` stays within bounds
(`tau_info[k] ± y`) for operational forms and shadow tests.
"""
function objective_match_information_curve!(model::Model, parms::Parameters)
    K, info, tau_info = parms.k, parms.info, parms.tau_info
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms
    forms -= parms.shadow_test > 0 ? 1 : 0

    # Constraints for operational forms
    @constraint(model,
                [f = 1:forms, k = 1:K],
                sum(info[i, k] * x[i, f] for i in 1:items) <= tau_info[k] + y)

    @constraint(model,
                [f = 1:forms, k = 1:K],
                sum(info[i, k] * x[i, f] for i in 1:items) >= tau_info[k] - y)

    # Constraints for shadow test forms (if applicable)
    if parms.shadow_test > 0
        shadow_test = parms.shadow_test

        @constraint(model,
                    [k = 1:K],
                    sum(info[i, k] * x[i, zcol] for i in 1:items) <=
                    (tau_info[k] + y) * shadow_test)

        @constraint(model,
                    [k = 1:K],
                    sum(info[i, k] * x[i, zcol] for i in 1:items) >=
                    (tau_info[k] - y) * shadow_test)
    end

    return model
end


"""
    objective_max_info(model::Model, parms::Parameters)

Maximizes information at each point `k`, ensuring that the sum of `info[i, k] * x[i, f]` meets or exceeds the
weighted target `R[k] * y` for both operational forms and shadow tests.
"""
function objective_max_info(model::Model, parms::Parameters)
    R = parms.relative_target_weights
    K = parms.k
    info = parms.info
    x, y = model[:x], model[:y]
    items, forms = size(x)

    shadow = parms.shadow_test
    forms -= shadow > 0 ? 1 : 0

    # Constraints for operational forms
    @constraint(model,
                [f = 1:forms, k = 1:K],
                sum(info[i, k] * x[i, f] for i in 1:items) >= R[k] * y)

    # Constraints for shadow test forms (if applicable)
    if shadow > 0
        @constraint(model,
                    [k = 1:K],
                    sum(info[i, k] * x[i, forms + 1] for i in 1:items) >= R[k] * y * shadow)
    end

    return model
end


"""
    objective_info_relative2(model::Model, parms::Parameters)

Maximizes information at alternating points `k` across forms, ensuring that each form meets or exceeds the weighted
target `R[k] * y` at its corresponding point.
"""
function objective_info_relative2(model::Model, parms::Parameters)
    R = parms.relative_target_weights
    K = length(parms.relative_target_weights)
    info = parms.info
    x, y = model[:x], model[:y]
    items, forms = size(x)

    k = 0
    for f in 1:forms
        # Alternate k for each form
        k = k % K + 1
        @constraint(model,
                    sum(info[i, k] * x[i, f] for i in 1:items) >= R[k] * y)
    end

    return model
end
