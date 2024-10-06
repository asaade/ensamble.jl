module Constraints

# Export all functions for external use
export constraint_items_per_form, constraint_item_count,
       constraint_item_sum, group_by_selected, constraint_friends_in_form,
       constraint_enemies_in_form, constraint_exclude_items, constraint_include_items,
       constraint_add_anchor!, constraint_max_use, constraint_forms_overlap,
       objective_match_characteristic_curve!, objective_match_information_curve!,
       objective_max_info, objective_info_relative2

#  using Infiltrator

using DataFrames
using JuMP

using ..Configuration

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

"""
    operational_forms(x, shadow_test) -> Int

Helper function to calculate the number of operational forms, adjusting for shadow tests.
"""
function operational_forms(x, shadow_test)
    forms = size(x, 2)
    return shadow_test > 0 ? forms - 1 : forms
end

"""
    group_by_selected(selected::Vector) -> GroupedDataFrame

Helper function to group items based on the `selected` vector, ignoring missing values.
"""
function group_by_selected(selected::Union{Vector, BitVector})
    data = DataFrame(; selected=selected, index=1:length(selected))
    if isa(selected, BitVector)
        data = data[data.selected, :]
    elseif eltype(data.selected) <: AbstractString
        data = data[data.selected !== missing, :]
    elseif eltype(data.selected) <: Number
        data = data[data.selected !== missing || data.selected .> 0, :]
    end
    return groupby(data, :selected; skipmissing=true)
end

# ---------------------------------------------------------------------------
# Constraint Functions for item counts
# ---------------------------------------------------------------------------

"""
    constraint_items_per_form(model::Model, parms::Parameters, minItems::Int, maxItems::Int=minItems)

Sets a constraint on the number of items in each form (test). The number of items must be between `minItems`
and `maxItems` for each form.
"""
function constraint_items_per_form(model::Model, parms::Parameters, minItems::Int,
                                   maxItems::Int=minItems)
    return constraint_item_count(model, parms, trues(size(parms.bank, 1)), minItems,
                                 maxItems)
end

"""
    constraint_item_count(model::Model, parms::Parameters, selected::BitVector, minItems::Int, maxItems::Int=minItems)

Combines the item count constraints for both operational forms and the shadow test (if applicable).
Ensures that the number of selected items is within the specified range.
"""
function constraint_item_count(model::Model, parms::Parameters, selected::BitVector,
                               minItems::Int, maxItems::Int=minItems)
    @assert(minItems <= maxItems, "Error in item_count: maxItems < minItems")

    x = model[:x]
    items = collect(1:size(x, 1))[selected]
    forms = operational_forms(x, parms.shadow_test)

    @constraint(model, [f = 1:forms], sum(x[i, f] for i in items) >= minItems)
    @constraint(model, [f = 1:forms], sum(x[i, f] for i in items) <= maxItems)

    if parms.shadow_test > 0
        shadow_test_col = size(x, 2)
        @constraint(model,
                    sum(x[i, shadow_test_col] for i in items) >=
                    minItems * parms.shadow_test)
        @constraint(model,
                    sum(x[i, shadow_test_col] for i in items) <=
                    maxItems * parms.shadow_test)
    end

    return model
end

# ---------------------------------------------------------------------------
# Constraint Functions for item value sums
# ---------------------------------------------------------------------------

"""
    constraint_item_sum(model::Model, parms::Parameters, vals, minVal, maxVal=minVal)

Combines the item sum constraints for operational forms and the shadow test.
Ensures the sum of item values is between `minVal` and `maxVal` for both operational and shadow forms.
"""
function constraint_item_sum(model::Model, parms::Parameters, vals, minVal, maxVal=minVal)
    @assert(minVal <= maxVal, "Error in item_sum: maxVal < minVal")

    x = model[:x]
    items, forms = size(x)
    forms = operational_forms(x, parms.shadow_test)

    val = size(vals, 2) == 1 ? vals[:, 1] : vals[:, 2]
    cond = size(vals, 2) == 1 ? trues(length(vals)) : vals[:, 1]

    @constraint(model, [f = 1:forms],
                sum(x[i, f] * val[i] for i in 1:items if cond[i]) >= minVal)
    @constraint(model, [f = 1:forms],
                sum(x[i, f] * val[i] for i in 1:items if cond[i]) <= maxVal)

    if parms.shadow_test > 0
        zcol = size(x, 2)
        @constraint(model,
                    sum(x[i, zcol] * val[i] for i in 1:items if cond[i]) >=
                    minVal * parms.shadow_test)
        @constraint(model,
                    sum(x[i, zcol] * val[i] for i in 1:items if cond[i]) <=
                    maxVal * parms.shadow_test)
    end

    return model
end

# ---------------------------------------------------------------------------
# Constraint Functions for item groups (friends, enemies, anchors)
# ---------------------------------------------------------------------------

"""
    constraint_friends_in_form(model::Model, parms::Parameters, selected)

Adds constraints to ensure that friend items (items that must always be together) are assigned to the same form.
"""
function constraint_friends_in_form(model::Model, parms::Parameters, selected)
    x = model[:x]
    forms  = operational_forms(x, parms.shadow_test)
    groups = group_by_selected(selected)

    for group in groups
        items = group[!, :index]
        pivot = items[1]  # Choose the first item as the reference
        cnt = length(items)

        # Only add constraints if the group has more than one item
        if cnt > 1
            @constraint(model, [f in 1:forms], sum(x[i, f] for i in items) == (cnt * x[pivot, f]))
        end
    end
end


"""
    constraint_enemies_in_form(model::Model, parms::Parameters, selected)

Adds constraints to prevent enemy items (items that should not appear together) from being assigned to the same form.
"""
function constraint_enemies_in_form(model::Model, parms::Parameters, selected)
    x = model[:x]
    forms  = operational_forms(x, parms.shadow_test)

    if isa(selected, BitVector)
        items = 1:size(x, 1)
        items = items[selected]
        if length(items) > 1
            @constraint(model, [f = 1:forms], sum(x[i, f] for i in items) <= 1)
        end
    else
        groups = group_by_selected(selected)
        for group in groups
            items = group[!, :index]
            @constraint(model, [f = 1:forms], sum(x[i, f] for i in items) <= 1)
        end
    end
end

"""
    constraint_exclude_items(model::Model, parms::Parameters, selected::BitVector)

Ensures that the selected items do not appear in any test form.
Throws an error if any of the selected items are part of the anchor test items.
"""
function constraint_exclude_items(model::Model, parms::Parameters, selected::BitVector)
    x = model[:x]
    forms = size(x, 2)

    selected_items = collect(1:size(x, 1))[selected]
    anchor_items = findall(parms.bank.ANCHOR .!== missing)
    conflicting_items = intersect(selected_items, anchor_items)

    if !isempty(conflicting_items)
        error("The following selected items are part of the anchor test and cannot be excluded: $conflicting_items")
    end

    @constraint(model, [i in selected_items], sum(x[i, f] for f in 1:forms) == 0)
    return model
end

"""
    constraint_include_items(model::Model, selected::BitVector)

Forces the inclusion of the selected items in all test forms.
"""
function constraint_include_items(model::Model, selected::BitVector)
    x = model[:x]
    forms = size(x, 2)
    selected_items = collect(1:size(x, 1))[selected]

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
        forms = operational_forms(x, parms.shadow_test)
        anchor_items = findall(parms.bank.ANCHOR .!== missing)

        # @infiltrate

        for i in anchor_items
            for f in 1:forms
                JuMP.fix(x[i, f], 1; force=true)
            end
        end
    end
    return model
end

# ---------------------------------------------------------------------------
# Constraint Functions for item sharing between forms
# ---------------------------------------------------------------------------

"""
    constraint_max_use(model::Model, parms::Parameters, selected::BitVector, max_use::Int)

Constrains the maximum number of times an item can appear in the test forms, excluding anchor items.
"""
function constraint_max_use(model::Model, parms::Parameters, selected::BitVector,
                            max_use::Int)
    x = model[:x]
    forms = operational_forms(x, parms.shadow_test)

    if max_use < forms
        selected_items = collect(1:size(x, 1))[selected]
        non_anchor_items =
            filter(i -> ismissing(parms.bank.ANCHOR[i]), selected_items)

        @constraint(model, max_use[i in non_anchor_items],
                    sum(x[i, f] for f in 1:forms) + parms.bank.ITEM_USE[i] <= max_use)
    end

    return model
end

"""
    constraint_forms_overlap(model::Model, parms::Parameters, minItems::Int, maxItems::Int=minItems)

Constrains the number of overlapping items between test forms. The overlap between two forms must be between `minItems` and `maxItems`.
"""
function constraint_forms_overlap(model::Model, parms::Parameters, minItems::Int,
                                  maxItems::Int=minItems)
    @assert(0 <= minItems <= maxItems, "Error in forms_overlap: maxItems < minItems")

    if parms.shadow_test < 1 && parms.anchor_tests == 0
        x = model[:x]
        num_items, num_forms = size(x)

        @variable(model, z[1:num_items, 1:num_forms, 1:num_forms], Bin)
        items = collect(1:num_items)
        # items = items[parms.bank.ANCHOR .=== missing]

        if minItems == maxItems
            @constraint(model, [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                        sum(z[i, t1, t2] for i in items) == maxItems)
        else
            @constraint(model, [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                        sum(z[i, t1, t2] for i in items) <= maxItems)
            @constraint(model, [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                        sum(z[i, t1, t2] for i in items) >= minItems)
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

Matches the characteristic curve by constraining the sum of the item parameters raised to a power `r` for each form and point `k`.
"""
function objective_match_characteristic_curve!(model::Model, parms::Parameters)
    R, K = 1:(parms.r), 1:(parms.k)
    P, tau = parms.p, parms.tau
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms
    forms -= parms.shadow_test > 0 ? 1 : 0

    w = [1.0 for _ in R]

    @constraint(model, [f = 1:forms, k = K, r = R],
                sum(P[i, k]^r * x[i, f] for i in 1:items) <= tau[r, k] + (w[r] * y))
    @constraint(model, [f = 1:forms, k = K, r = R],
                sum(P[i, k]^r * x[i, f] for i in 1:items) >= tau[r, k] - (w[r] * y))

    if parms.shadow_test > 0
        shadow_test = parms.shadow_test
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

Constrains the information curve, ensuring that the sum of `info[i, k] * x[i, f]` stays within bounds (`tau_info[k] ± y`).
"""
function objective_match_information_curve!(model::Model, parms::Parameters)
    K, info, tau_info = parms.k, parms.info, parms.tau_info
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms
    forms -= parms.shadow_test > 0 ? 1 : 0

    @constraint(model, [f = 1:forms, k = 1:K],
                sum(info[i, k] * x[i, f] for i in 1:items) <= tau_info[k] + y)
    @constraint(model, [f = 1:forms, k = 1:K],
                sum(info[i, k] * x[i, f] for i in 1:items) >= tau_info[k] - y)

    if parms.shadow_test > 0
        shadow_test = parms.shadow_test
        @constraint(model, [k = 1:K],
                    sum(info[i, k] * x[i, zcol] for i in 1:items) <=
                    (tau_info[k] + y) * shadow_test)
        @constraint(model, [k = 1:K],
                    sum(info[i, k] * x[i, zcol] for i in 1:items) >=
                    (tau_info[k] - y) * shadow_test)
    end

    return model
end

"""
    objective_max_info(model::Model, parms::Parameters)

Maximizes information at each point `k`, ensuring that the sum of `info[i, k] * x[i, f]` meets or exceeds the weighted target `R[k] * y`.
"""
function objective_max_info(model::Model, parms::Parameters)
    R = parms.relative_target_weights
    K = parms.k
    info = parms.info
    x, y = model[:x], model[:y]
    items, forms = size(x)

    shadow = parms.shadow_test
    forms -= shadow > 0 ? 1 : 0

    @constraint(model, [f = 1:forms, k = 1:K],
                sum(info[i, k] * x[i, f] for i in 1:items) >= R[k] * y)

    if shadow > 0
        @constraint(model, [k = 1:K],
                    sum(info[i, k] * x[i, forms + 1] for i in 1:items) >= R[k] * y * shadow)
    end

    return model
end

"""
    objective_info_relative2(model::Model, parms::Parameters)

Maximizes information at alternating points `k` across forms, ensuring that each form meets or exceeds the weighted target `R[k] * y`.
"""
function objective_info_relative2(model::Model, parms::Parameters)
    R = parms.relative_target_weights
    K = length(parms.relative_target_weights)
    info = parms.info
    x, y = model[:x], model[:y]
    items, forms = size(x)

    k = 0
    for f in 1:forms
        k = k % K + 1
        @constraint(model, sum(info[i, k] * x[i, f] for i in 1:items) >= R[k] * y)
    end

    return model
end

end  # module ATAConstraints
