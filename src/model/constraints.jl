module Constraints

export constraint_items_per_form,
       constraint_item_count,
       constraint_score_sum,
       constraint_item_sum,
       constraint_friends_in_form,
       constraint_enemies,
       constraint_exclude_items,
       constraint_include_items,
       constraint_add_anchor!,
       constraint_max_use,
       constraint_forms_overlap,
       constraint_forms_overlap2, # Experimental
       objective_match_characteristic_curve!,
       objective_match_mean_var!,
       objective_match_information_curve!,
       objective_max_info,
       objective_info_relative2,
       constraint_fix_items

#

using ..Configuration
using DataFrames
using JuMP

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

"""
    operational_forms(x, shadow_test_size)

Helper function to calculate the number of operational forms, adjusting for shadow tests.
"""
function operational_forms(x, shadow_test_size::Int)
    forms = size(x, 2)
    return shadow_test_size > 0 ? forms - 1 : forms
end

"""
    group_by_selected(selected::Union{Vector, BitVector})

Helper function to group items based on the `selected` vector, ignoring missing values.
"""
function group_by_selected(selected::Union{Vector, BitVector})
    data = DataFrame(; selected = selected, index = 1:length(selected))
    if isa(selected, BitVector)
        data = data[data.selected, :]
    elseif eltype(data.selected) <: AbstractString
        data = data[data.selected !== missing, :]
    elseif eltype(data.selected) <: Number
        data = data[data.selected !== missing || data.selected .> 0, :]
    end
    return groupby(data, :selected; skipmissing = true)
end

# ---------------------------------------------------------------------------
# Constraint Functions for item counts
# ---------------------------------------------------------------------------

"""
    constraint_items_per_form(model::Model, parms::Parameters, minItems::Int, maxItems::Int=minItems)

Sets a constraint on the number of items in each form (test). The number of items must be between `minItems`
and `maxItems` for each form.
"""
function constraint_items_per_form(
        model::Model, parms::Parameters, minItems::Int, maxItems::Int = minItems
)
    return constraint_item_count(
        model, parms, trues(size(parms.bank, 1)), minItems, maxItems
    )
end

"""
    constraint_item_count(model::Model, parms::Parameters, selected::BitVector,
                          minItems::Int, maxItems::Int=minItems)

Constrains the count of selected items to fall within `minItems` and `maxItems` for each form.
"""
function constraint_item_count(model::Model, parms::Parameters, selected::BitVector,
        minItems::Int, maxItems::Int = minItems)
    @assert(minItems<=maxItems, "Error in item_count: maxItems < MinItems")

    x = model[:x]
    items = collect(1:size(x, 1))[selected]
    forms = operational_forms(x, parms.shadow_test_size)

    # Constraints for operational forms
    @constraint(model, [f = 1:forms], sum(x[items, f])>=minItems)
    @constraint(model, [f = 1:forms], sum(x[items, f])<=maxItems)

    # Shadow test constraints if applicable
    if parms.shadow_test_size > 0
        shadow_test_col = size(x, 2)

        if parms.anchor_tests > 0
            # Separate anchor and non-anchor items
            non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), items)
            anchor_count = sum(!ismissing(parms.bank.ANCHOR[i]) for i in items)
        else
            non_anchor_items = items
            anchor_count = 0
        end

        # Adjust min and max based on anchor item count
        effective_minItems = max(0, minItems - anchor_count)
        effective_maxItems = max(0, maxItems - anchor_count)

        # Shadow test constraints for non-anchor items
        @constraint(model,
            sum(x[non_anchor_items,
                shadow_test_col])>=effective_minItems * parms.shadow_test_size)
        @constraint(model,
            sum(x[non_anchor_items,
                shadow_test_col])<=effective_maxItems * parms.shadow_test_size)
    end

    return model
end

"""
    constraint_score_sum(model::Model, parms::Parameters, selected::BitVector,
                              minScore::Float64, maxScore::Float64=minScore)

Constrains the sum of selected item scores within `minScore` and `maxScore` bounds for each form.
"""
function constraint_score_sum(model::Model, parms::Parameters, selected::BitVector,
        minScore::Int64, maxScore::Int64 = minScore)
    @assert(minScore<=maxScore, "Error in item_score_sum: maxScore < minScore")

    x = model[:x]
    items = collect(1:size(x, 1))[selected]
    forms = operational_forms(x, parms.shadow_test_size)

    # Calculate scores based on num_categories in Parameters
    scores = [parms.bank.NUM_CATEGORIES[i] - 1 for i in items]

    # Constraints for operational forms
    @constraint(model, [f = 1:forms],
        sum(x[items[i], f] * scores[i] for i in eachindex(items))>=minScore)
    @constraint(model, [f = 1:forms],
        sum(x[items[i], f] * scores[i] for i in eachindex(items))<=maxScore)

    # Shadow test constraints if applicable
    if parms.shadow_test_size > 0
        shadow_test_col = size(x, 2)

        if parms.anchor_tests > 0
            # Separate anchor and non-anchor items within the selected items
            non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), items)
            anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), items)

            # Calculate anchor item contribution to the score sum
            anchor_score_sum = sum(scores[i] for i in 1:length(anchor_items))
            effective_minScore = max(0, minScore - anchor_score_sum)
            effective_maxScore = max(0, maxScore - anchor_score_sum)
        else
            non_anchor_items = items
            effective_minScore, effective_maxScore = minScore, maxScore
        end

        # Shadow test constraints for non-anchor items
        @constraint(model,
            sum(x[non_anchor_items[i], shadow_test_col] * scores[i]
            for i in eachindex(non_anchor_items))>=effective_minScore *
                                                   parms.shadow_test_size)
        @constraint(model,
            sum(x[non_anchor_items[i], shadow_test_col] * scores[i]
            for i in eachindex(non_anchor_items))<=effective_maxScore *
                                                   parms.shadow_test_size)
    end

    return model
end

# ---------------------------------------------------------------------------
# Constraint Functions for item value sums
# ---------------------------------------------------------------------------

"""
    constraint_item_sum(model::Model, parms::Parameters, vals, minVal, maxVal=minVal)

Constrains the sum of selected values to be within `minVal` and `maxVal` for each form.
"""
function constraint_item_sum(model::Model, parms::Parameters, vals, minVal, maxVal = minVal)
    @assert(minVal<=maxVal, "Error in item_sum: maxVal < minVal")

    x = model[:x]
    items, forms = size(x)
    forms = operational_forms(x, parms.shadow_test_size)

    # Separate val and condition columns, if provided
    val = ndims(vals) == 1 ? vals : vals[:, 2]
    cond = ndims(vals) == 1 ? trues(length(vals)) : vals[:, 1]

    # Separate anchor and non-anchor items if anchor tests are used
    if parms.anchor_tests > 0
        anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), 1:items)
        non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), 1:items)

        # Calculate anchor item contributions for sum
        anchor_val_sum = sum(val[i] for i in anchor_items if cond[i])

        # Adjust minVal and maxVal based on anchor items
        effective_minVal = max(0, minVal - anchor_val_sum)
        effective_maxVal = max(0, maxVal - anchor_val_sum)
    else
        non_anchor_items = 1:items
        effective_minVal, effective_maxVal = minVal, maxVal
    end

    # Constraints for operational forms
    @constraint(model, [f = 1:forms],
        sum(x[i, f] * val[i] for i in 1:items if cond[i])>=minVal)
    @constraint(model, [f = 1:forms],
        sum(x[i, f] * val[i] for i in 1:items if cond[i])<=maxVal)

    # Shadow test constraints if applicable
    if parms.shadow_test_size > 0
        zcol = size(x, 2)

        @constraint(model,
            sum(x[i, zcol] * val[i]
            for i in non_anchor_items if cond[i])>=
            effective_minVal * parms.shadow_test_size)
        @constraint(model,
            sum(x[i, zcol] * val[i]
            for i in non_anchor_items if cond[i])<=
            effective_maxVal * parms.shadow_test_size)
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
    forms = operational_forms(x, parms.shadow_test_size)
    groups = group_by_selected(selected)

    for group in groups
        items = group[!, :index]
        pivot = items[1]  # Choose the first item as the reference
        cnt = length(items)

        # Only add constraints if the group has more than one item
        if cnt > 1
            @constraint(model, [f in 1:forms],
                sum(x[i, f] for i in items)==(cnt * x[pivot, f]))
        end
    end
end

"""
    constraint_enemies(model::Model, parms::Parameters, selected)

Adds constraints to prevent enemy items (items that should not appear together) from being assigned to the same form.
"""
function constraint_enemies(model::Model, parms::Parameters, selected)
    x = model[:x]
    forms = operational_forms(x, parms.shadow_test_size)

    if isa(selected, BitVector)
        items = 1:size(x, 1)
        items = items[selected]
        if length(items) > 1
            @constraint(model, [f = 1:forms], sum(x[i, f] for i in items)<=1)
        end
    else
        groups = group_by_selected(selected)
        for group in groups
            items = group[!, :index]
            @constraint(model, [f = 1:forms], sum(x[i, f] for i in items)<=1)
        end
    end
end

"""
    constraint_exclude_items(model::Model, exclude::BitVector)

Excludes specified items from being selected in any form.
"""
function constraint_exclude_items(model::Model, exclude::BitVector)
    x = model[:x]
    items = collect(1:size(x, 1))[exclude]
    forms = size(x, 2)

    # Ensure no excluded items are selected
    for i in items
        for f in 1:forms
            JuMP.fix(x[i, f], 0; force = true)
        end
    end
    return model
end

"""
    constraint_fix_items(model::Model, fixed::BitVector)

Forces certain items to be included in specific forms.
"""
function constraint_fix_items(model::Model, fixed::BitVector)
    x = model[:x]
    items = collect(1:size(x, 1))[fixed]
    forms = size(x, 2)

    for i in items
        for f in 1:forms
            JuMP.fix(x[i, f], 1; force = true)
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
        forms = operational_forms(x, parms.shadow_test_size)
        anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), 1:size(x, 1))

        for i in anchor_items
            for f in 1:forms
                JuMP.fix(x[i, f], 1; force = true)
            end
        end
    end
    return model
end

# ---------------------------------------------------------------------------
# Constraint Functions for item sharing between forms
# ---------------------------------------------------------------------------

"""
    constraint_max_use(model::Model, parms::Parameters, selected::BitVector)

Constrains the maximum number of times an item can appear in the test forms, excluding anchor items.
"""
function constraint_max_use(
        model::Model, parms::Parameters, selected::BitVector
)
    x = model[:x]
    forms = operational_forms(x, parms.shadow_test_size)

    max_use = parms.max_item_use

    if max_use < forms
        selected_items = collect(1:size(x, 1))[selected]
        non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), selected_items)

        @constraint(model,
            max_use[i in non_anchor_items],
            sum(x[i, f] for f in 1:forms) + parms.bank.ITEM_USE[i]<=max_use)
    end

    return model
end

"""
    constraint_forms_overlap(model::Model, parms::Parameters, minItems::Int, maxItems::Int=minItems)

Constrains the number of overlapping items between test forms. The overlap between two forms must be between `minItems` and `maxItems`.
"""
function constraint_forms_overlap(
        model::Model, parms::Parameters, minItems::Int, maxItems::Int = minItems
)
    @assert(0<=minItems<=maxItems, "Error in forms_overlap: maxItems < minItems")

    if parms.shadow_test_size < 1 && parms.anchor_tests == 0
        x = model[:x]
        num_items, num_forms = size(x)

        @variable(model, z[1:num_items, 1:num_forms, 1:num_forms], Bin)
        items = collect(1:num_items)

        if minItems == maxItems
            @constraint(model,
                [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                sum(z[i, t1, t2] for i in items)==maxItems)
        else
            @constraint(model,
                [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                sum(z[i, t1, t2] for i in items)<=maxItems)
            @constraint(model,
                [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                sum(z[i, t1, t2] for i in items)>=minItems)
        end

        @constraint(model,
            [i = items, t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
            2 * z[i, t1, t2]<=x[i, t1] + x[i, t2])
        @constraint(model,
            [i = items, t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
            z[i, t1, t2]>=x[i, t1] + x[i, t2] - 1)
    end
end

"""
    constraint_forms_overlap2(
        model::Model, parms::Parameters, minItems::Int, maxItems::Int = minItems
    )

Constrains the number of overlapping items across test forms to be within specified bounds.
"""
function constraint_forms_overlap2(
        model::Model, parms::Parameters, minItems::Int, maxItems::Int = minItems
)
    @assert(0<=minItems<=maxItems, "Error in forms_overlap: maxItems < MinItems")

    x = model[:x]
    num_items, num_forms = size(x)

    # Create the binary variable `z` for item overlap tracking
    try
        @variable(model, z[1:num_items, 1:num_forms, 1:num_forms], Bin)
    catch e
        println(e)
    end

    z = model[:z]

    parms.max_item_use = max(2, parms.max_item_use)

    # Set up strict overlap constraints when shadow tests and anchors are not in use
    if parms.shadow_test_size < 1 && parms.anchor_tests == 0
        items = collect(1:num_items)

        # Apply overlap constraints
        if minItems == maxItems
            @constraint(model,
                [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                sum(z[i, t1, t2] for i in items)==maxItems)
        else
            @constraint(model,
                [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                sum(z[i, t1, t2] for i in items)<=maxItems)
            @constraint(model,
                [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                sum(z[i, t1, t2] for i in items)>=minItems)
        end

        # Ensure `z` matches the overlap between x[i, t1] and x[i, t2]
        @constraint(model,
            [i = items, t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
            2 * z[i, t1, t2]<=x[i, t1] + x[i, t2])
        @constraint(model,
            [i = items, t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
            z[i, t1, t2]>=x[i, t1] + x[i, t2] - 1)

        # Handle shadow tests by introducing auxiliary variables to represent overlap
    elseif parms.shadow_test_size > 0
        shadow_test_col = size(x, 2)

        # Define auxiliary binary variable `w` for overlap between `x[i, f]` and `x[i, shadow_test_col]`
        @variable(model, w[1:num_items, 1:num_forms], Bin)

        # Set up constraints for `w` to represent the overlap condition
        @constraint(model, [i = 1:num_items, f = 1:num_forms], w[i, f]<=x[i, f])
        @constraint(model, [i = 1:num_items, f = 1:num_forms],
            w[i, f]<=x[i, shadow_test_col])
        @constraint(model, [i = 1:num_items, f = 1:num_forms],
            w[i, f]>=x[i, f] + x[i, shadow_test_col] - 1)

        # Apply overlap constraints to `w` instead of the product of `x` terms
        if minItems == maxItems
            @constraint(model,
                [f = 1:num_forms],
                sum(w[i, f] for i in 1:num_items)==maxItems * parms.shadow_test_size)
        else
            @constraint(model,
                [f = 1:num_forms],
                sum(w[i, f] for i in 1:num_items)<=maxItems * parms.shadow_test_size)
            @constraint(model,
                [f = 1:num_forms],
                sum(w[i, f] for i in 1:num_items)>=minItems * parms.shadow_test_size)
        end
    end

    return model
end

# ---------------------------------------------------------------------------
# Objective Functions for Test Assembly Optimization
# ---------------------------------------------------------------------------

"""
    objective_match_characteristic_curve!(model::Model, parms::Parameters)

Constrains test characteristic curves to match a reference (tau matrix) within tolerance.
"""
function objective_match_characteristic_curve!(model::Model, parms::Parameters)
    R, K = 1:(parms.r), 1:(parms.k)
    P, tau = parms.score_matrix, parms.tau
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms  # Column for shadow test if applicable
    forms -= parms.shadow_test_size > 0 ? 1 : 0

    w = [1.1 - 0.1 * r for r in R]

    # Separate anchor and non-anchor items if anchor tests are used
    if parms.anchor_tests > 0
        anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), 1:items)
        non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), 1:items)

        # Calculate anchor contributions for (r, k) pairs
        anchor_contribution = zeros(Float64, length(R), length(K))
        for k in K, r in R
            anchor_contribution[r, k] = sum(P[i, k]^r for i in anchor_items)
        end
    else
        non_anchor_items = 1:items
        anchor_contribution = zeros(Float64, length(R), length(K))
    end

    # Constraints for operational forms
    @constraint(model, [f = 1:forms, k = K, r = R],
        sum(P[i, k]^r * x[i, f] for i in 1:items)<=tau[r, k] + w[r] * y)
    @constraint(model, [f = 1:forms, k = K, r = R],
        sum(P[i, k]^r * x[i, f] for i in 1:items)>=tau[r, k] - w[r] * y)

    # Constraints for shadow test
    if parms.shadow_test_size > 0
        shadow_test_size = parms.shadow_test_size

        @constraint(model, [k = K, r = R],
            sum(P[i, k]^r * x[i, zcol]
            for i in non_anchor_items)<=
            (tau[r, k] - anchor_contribution[r, k] + w[r] * y) * shadow_test_size)
        @constraint(model, [k = K, r = R],
            sum(P[i, k]^r * x[i, zcol]
            for i in non_anchor_items)>=
            (tau[r, k] - anchor_contribution[r, k] - w[r] * y) * shadow_test_size)
    end

    return model
end

"""
    objective_match_mean_var!(model::Model, parms::Parameters, α::Float64 = 1.0)

Matches expected score means and variances of test forms to reference values across theta points.
"""
function objective_match_mean_var!(model::Model, parms::Parameters, α::Float64 = 1.0)
    K = 1:(parms.k)  # Number of theta points
    tau_mean, tau_var = parms.tau_mean, parms.tau_var
    expected_scores = parms.score_matrix
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms  # Shadow test column if applicable
    forms -= parms.shadow_test_size > 0 ? 1 : 0

    # Separate anchor and non-anchor items if anchor tests are used
    if parms.anchor_tests > 0
        anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), 1:items)
        non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), 1:items)

        # Calculate anchor contributions to means and variances for each theta point
        anchor_mean_contribution = zeros(Float64, length(K))
        anchor_var_contribution = zeros(Float64, length(K))
        for k in K
            anchor_mean_contribution[k] = sum(expected_scores[i, k] for i in anchor_items)
            anchor_var_contribution[k] = sum((expected_scores[i, k] - tau_mean[k])^2
            for i in anchor_items)
        end
    else
        non_anchor_items = 1:items
        anchor_mean_contribution = zeros(Float64, length(K))
        anchor_var_contribution = zeros(Float64, length(K))
    end

    # Constraints for operational forms: match mean and variance
    @constraint(model, [f = 1:forms, k = K],
        sum(expected_scores[i, k] * x[i, f] for i in 1:items)<=tau_mean[k] + y)
    @constraint(model, [f = 1:forms, k = K],
        sum(expected_scores[i, k] * x[i, f] for i in 1:items)>=tau_mean[k] - y)

    @constraint(model, [f = 1:forms, k = K],
        sum(α * (expected_scores[i, k] - tau_mean[k])^2 * x[i, f]
        for i in 1:items)<=tau_var[k] + y)
    @constraint(model, [f = 1:forms, k = K],
        sum(α * (expected_scores[i, k] - tau_mean[k])^2 * x[i, f]
        for i in 1:items)>=tau_var[k] - y)

    # Constraints for shadow test if applicable
    if parms.shadow_test_size > 0
        shadow_test_size = parms.shadow_test_size

        for k in K
            effective_tau_mean = tau_mean[k] - anchor_mean_contribution[k]
            effective_tau_var = tau_var[k] - anchor_var_contribution[k]

            @constraint(model,
                sum(expected_scores[i, k] * x[i, zcol]
                for i in non_anchor_items)<=effective_tau_mean + y * shadow_test_size)
            @constraint(model,
                sum(expected_scores[i, k] * x[i, zcol]
                for i in non_anchor_items)>=effective_tau_mean - y * shadow_test_size)

            @constraint(model,
                sum(α * (expected_scores[i, k] - tau_mean[k])^2 * x[i, zcol]
                for i in non_anchor_items)<=effective_tau_var + y * shadow_test_size)
            @constraint(model,
                sum(α * (expected_scores[i, k] - tau_mean[k])^2 * x[i, zcol]
                for i in non_anchor_items)>=effective_tau_var - y * shadow_test_size)
        end
    end

    return model
end

"""
    objective_match_information_curve!(model::Model, parms::Parameters)

Constrains information curves to match target values at each theta point.
"""
function objective_match_information_curve!(model::Model, parms::Parameters)
    K, info, tau_info = parms.k, parms.info_matrix, parms.tau_info
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms
    forms -= parms.shadow_test_size > 0 ? 1 : 0

    # Separate anchor and non-anchor items if anchors are used
    if parms.anchor_tests > 0
        anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), 1:items)
        non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), 1:items)

        # Calculate anchor contributions for each theta point
        anchor_info_contribution = zeros(Float64, K)
        for k in 1:K
            anchor_info_contribution[k] = sum(info[i, k] for i in anchor_items)
        end
    else
        non_anchor_items = 1:items
        anchor_info_contribution = zeros(Float64, K)
    end

    # Constraints for operational forms
    @constraint(model, [f = 1:forms, k = 1:K],
        sum(info[i, k] * x[i, f] for i in 1:items)<=tau_info[k] + y)
    @constraint(model, [f = 1:forms, k = 1:K],
        sum(info[i, k] * x[i, f] for i in 1:items)>=tau_info[k] - y)

    # Shadow test constraints if applicable
    if parms.shadow_test_size > 0
        shadow_test_size = parms.shadow_test_size

        @constraint(model, [k = 1:K],
            sum(info[i, k] * x[i, zcol]
            for i in non_anchor_items)<=
            (tau_info[k] - anchor_info_contribution[k] + y) * shadow_test_size)
        @constraint(model, [k = 1:K],
            sum(info[i, k] * x[i, zcol]
            for i in non_anchor_items)>=
            (tau_info[k] - anchor_info_contribution[k] - y) * shadow_test_size)
    end

    return model
end

"""
    objective_info_relative2(model::Model, parms::Parameters)

Maximizes information at alternating theta points across forms, with each form meeting a weighted target.
"""
function objective_info_relative(model::Model, parms::Parameters)
    R = parms.relative_target_weights
    K = length(R)  # Number of theta points
    info = parms.info_matrix
    x, y = model[:x], model[:y]
    items, forms = size(x)

    # Separate anchor and non-anchor items if applicable
    if parms.anchor_tests > 0
        anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), 1:items)
        non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), 1:items)

        # Calculate anchor contributions
        anchor_info_contribution = zeros(Float64, K)
        for k in 1:K
            anchor_info_contribution[k] = sum(info[i, k] for i in anchor_items)
        end
    else
        non_anchor_items = 1:items
        anchor_info_contribution = zeros(Float64, K)
    end

    # Constraints for each form, rotating theta points (k) for max information at each
    k = 0
    for f in 1:forms
        k = k % K + 1  # Rotate over the theta points (K)

        # Information constraints at the current theta point
        @constraint(model,
            sum(info[i, k] * x[i, f]
            for i in 1:items)>=R[k] * y + anchor_info_contribution[k])
    end

    return model
end

"""
    objective_max_info(model::Model, parms::Parameters)

Maximizes information at each theta point, ensuring the sum of information meets a weighted target.
"""
function objective_max_info(model::Model, parms::Parameters)
    R = parms.relative_target_weights
    K = parms.k
    info = parms.info_matrix
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms
    forms -= parms.shadow_test_size > 0 ? 1 : 0

    # Separate anchor and non-anchor items if anchor tests are included
    if parms.anchor_tests > 0
        anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), 1:items)
        non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), 1:items)

        # Calculate anchor contributions for each theta point
        anchor_info_contribution = zeros(Float64, K)
        for k in 1:K
            anchor_info_contribution[k] = sum(info[i, k] for i in anchor_items)
        end
    else
        non_anchor_items = 1:items
        anchor_info_contribution = zeros(Float64, K)
    end

    # Constraints for operational forms
    @constraint(model, [f = 1:forms, k = 1:K],
        sum(info[i, k] * x[i, f] for i in 1:items)>=R[k] * y)

    # Constraints for shadow test if applicable
    if parms.shadow_test_size > 0
        shadow_test_size = parms.shadow_test_size

        @constraint(model, [k = 1:K],
            sum(info[i, k] * x[i, zcol]
            for i in non_anchor_items)>=
            (R[k] * y * shadow_test_size - anchor_info_contribution[k]))
    end

    return model
end

end  # module ATAConstraints
