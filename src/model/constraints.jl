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

using DataFrames
using JuMP
using ..Configuration
using ..Utils

# ---------------------------------------------------------------------------
# Note: Anchor Tests are designed to appear repeatedly, cycling between forms. This behaviour
# imposes the need of a special treatment, different to the one used with regular items. In order
# to use the Shadow Test method, our solution was to calculate the contribution of these special
# items to the target variables and use the result as a proxy for other anchor items in subsequent forms.
# This special treatment has made the code more complex in most cases.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

"""
    operational_forms(x::AbstractMatrix, shadow_test_size::Int)::Int

Calculate the number of operational forms in the decision variable matrix `x`, adjusting for shadow tests.

# Arguments

  - `x`: The decision variable matrix where rows correspond to items and columns to forms.
  - `shadow_test_size`: The number of shadow tests included. If greater than zero, the last column(s) are shadow tests.

# Returns

  - The number of operational (non-shadow) forms.
"""
function operational_forms(x, shadow_test_size::Int)
    forms = size(x, 2)
    return shadow_test_size > 0 ? forms - 1 : forms
end

"""
    group_by_selected(selected::Union{Vector{T}, BitVector}) where {T}

Group items based on the `selected` vector, ignoring missing values.

# Arguments

  - `selected`: A vector indicating group assignments or selection criteria for items.

# Returns

  - A `GroupedDataFrame` where each group corresponds to a unique value in `selected`.

# Notes

  - Missing values and items not satisfying the selection criteria are excluded.
"""
function group_by_selected(selected::Union{Vector, BitVector})
    data = DataFrame(; selected = selected, index = 1:length(selected))
    if isa(selected, BitVector)
        data = data[data.selected, :]
    elseif eltype(data.selected) <: AbstractString
        data = data[data.selected .!== missing, :]
    elseif eltype(data.selected) <: Number
        data = data[(!ismissing).(data.selected) .| (data.selected .> 0), :]
    end
    return groupby(data, :selected; skipmissing = true)
end

# ---------------------------------------------------------------------------
# Constraint Functions for item counts
# ---------------------------------------------------------------------------

"""
    constraint_items_per_form(
        model::Model,
        parms::Parameters,
        minItems::Int,
        maxItems::Int = minItems
    )::Model

Ensure each operational form contains between `minItems` and `maxItems` items.

# Arguments

  - `model`: The JuMP model to which the constraints are added.
  - `parms`: Parameters containing the item bank and settings.
  - `minItems`: Minimum number of items required in each form.
  - `maxItems`: Maximum number of items allowed in each form (defaults to `minItems`).

# Returns

  - The updated `Model` with the new constraints.
"""
function constraint_items_per_form(
        model::Model, parms::Parameters, minItems::Int, maxItems::Int = minItems
)
    return constraint_item_count(
        model, parms, trues(size(parms.bank, 1)), minItems, maxItems
    )
end

"""
    constraint_item_count(
        model::Model,
        parms::Parameters,
        selected::BitVector,
        minItems::Int,
        maxItems::Int = minItems
    )::Model

Ensure the number of selected items falls within specified bounds in each form.

# Arguments

  - `model`: The JuMP model to which the constraints are added.
  - `parms`: Parameters containing the item bank and settings.
  - `selected`: Boolean vector indicating which items to include.
  - `minItems`: Minimum number of selected items required in each form.
  - `maxItems`: Maximum number of selected items allowed in each form (defaults to `minItems`).

# Returns

  - The updated `Model` with the new constraints.
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
    constraint_score_sum(
        model::Model,
        parms::Parameters,
        selected::BitVector,
        minScore::Int64,
        maxScore::Int64 = minScore
    )::Model

Ensure the sum of selected item scores is within `minScore` and `maxScore` for each form.

# Arguments

  - `model`: The JuMP model to which the constraints are added.
  - `parms`: Parameters containing the item bank and settings.
  - `selected`: Boolean vector indicating which items to include.
  - `minScore`: Minimum total score required for the selected items.
  - `maxScore`: Maximum total score allowed (defaults to `minScore`).

# Returns

  - The updated `Model` with the new constraints.
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
    constraint_item_sum(
        model::Model,
        parms::Parameters,
        vals,
        minVal::Real,
        maxVal::Real = minVal
    )::Model

Ensure the sum of item values is within specified bounds for each form.

# Arguments

  - `model`: The JuMP model to which the constraints are added.
  - `parms`: Parameters containing the item bank and settings.
  - `vals`: A vector or matrix of item values (and optionally conditions).
  - `minVal`: Minimum total value required.
  - `maxVal`: Maximum total value allowed (defaults to `minVal`).

# Returns

  - The updated `Model` with the new constraints.
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
    constraint_friends_in_form(
        model::Model,
        parms::Parameters,
        selected::Union{Vector, BitVector}
    )::Model

Ensure that friend items are assigned to the same form.

# Arguments

  - `model`: The JuMP model to which the constraints are added.
  - `parms`: Parameters containing the item bank and settings.
  - `selected`: Vector indicating groups of friend items.

# Returns

  - The updated `Model` with the new constraints.
"""
function constraint_friends_in_form(model::Model, parms::Parameters, selected)
    x = model[:x]
    forms = operational_forms(x, parms.shadow_test_size)
    groups = group_by_selected(selected)

    for group in groups
        items = group[!, :index]
        pivot = items[1]  # Reference item
        cnt = length(items)

        if cnt > 1
            @constraint(model, [f in 1:forms],
                sum(x[i, f] for i in items)==(cnt * x[pivot, f]))
        end
    end
end

"""
    constraint_enemies(
        model::Model,
        parms::Parameters,
        selected::Union{Vector, BitVector}
    )::Model

Ensure that only one enemy item is assigned to the same form.

# Arguments

  - `model`: The JuMP model to which the constraints are added.
  - `parms`: Parameters containing the item bank and settings.
  - `selected`: Vector indicating groups of enemy items.

# Returns

  - The updated `Model` with the new constraints.
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
    constraint_exclude_items(
        model::Model,
        exclude::BitVector
    )::Model

Exclude specified items from being selected in any form.

# Arguments

  - `model`: The JuMP model to which the constraints are added.
  - `exclude`: Boolean vector indicating items to exclude (`true` to exclude).

# Returns

  - The updated `Model` with the items fixed to 0.
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
    constraint_fix_items(
        model::Model,
        fixed::BitVector
    )::Model

Force certain items to be included in every form.

# Arguments

  - `model`: The JuMP model to which the constraints are added.
  - `fixed`: Boolean vector indicating items to include (`true` to include).

# Returns

  - The updated `Model` with the items fixed to 1.
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
    constraint_add_anchor!(
        model::Model,
        parms::Parameters
    )::Model

Ensure anchor items are included in all operational forms, ignoring shadow forms.

# Arguments

  - `model`: The JuMP model to which the constraints are added.
  - `parms`: Parameters containing the item bank and settings.

# Returns

  - The updated `Model` with anchor items fixed to 1.
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
    constraint_max_use(
        model::Model,
        parms::Parameters,
        selected::BitVector
    )::Model

Constrain the maximum number of times an item can appear across the test forms.

# Arguments

  - `model`: The JuMP model to which the constraints are added.
  - `parms`: Parameters containing the item bank and settings.
  - `selected`: Boolean vector indicating items subject to the constraint.

# Returns

  - The updated `Model` with the new constraints.
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
    constraint_forms_overlap(
        model::Model,
        parms::Parameters,
        minItems::Int,
        maxItems::Int = minItems
    )::Model

Constrain the number of overlapping (repeated) items between test forms.

# Arguments

  - `model`: The JuMP model to which the constraints are added.
  - `parms`: Parameters containing the item bank and settings.
  - `minItems`: Minimum number of items that must overlap between any two forms.
  - `maxItems`: Maximum number of items that can overlap (defaults to `minItems`).

# Returns

  - The updated `Model` with the new constraints.
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
        model::Model,
        parms::Parameters,
        minItems::Int,
        maxItems::Int = minItems
    )::Model

Constrain the number of overlapping items across test forms to be within specified bounds.
The use of this constraint uses more resources, as it implies a much larger problem.

# Arguments

  - `model`: The JuMP model to which the constraints are added.
  - `parms`: Parameters containing the item bank and settings.
  - `minItems`: Minimum number of items that must overlap between any two forms.
  - `maxItems`: Maximum number of items that can overlap (defaults to `minItems`).

# Returns

  - The updated `Model` with the new constraints.
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
                overlap[t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                sum(z[i, t1, t2] for i in items)==maxItems)
        else
            @constraint(model,
                overlap_upper[t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                sum(z[i, t1, t2] for i in items)<=maxItems)
            @constraint(model,
                overlap_lower[t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                sum(z[i, t1, t2] for i in items)>=minItems)
        end

        # Ensure `z` matches the overlap between x[i, t1] and x[i, t2]
        @constraint(model,
            [i = items, t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
            2 * z[i, t1, t2]<=x[i, t1] + x[i, t2])
        @constraint(model,
            [i = items, t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
            z[i, t1, t2]>=x[i, t1] + x[i, t2] - 1)

    elseif parms.shadow_test_size > 0
        # Handle shadow tests by introducing auxiliary variables
        shadow_test_col = size(x, 2)

        # Define auxiliary binary variable `w` for overlap
        @variable(model, w[1:num_items, 1:num_forms], Bin)

        # Set up constraints for `w`
        @constraint(model, [i = 1:num_items, f = 1:num_forms], w[i, f]<=x[i, f])
        @constraint(model, [i = 1:num_items, f = 1:num_forms],
            w[i, f]<=x[i, shadow_test_col])
        @constraint(model, [i = 1:num_items, f = 1:num_forms],
            w[i, f]>=x[i, f] + x[i, shadow_test_col] - 1)

        # Apply overlap constraints to `w`
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
    objective_match_characteristic_curve!(
        model::Model,
        parms::Parameters
    )::Model

Match test characteristic curves to target values. Following the suggestion
of Wim van der Linden, the curves are also compared with their porwers 1..R.
for a closer match.
(in the book'Linear models for Optimal Test Desting')

# Arguments

  - `model`: The JuMP model containing decision variables `x` and `y`.
  - `parms`: Parameters containing target curves and item probabilities.

# Returns

  - The updated `Model` with the new constraints.
"""
function objective_match_characteristic_curve!(model::Model, parms::Parameters)
    @assert haskey(model, :x)&&haskey(model, :y) "Model must contain variables x and y"

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
    @constraint(model, tcc_upper[f = 1:forms, k = K, r = R],
        sum(P[i, k]^r * x[i, f] for i in 1:items)<=tau[r, k] + w[r] * y)
    @constraint(model, tcc_lower[f = 1:forms, k = K, r = R],
        sum(P[i, k]^r * x[i, f] for i in 1:items)>=tau[r, k] - w[r] * y)

    # Constraints for shadow test
    if parms.shadow_test_size > 0
        shadow_test_size = parms.shadow_test_size

        @constraint(model, shadow_tcc_upper[k = K, r = R],
            sum(P[i, k]^r * x[i, zcol]
            for i in non_anchor_items)<=
            (tau[r, k] - anchor_contribution[r, k] + w[r] * y) * shadow_test_size)
        @constraint(model, shadow_tcc_lower[k = K, r = R],
            sum(P[i, k]^r * x[i, zcol]
            for i in non_anchor_items)>=
            (tau[r, k] - anchor_contribution[r, k] - w[r] * y) * shadow_test_size)
    end

    return model
end

"""
    objective_match_mean_var!(
        model::Model,
        parms::Parameters,
        α::Float64 = 3.0
    )::Model

Match expected score means and variances of test forms to reference values at selected theta points.

# Arguments

  - `model`: The JuMP model containing decision variables `x` and `y`.
  - `parms`: Parameters containing expected scores and variances.
  - `α`: Weight factor for variance matching (default: 3.0). Compensates for difference
    in the scale of the variables (mean and variance).

# Returns

  - The updated `Model` with the new constraints.
"""
function objective_match_mean_var!(model::Model, parms::Parameters, α::Float64 = 10.0)
    # Validate inputs
    @assert α>0 "Weight factor α must be positive"
    @assert haskey(model, :x)&&haskey(model, :y) "Model must contain variables x and y"

    # Extract parameters
    K = 1:(parms.k)  # Theta points

    # Mean of expected total score for the reference test form and
    # Variance of the individual items  the reference test form and
    tau_mean, item_score_means = parms.tau_mean, parms.item_score_means
    expected_scores = parms.score_matrix
    x, y = model[:x], model[:y]
    num_items, num_forms = size(x)

    # Handle shadow test configuration
    has_shadow_test = parms.shadow_test_size > 0
    if has_shadow_test
        shadow_test_col = num_forms
        num_forms -= 1
    end

    # Process anchor items
    if parms.anchor_tests > 0
        anchor_items = [i for i in 1:num_items if !ismissing(parms.bank.ANCHOR[i])]
        non_anchor_items = [i for i in 1:num_items if ismissing(parms.bank.ANCHOR[i])]

        # Pre-compute anchor contributions
        anchor_mean_contribution = [sum(expected_scores[i, k] for i in anchor_items)
                                    for k in K]
        anchor_var_contribution = [sum((expected_scores[i, k] - item_score_means[k])^2
                                   for i in anchor_items) for k in K]
    else
        non_anchor_items = 1:num_items
        anchor_mean_contribution = zeros(Float64, length(K))
        anchor_var_contribution = zeros(Float64, length(K))
    end

    # Adjust item_score_means for scale consistency with tau_mean
    # scale_factor = maximum(tau_mean) / (maximum(item_score_means) + eps())  # Avoid division by zero
    # item_score_means_scaled = item_score_means * scale_factor

    # Operational forms constraints
    @constraint(model, mean_upper[f = 1:num_forms, k = K],
        sum(expected_scores[i, k] * x[i, f] for i in 1:num_items)<=tau_mean[k] + y)
    @constraint(model, mean_lower[f = 1:num_forms, k = K],
        sum(expected_scores[i, k] * x[i, f] for i in 1:num_items)>=tau_mean[k] - y)

    @constraint(model, var_upper[f = 1:num_forms, k = K],
        sum(α * (expected_scores[i, k] - item_score_means[k])^2 * x[i, f]
        for i in 1:num_items)<=item_score_means[k] + y)
    @constraint(model, var_lower[f = 1:num_forms, k = K],
        sum(α * (expected_scores[i, k] - item_score_means[k])^2 * x[i, f]
        for i in 1:num_items)>=item_score_means[k] - y)

    # Shadow test constraints (if applicable)
    if has_shadow_test
        shadow_test_size = parms.shadow_test_size
        effective_tau_mean = [tau_mean[k] - anchor_mean_contribution[k] for k in K]
        effective_item_score_means = [item_score_means[k] - anchor_var_contribution[k]
                                      for k in K]

        @constraint(model, shadow_mean_upper[k = K],
            sum(expected_scores[i, k] * x[i, shadow_test_col]
            for i in non_anchor_items)<=
            effective_tau_mean[k] + shadow_test_size * y)
        @constraint(model, shadow_mean_lower[k = K],
            sum(expected_scores[i, k] * x[i, shadow_test_col]
            for i in non_anchor_items)>=
            effective_tau_mean[k] - shadow_test_size * y)

        @constraint(model, shadow_var_upper[k = K],
            sum(α * (expected_scores[i, k] - item_score_means[k])^2 * x[i, shadow_test_col]
            for i in non_anchor_items)<=
            effective_item_score_means[k] + shadow_test_size * y)
        @constraint(model, shadow_var_lower[k = K],
            sum(α * (expected_scores[i, k] - item_score_means[k])^2 * x[i, shadow_test_col]
            for i in non_anchor_items)>=
            effective_item_score_means[k] - shadow_test_size * y)
    end

    return model
end

"""
    objective_match_information_curve!(
        model::Model,
        parms::Parameters
    )::Model

Match information curves to target values at each theta point.

# Arguments

  - `model`: The JuMP model containing decision variables `x` and `y`.
  - `parms`: Parameters containing item information and target values.

# Returns

  - The updated `Model` with the new constraints.
"""
function objective_match_information_curve!(model::Model, parms::Parameters)
    K, info, tau_info = parms.k, parms.info_matrix, parms.tau_info
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms
    forms -= parms.shadow_test_size > 0 ? 1 : 0
    if parms.method == "MIXED"
        α = 1.0
    else
        α = 0.5
    end

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
    @constraint(model, info_upper[f = 1:forms, k = 1:K],
        sum(info[i, k] * x[i, f] for i in 1:items)<=tau_info[k] + y * α)
    @constraint(model, info_lower[f = 1:forms, k = 1:K],
        sum(info[i, k] * x[i, f] for i in 1:items)>=tau_info[k] - y * α)

    # Shadow test constraints if applicable
    if parms.shadow_test_size > 0
        shadow_test_size = parms.shadow_test_size

        @constraint(model, shadow_info_upper[k = 1:K],
            sum(info[i, k] * x[i, zcol]
            for i in non_anchor_items)<=
            (tau_info[k] - anchor_info_contribution[k] + y * α) * shadow_test_size)
        @constraint(model, shadow_info_lower[k = 1:K],
            sum(info[i, k] * x[i, zcol]
            for i in non_anchor_items)>=
            (tau_info[k] - anchor_info_contribution[k] - y * α) * shadow_test_size)
    end

    return model
end

"""
    objective_info_relative2(
        model::Model,
        parms::Parameters
    )::Model

Maximize information at alternating theta points across forms.

# Arguments

  - `model`: The JuMP model containing decision variables `x` and `y`.
  - `parms`: Parameters containing weights and item information.

# Returns

  - The updated `Model` with the new constraints.
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
    objective_max_info(
        model::Model,
        parms::Parameters
    )::Model

Maximize information at each theta point, meeting a weighted target.

# Arguments

  - `model`: The JuMP model containing decision variables `x` and `y`.
  - `parms`: Parameters containing weights and item information.

# Returns

  - The updated `Model` with the new constraints.
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
    @constraint(model, max_info[f = 1:forms, k = 1:K],
        sum(info[i, k] * x[i, f] for i in 1:items)>=R[k] * y)

    # Constraints for shadow test if applicable
    if parms.shadow_test_size > 0
        shadow_test_size = parms.shadow_test_size

        @constraint(model, shadow_max_info[k = 1:K],
            sum(info[i, k] * x[i, zcol]
            for i in non_anchor_items)>=
            (R[k] * y * shadow_test_size - anchor_info_contribution[k]))
    end

    return model
end

end  # module Constraints
