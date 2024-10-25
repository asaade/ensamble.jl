module Constraints

# Export all functions for external use
export constraint_items_per_form, constraint_item_count,
    constraint_item_sum, group_by_selected, constraint_friends_in_form,
    constraint_enemies_in_form, constraint_exclude_items, constraint_include_items,
    constraint_add_anchor!, constraint_max_use, constraint_forms_overlap,
    objective_match_characteristic_curve!, objective_match_information_curve!,
    objective_max_info, objective_info_relative2, constraint_fix_items

#

using DataFrames
using JuMP

using ..Configuration

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

"""
    operational_forms(x, shadow_test_size) -> Int

Helper function to calculate the number of operational forms, adjusting for shadow tests.
"""
function operational_forms(x, shadow_test_size)
    forms = size(x, 2)
    return shadow_test_size > 0 ? forms - 1 : forms
end

"""
    group_by_selected(selected::Vector) -> GroupedDataFrame

Helper function to group items based on the `selected` vector, ignoring missing values.
"""
function group_by_selected(selected::Union{Vector,BitVector})
    data = DataFrame(; selected=selected, index=1:length(selected))
    if isa(selected, BitVector)
        data = data[data.selected, :]
    elseif eltype(data.selected) <: AbstractString
        data = data[data.selected!==missing, :]
    elseif eltype(data.selected) <: Number
        data = data[data.selected!==missing||data.selected.>0, :]
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

Constrains the number of selected items, ensuring it is within the specified range for both operational forms and the shadow test (if applicable).
"""
function constraint_item_count(model::Model, parms::Parameters, selected::BitVector,
    minItems::Int, maxItems::Int=minItems)
    @assert(minItems <= maxItems, "Error in item_count: maxItems < MinItems")

    x = model[:x]
    items = collect(1:size(x, 1))[selected]
    forms = operational_forms(x, parms.shadow_test_size)

    # Constraints for item count in operational forms
    @constraint(model, [f=1:forms], sum(x[items, f]) >= minItems)
    @constraint(model, [f=1:forms], sum(x[items, f]) <= maxItems)

    # Handle shadow test constraints if applicable and if anchor test is in use
    if parms.shadow_test_size > 0
        shadow_test_col = size(x, 2)

        # Check if an anchor test is selected
        if parms.anchor_tests > 0
            # Separate anchor and non-anchor items
            non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), items)
            anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), items)

            # Adjust item count bounds by excluding anchor items
            adjusted_minItems = max(minItems - length(anchor_items), 0)
            adjusted_maxItems = max(maxItems - length(anchor_items), 0)

            # Constraints for shadow test item count (non-anchor items only)
            @constraint(model,
                sum(x[non_anchor_items, shadow_test_col]) >= adjusted_minItems * parms.shadow_test_size)
            @constraint(model,
                sum(x[non_anchor_items, shadow_test_col]) <= adjusted_maxItems * parms.shadow_test_size)
        else
            # No anchor test: apply minItems and maxItems constraints to all selected items
            @constraint(model,
                sum(x[items, shadow_test_col]) >= minItems * parms.shadow_test_size)
            @constraint(model,
                sum(x[items, shadow_test_col]) <= maxItems * parms.shadow_test_size)
        end
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
    forms = operational_forms(x, parms.shadow_test_size)

    # Handle cases for single-column and two-column `vals` input
    if ndims(vals) == 1
        val = vals
        cond = trues(length(vals))  # Default to including all items if no condition provided
    else
        val = vals[:, 2]
        cond = vals[:, 1]
    end

    items = collect(1:size(x, 1))[cond]

        # Sum constraints for operational forms (include all items
    @constraint(model, [f=1:forms],
        sum(x[i, f] * val[i] for i in items) >= minVal)
    @constraint(model, [f=1:forms],
        sum(x[i, f] * val[i] for i in items) <= maxVal)

    # Separate items into anchor and non-anchor groups if anchor tests are enabled
    if parms.anchor_tests > 0
        anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), items)
        non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), items)

        # Calculate the sum for anchor items in operational forms
        anchor_val_sum = sum(val[i] for i in anchor_items)

        # Adjust minVal and maxVal based on anchor items' contribution
        effective_minVal = max(0, minVal - anchor_val_sum)
        effective_maxVal = max(0, maxVal - anchor_val_sum)
    else
        # No anchor test: treat all items as regular items
        anchor_items = Int[]   # Empty list of anchor items
        non_anchor_items = items  # All items are non-anchor
        effective_minVal, effective_maxVal = minVal, maxVal
    end

    # Shadow test constraints (only for non-anchor items if anchors are enabled)
    if parms.shadow_test_size > 0
        shadow_test_col = size(x, 2)

        @constraint(model,
            sum(x[i, shadow_test_col] * val[i] for i in non_anchor_items) >=
            effective_minVal * parms.shadow_test_size)
        @constraint(model,
            sum(x[i, shadow_test_col] * val[i] for i in non_anchor_items) <=
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
                sum(x[i, f] for i in items) == (cnt * x[pivot, f]))
        end
    end
end

"""
    constraint_enemies_in_form(model::Model, parms::Parameters, selected)

Adds constraints to prevent enemy items (items that should not appear together) from being assigned to the same form.
"""
function constraint_enemies_in_form(model::Model, parms::Parameters, selected)
    x = model[:x]
    forms = operational_forms(x, parms.shadow_test_size)

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
            JuMP.fix(x[i, f], 0; force=true)
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
        forms = operational_forms(x, parms.shadow_test_size)
        anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), 1:size(x, 1))

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
    forms = operational_forms(x, parms.shadow_test_size)

    if max_use < forms
        selected_items = collect(1:size(x, 1))[selected]
        non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), selected_items)

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

    if parms.shadow_test_size < 1 && parms.anchor_tests == 0
        x = model[:x]
        num_items, num_forms = size(x)

        @variable(model, z[1:num_items, 1:num_forms, 1:num_forms], Bin)
        items = collect(1:num_items)
        #items = items[parms.bank.ANCHOR .=== missing]

        if minItems == maxItems
            @constraint(model, [t1 = 1:(num_forms-1), t2 = (t1+1):num_forms],
                sum(z[i, t1, t2] for i in items) == maxItems)
        else
            @constraint(model, [t1 = 1:(num_forms-1), t2 = (t1+1):num_forms],
                sum(z[i, t1, t2] for i in items) <= maxItems)
            @constraint(model, [t1 = 1:(num_forms-1), t2 = (t1+1):num_forms],
                sum(z[i, t1, t2] for i in items) >= minItems)
        end

        @constraint(model, [i = items, t1 = 1:(num_forms-1), t2 = (t1+1):num_forms], 2 * z[i, t1, t2] <= x[i, t1] + x[i, t2])
        @constraint(model, [i = items, t1 = 1:(num_forms-1), t2 = (t1+1):num_forms], z[i, t1, t2] >= x[i, t1] + x[i, t2] - 1)
    end
end

# ---------------------------------------------------------------------------
# Objective Functions for Test Assembly Optimization
# ---------------------------------------------------------------------------

"""
    objective_match_characteristic_curve!(model::Model, parms::Parameters)

Constrains the characteristic curve by matching the sum of the probability of correct response for each form at theta points `k`.
"""
function objective_match_characteristic_curve!(model::Model, parms::Parameters)
    R, K = 1:(parms.r), 1:(parms.k)
    P, tau = parms.p_matrix, parms.tau
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms
    forms -= parms.shadow_test_size > 0 ? 1 : 0

    # Weighting factors for each power r
    w = [1.1 - (0.1 * r) for r in R]

    # Constraints for operational forms (include both anchor and non-anchor items)
    @constraint(model, [f = 1:forms, k = K, r = R],
        sum(P[i, k]^r * x[i, f] for i in 1:items) <= tau[r, k] + w[r] * y)
    @constraint(model, [f = 1:forms, k = K, r = R],
        sum(P[i, k]^r * x[i, f] for i in 1:items) >= tau[r, k] - w[r] * y)

    # Handle shadow test constraints if applicable
    if parms.shadow_test_size > 0
        shadow_test_size = parms.shadow_test_size

        if parms.anchor_tests > 0
            # Separate anchor and non-anchor items
            anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), 1:items)
            non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), 1:items)

            # Initialize and calculate anchor contributions for each (r, k) pair
            anchor_contribution = zeros(Float64, length(R), length(K))
            for k in K, r in R
                anchor_contribution[r, k] = sum(P[i, k]^r for i in anchor_items)
            end

            # Shadow test constraints with anchor contribution subtracted
            @constraint(model, [k = K, r = R],
                sum(P[i, k]^r * x[i, zcol] for i in non_anchor_items) <=
                    (tau[r, k] - anchor_contribution[r, k] + w[r] * y) * shadow_test_size)
            @constraint(model, [k = K, r = R],
                sum(P[i, k]^r * x[i, zcol] for i in non_anchor_items) >=
                    (tau[r, k] - anchor_contribution[r, k] - w[r] * y) * shadow_test_size)
        else
            # No anchors: shadow test constraints without anchor adjustment
            @constraint(model, [k = K, r = R],
                sum(P[i, k]^r * x[i, zcol] for i in 1:items) <=
                    (tau[r, k] + w[r] * y) * shadow_test_size)
            @constraint(model, [k = K, r = R],
                sum(P[i, k]^r * x[i, zcol] for i in 1:items) >=
                    (tau[r, k] - w[r] * y) * shadow_test_size)
        end
    end

    return model
end


"""
    objective_match_mean_var!(model::Model, parms::Parameters, α::Float64 = 1.0)

Adds constraints to the MIP model to ensure that the test forms' expected score means and variances match those of the reference form at each theta point, using a single weighted slack variable.

# Arguments

- `model::Model`: The optimization model to which the constraints will be added.
- `parms::Parameters`: A struct or dictionary containing the following fields:
    - `tau_mean::Vector{Float64}`: The expected score means for the reference form at each theta point (vector of length `K`).
    - `tau_var::Vector{Float64}`: The expected score variances for the reference form at each theta point (vector of length `K`).
    - `expected_score_matrix::Matrix{Float64}`: The matrix of expected scores for each item at each theta point (items x theta points).
    - `x::VariableRef`: The binary decision variable matrix indicating item selection for each form (items x forms).
    - `y::VariableRef`: The slack variable for deviations in both mean and variance constraints (scalar or vector).
- `α::Float64`: Weighting factor for the variance constraint (default is 1.0, meaning equal importance for mean and variance).

# Behavior

This function adds constraints to match the means and variances of expected scores between the test forms and the reference form, using a single slack variable to balance both constraints.

## Constraints:
1. **Mean of Expected Scores**: Ensures that the sum of expected scores for each test form at each theta point matches the corresponding expected score from the reference form.
    - ( sum_{i=1}^{N} E[X_i(    heta_k)] x_i leq    au_{    ext{mean}}(     heta_k) + y )
    - ( sum_{i=1}^{N} E[X_i(    heta_k)] x_i geq    au_{    ext{mean}}(     heta_k) - y )

2. **Variance of Expected Scores**: Ensures that the variance of expected scores for each test form at each theta point matches that of the reference form, scaled by the weight ( lpha ).
    - ( sum_{i=1}^{N} lpha (E[X_i(    heta_k)] -      au_{    ext{mean}}(     heta_k))^2 x_i leq      au_{    ext{var}}(  heta_k) + y )
    - ( sum_{i=1}^{N} lpha (E[X_i(    heta_k)] -      au_{    ext{mean}}(     heta_k))^2 x_i geq      au_{    ext{var}}(  heta_k) - y )

The constraints are added for all test forms and all theta points.

# Returns
- `model::Model`: The updated model with the new constraints added.
"""
function objective_match_mean_var!(model::Model, parms::Parameters, α::Float64=1.0)
    K = 1:(parms.k)  # Number of theta points
    tau_mean, tau_var = parms.tau_mean, parms.tau_var  # Expected score means and variances for the reference form
    expected_scores = parms.expected_score_matrix  # Matrix of expected scores (items x theta points)
    x, y = model[:x], model[:y]  # Decision variables
    items, forms = size(x)  # Number of items and forms

    # Constraints for matching the mean of expected scores
    @constraint(model, [f = 1:forms, k = K],
        sum(expected_scores[i, k] * x[i, f] for i in 1:items) <= tau_mean[k] + y)

    @constraint(model, [f = 1:forms, k = K],
        sum(expected_scores[i, k] * x[i, f] for i in 1:items) >= tau_mean[k] - y)

    # Constraints for matching the variance of expected scores, scaled by α
    @constraint(model, [f = 1:forms, k = K],
        sum(α * (expected_scores[i, k] - tau_mean[k])^2 * x[i, f] for i in 1:items) <= tau_var[k] + y)

    @constraint(model, [f = 1:forms, k = K],
        sum(α * (expected_scores[i, k] - tau_mean[k])^2 * x[i, f] for i in 1:items) >= tau_var[k] - y)

    return model
end




"""
    objective_max_info(model::Model, parms::Parameters)

Maximizes information at each point `k`, ensuring that the sum of `info[i, k] * x[i, f]` meets or exceeds the weighted target.
"""
function objective_match_information_curve!(model::Model, parms::Parameters)
    K, info, tau_info = parms.k, parms.info_matrix, parms.tau_info
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms
    forms -= parms.shadow_test_size > 0 ? 1 : 0

    α = parms.method == "MIXED" ? 1.3 : 1.0

    # Constraints for information curve in operational forms (include anchor and non-anchor items)
    @constraint(model, [f = 1:forms, k = 1:K],
        sum(info[i, k] * x[i, f] for i in 1:items) <= tau_info[k] + (α * y))
    @constraint(model, [f = 1:forms, k = 1:K],
        sum(info[i, k] * x[i, f] for i in 1:items) >= tau_info[k] - (α * y))

    # Handle shadow test constraints if applicable
    if parms.shadow_test_size > 0
        shadow_test_size = parms.shadow_test_size

        if parms.anchor_tests > 0
            # Separate anchor and non-anchor items
            anchor_items = filter(i -> !ismissing(parms.bank.ANCHOR[i]), 1:items)
            non_anchor_items = filter(i -> ismissing(parms.bank.ANCHOR[i]), 1:items)

            # Calculate anchor contribution for each theta point k
            anchor_info_contribution = zeros(Float64, K)
            for k in 1:K
                anchor_info_contribution[k] = sum(info[i, k] for i in anchor_items)
            end

            # Shadow test constraints with anchor contribution subtracted
            @constraint(model, [k = 1:K],
                sum(info[i, k] * x[i, zcol] for i in non_anchor_items) <=
                    (tau_info[k] - anchor_info_contribution[k] + (α * y)) * shadow_test_size)
            @constraint(model, [k = 1:K],
                sum(info[i, k] * x[i, zcol] for i in non_anchor_items) >=
                    (tau_info[k] - anchor_info_contribution[k] - (α * y)) * shadow_test_size)
        else
            # No anchors: appl(α * y) constraints to all items directl(α * y)
            @constraint(model, [k = 1:K],
                sum(info[i, k] * x[i, zcol] for i in 1:items) <=
                    (tau_info[k] + (α * y)) * shadow_test_size)
            @constraint(model, [k = 1:K],
                sum(info[i, k] * x[i, zcol] for i in 1:items) >=
                    (tau_info[k] - (α * y)) * shadow_test_size)
        end
    end

    return model
end

"""
    objective_info_relative2(model::Model, parms::Parameters)

Maximizes information at alternating points `k` across forms, ensuring that each form meets or exceeds the weighted target.
"""
function objective_info_relative2(model::Model, parms::Parameters)
    R = parms.relative_target_weights  # Vector of weights for each theta point k
    K = length(R)                      # Number of theta points
    info = parms.info_matrix            # Item information matrix (Items x K)
    x, y = model[:x], model[:y]         # Decision variables and slack variable
    _, forms = size(x)

    # Define a tolerance for the information consistency across forms sharing the same k
    # tolerance = 0.05  # Allow a small tolerance for information consistency

    # Group forms by their associated k points
    form_groups = Dict(k => [] for k in 1:K)
    for f in 1:forms
        k = (f - 1) % K + 1  # Rotate over theta points
        push!(form_groups[k], f)
    end

    # Maximize information at the specific theta point k for each form
    for f in 1:forms
        k = (f - 1) % K + 1  # Rotate over theta points

        # Maximize the information at theta point k for form f
        @constraint(model, info[:, k]' * x[:, f] >= R[k] * y)
    end

    # # Enforce consistency: Ensure that forms sharing the same k point have similar information
    # for k in 1:K
    #     # Get the forms that share the same k point
    #     forms_for_k = form_groups[k]

    #     # Ensure the information levels are consistent (within tolerance) across these forms
    #     for i in 1:(length(forms_for_k) - 1)
    #         f1 = forms_for_k[i]
    #         f2 = forms_for_k[i + 1]

    #         # Constrain the information of form f1 and form f2 to be within the tolerance at point k
    #         @constraint(model, info[:, k]' * x[:, f1] - info[:, k]' * x[:, f2] <= tolerance)
    #         @constraint(model, info[:, k]' * x[:, f1] - info[:, k]' * x[:, f2] >= tolerance)
    #     end
    # end

    return model
end



end  # module ATAConstraints
