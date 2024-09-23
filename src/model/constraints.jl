using JuMP

## Number of items in forms
function constraint_items_per_form(model::Model, parms::Parameters, minItems::Int,
                                   maxItems::Int=minItems)
    return constraint_item_count(model::Model,
                                 parms::Parameters,
                                 trues(size(parms.bank, 1)),
                                 minItems::Int,
                                 maxItems::Int)
end

## Includes a number of items between minItems and maxItems from the "selected" list in each test (F)
function constraint_item_count_aux(model::Model, parms::Parameters, selected::BitVector,
                                   minItems::Int, maxItems::Int=minItems)
    @assert(0 <= minItems <= maxItems, "Error in item_count. maxItems < minItems")

    x = model[:x]
    rows, forms = size(x)
    forms -= parms.shadow_test > 0 ? 1 : 0
    items = collect(1:rows)[selected]

    if minItems == maxItems
        @constraint(model, [f = 1:forms], sum(x[i, f] for i in items) == maxItems)
    else
        @constraint(model, [f = 1:forms], sum(x[i, f] for i in items) <= maxItems)
        @constraint(model, [f = 1:forms], sum(x[i, f] for i in items) >= minItems)
    end
    return model
end

## Include a number of items between minItems and maxItems from the list for the shadow test
function constraint_item_count_shadow_aux(model::Model, parms::Parameters,
                                          selected::BitVector,
                                          minItems::Int, maxItems::Int=minItems)
    shadow_test = parms.shadow_test
    if shadow_test > 0
        @assert(minItems <= maxItems, "Error in item_count. maxItems < minItems")
        x = model[:x]
        zcol = size(x, 2)
        items = collect(1:size(x, 1))[selected]

        if minItems == maxItems
            @constraint(model, sum(x[i, zcol] for i in items) == minItems * shadow_test)
        else
            @constraint(model, sum(x[i, zcol] for i in items) >= minItems * shadow_test)
            @constraint(model, sum(x[i, zcol] for i in items) <= maxItems * shadow_test)
        end
    end

    return model
end

function constraint_item_count(model::Model, parms::Parameters, selected::BitVector,
                               minItems::Int,
                               maxItems::Int=minItems)
    @assert(minItems <= maxItems, "Error in item_count. maxItems < minItems")

    constraint_item_count_aux(model::Model, parms::Parameters, selected, minItems::Int,
                              maxItems::Int)
    constraint_item_count_shadow_aux(model::Model, parms::Parameters, selected,
                                     minItems::Int, maxItems::Int)
    return model
end

## Enemies
function constraint_enemies_in_form(model::Model, parms::Parameters, selected)
    x = model[:x]
    forms = size(x, 2)
    forms -= parms.shadow_test > 0 ? 1 : 0

    data = DataFrame(; selected=selected, index=1:length(selected))
    dropmissing!(data, :selected)

    groups = groupby(data, :selected; sort=false, skipmissing=true)
    for group in groups
        items = group[!, :index]
        @constraint(model, [f = 1:forms], sum(x[i, f] for i in items) <= 1)
    end
end

## Friends
function constraint_friends_in_form(model::Model, parms::Parameters, selected)
    x = model[:x]
    forms = size(x, 2)
    forms -= parms.shadow_test > 0 ? 1 : 0

    data = DataFrame(; selected=selected, index=1:length(selected))
    dropmissing!(data, :selected)

    groups = groupby(data, :selected; sort=false, skipmissing=true)
    for group in groups
        items = group[!, :index]
        pivot = items[1]
        cnt = length(items)
        @constraint(model, [f = 1:forms],
                    sum(x[i, f] for i in items) == cnt * x[pivot, f])
    end
end

## Suma de los valores de los reactivos entre [minVal, maxVal] en la prueba sombra
function constraint_item_sum_shadow_aux(model::Model, parms::Parameters, vals, minVal,
                                        maxVal=minVal)
    @assert(minVal <= maxVal, "Error in item_sum. maxVal < minVal")

    shadow_test = parms.shadow_test
    if shadow_test > 0
        x = model[:x]
        items, zcol = size(x)

        if size(vals, 2) == 1
            cond = fill(true, length(vals))
            val = vals
        else
            cond, val = eachcol(vals)
        end

        @constraint(model,
                    sum([x[i, zcol] * vals[i] for i in 1:items if cond[1]]) >=
                    minVal * shadow_test)
        @constraint(model,
                    sum([x[i, zcol] * vals[i] for i in 1:items if cond[1]]) <=
                    maxVal * shadow_test)
    end

    return model
end

## Suma de los valores de los reactivos entre [minVal, maxVal] en cada prueba(F)
function constraint_item_sum_aux(model::Model, parms::Parameters, vals, minVal,
                                 maxVal=minVal)
    @assert(minVal <= maxVal, "Error in item_sum. maxVal < minVal")

    x = model[:x]
    if size(vals, 2) == 1
        cond = fill(true, length(vals))
        val = vals
    else
        cond, val = eachcol(vals)
    end

    items, forms = size(x)
    forms -= parms.shadow_test > 0 ? 1 : 0

    @constraint(model,
                [f = 1:forms],
                sum([x[i, f] * val[i] for i in 1:items if cond[1]]) <= maxVal)
    @constraint(model,
                [f = 1:forms],
                sum([x[i, f] * val[i] for i in 1:items if cond[1]]) >= minVal)
    return model
end

function constraint_item_sum(model::Model, parms::Parameters, vals, minVal, maxVal=minVal)
    @assert(minVal <= maxVal, "Error in item_sum. maxVal < minVal")

    constraint_item_sum_aux(model::Model, parms::Parameters, vals, minVal, maxVal)
    constraint_item_sum_shadow_aux(model::Model, parms::Parameters, vals, minVal, maxVal)
    return model
end

## Minimiza tolerancia en todos los puntos k de la curva característica
function objective_match_characteristic_curve!(model::Model, parms::Parameters)
    R, K = 1:(parms.r), 1:(parms.k)
    P, tau = parms.p, parms.tau
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms
    forms -= parms.shadow_test > 0 ? 1 : 0

    w = [1.0 for _ in R]

    # constrints for operational forms
    @constraint(model,
                [f = 1:forms, k = K, r = R],
                sum([P[i, k]^r * x[i, f] for i in 1:items]) <= tau[r, k] + (w[r] * y))

    @constraint(model,
                [f = 1:forms, k = K, r = R],
                sum([P[i, k]^r * x[i, f] for i in 1:items]) >= tau[r, k] - (w[r] * y))

    # Constraints for shadow test only
    if parms.shadow_test > 0
        shadow_test = parms.shadow_test

        #Stricter weights for the shadow test?
        w = [1.0, 0.8, 0.7, 0.75]

        @constraint(model, [k = K, r = R],
                    sum([P[i, k]^r * x[i, zcol] for i in 1:items]) <=
                    ((tau[r, k] + (w[r] * y)) * shadow_test))

        @constraint(model, [k = K, r = R],
                    sum([P[i, k]^r * x[i, zcol] for i in 1:items]) >=
                    ((tau[r, k] - (w[r] * y)) * shadow_test))
    end

    return model
end

## Tolerancia de y en todos los puntos k de la curva característica
function objective_match_information_curve!(model::Model, parms::Parameters)
    K, info, tau_info = parms.k, parms.info, parms.tau_info
    x, y = model[:x], model[:y]
    items, forms = size(x)
    zcol = forms
    forms -= parms.shadow_test > 0 ? 1 : 0

    @constraint(model,
                [f = 1:forms, k = 1:K],
                sum([info[i, k] * x[i, f] for i in 1:items]) <= tau_info[k] + y)

    @constraint(model,
                [f = 1:forms, k = 1:K],
                sum([info[i, k] * x[i, f] for i in 1:items]) >= tau_info[k] - y)

    if parms.shadow_test > 0
        shadow_test = parms.shadow_test

        @constraint(model,
                    [k = 1:K],
                    sum([info[i, k] * x[i, zcol] for i in 1:items]) <=
                    (tau_info[k] + y) * shadow_test)

        @constraint(model,
                    [k = 1:K],
                    sum([info[i, k] * x[i, zcol] for i in 1:items]) >=
                    (tau_info[k] - y) * shadow_test)
    end

    return model
end

## Tolerancia de y en todos los puntos k de la función de información
function objective_max_info(model::Model, parms::Parameters)
    R = parms.relative_target_weights
    K = parms.k
    info = parms.info
    x, y = model[:x], model[:y]
    (items, forms) = size(x)

    shadow = parms.shadow_test
    forms -= shadow > 0 ? 1 : 0

    @constraint(model,
                [f = 1:forms, k = 1:K],
                sum([info[i, k] * x[i, f] for i in 1:items]) >= R[k] * y)

    shadow > 0 && @constraint(model,
                              [k = 1:K],
                              sum([info[i, k] * x[i, forms + 1] for i in 1:items]) >=
                              R[k] * y * shadow)

    return model
end

## Maximiza info en puntos k
function objective_info_relative2(model::Model, parms::Parameters)
    R = parms.relative_target_weights
    K = length(parms.relative_target_weights)
    info = parms.info
    x, y = model[:x], model[:y]
    items, forms = size(x)
    k = 0
    for f in 1:forms
        # alterna k para cada versión
        k = k % K + 1
        @constraint(model,
                    sum([info[i, k] * x[i, f] for i in 1:items]) >= R[k] * y)
    end
    # shadow > 0 && @constraint(model,
    #                           sum([info[i, k] * x[i, xs] for i in 1:items])>=R[k] * y * shadow)
    return model
end

function constraint_add_anchor!(model::Model, parms::Parameters)
    if parms.anchor_tests > 0
        x = model[:x]
        all_items, forms = size(x)
        items = collect(1:all_items)
        anchor_items = items[parms.bank.ANCHOR .> 0]

        # Force anchors in operational forms (ignore shadow)
        forms -= parms.shadow_test > 0 ? 1 : 0

        for i in anchor_items
            for f in 1:forms
                JuMP.fix(x[i, f], 1; force=true)
            end
        end
    end
    return model
end

function constraint_max_use(model::Model, parms::Parameters, selected::BitVector,
                            max_use::Int)
    x = model[:x]
    num_items, forms = size(x)
    forms -= parms.shadow_test > 0 ? 1 : 0
    
    if max_use < forms
        items = collect(1:num_items)
        items[selected]
        items = parms.anchor_size > 0 ? items[parms.bank.ANCHOR .== 0] : items
    
        @constraint(model, max_use[i in items],
            sum([x[i, f] for f in 1:forms]) + parms.bank.ITEM_USE[i] <= max_use)
    end
    return model
end

# Define the function to constrain item overlap between tests
function constraint_forms_overlap(model::Model, parms::Parameters, minItems::Int,
                                  maxItems::Int=minItems)
    @assert(0 <= minItems <= maxItems, "Error in forms_overlap. maxItems < minItems")
    if parms.shadow_test < 1 && parms.anchor_tests == 0
        # Retrieve x and z from the model
        x = model[:x]
        # Get the number of forms (tests) from the size of x
        num_items, num_forms = size(x)

        @variable(model, z[1:num_items, 1:num_forms, 1:num_forms], Bin)
        items = collect(1:num_items)
        items = items[(parms.bank.ANCHOR .!= parms.anchor_tests)]

        # Loop over all items and test pairs (t1, t2) where t1 < t2
        # Sum of overlaps for item i between test t1 and t2 should be within bounds (minItems, maxItems)
        if minItems == maxItems
            @constraint(model, [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                        sum(z[i, t1, t2] for i in items) == maxItems)
        else
            @constraint(model, [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                        sum(z[i, t1, t2] for i in items) <= maxItems)
            @constraint(model, [t1 = 1:(num_forms - 1), t2 = (t1 + 1):num_forms],
                        sum(z[i, t1, t2] for i in items) >= minItems)
        end

        # Constraints to link z[i, t1, t2] with x[i, t1] and x[i, t2]
        @constraint(model, [i = items], 2 * z[i, t1, t2] <= x[i, t1] + x[i, t2])
        @constraint(model, [i = items], z[i, t1, t2] >= x[i, t1] + x[i, t2] - 1)
    end
end

# function constraint_prevent_overlap!(model::Model, parms::Parameters)
#     return constraint_max_use(model, parms, 1)
# end
