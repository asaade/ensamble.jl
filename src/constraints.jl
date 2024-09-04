using JuMP

include("types.jl")

## Number of items in forms
function constraint_items_per_form(model::Model, parms::Parameters, minItems::Int,
                                   maxItems::Int = minItems)
    return constraint_item_count(model::Model,
                                 parms::Parameters,
                                 1:size(parms.bank, 1),
                                 minItems::Int,
                                 maxItems::Int)
end

## Enemies
function constraint_enemies_in_form(model::Model, parms::Parameters, selected)
    x = model[:x]
    forms = size(x, 2)
    forms -= parms.shadow_test > 0 ? 1 : 0

    data = DataFrame(; selected = selected, index = 1:length(selected))
    dropmissing!(data, :selected)

    if isa(selected, BitVector)
        data = data[data.selected, :]
    end

    groups = groupby(data, :selected; sort = false, skipmissing = true)
    for group in groups
        items = group[!, :index]
        @constraint(model, [f = 1:forms], sum(x[i, f] for i in items)<=1)
    end
end

## Enemies
function constraint_friends_in_form(model::Model, parms::Parameters, selected)
    x = model[:x]
    forms = size(x, 2)
    forms -= parms.shadow_test > 0 ? 1 : 0

    data = DataFrame(; selected = selected, index = 1:length(selected))
    dropmissing!(data, :selected)
    if isa(selected, BitVector)
        data = data[data.selected, :]
    end
    groups = groupby(data, :selected; sort = false, skipmissing = true)
    for group in groups
        items = group[!, :index]
        pivot = items[1]
        cnt = length(items)
        @constraint(model, [f = 1:forms],
                    sum(x[i, f] for i in items)==cnt * x[pivot, f])
    end
end

## Incluye un número de reactivos ente minItems y maxItems de la lista para la prueba sombra
function constraint_item_count_shadow_aux(model::Model, parms::Parameters, selected,
                                          minItems::Int, maxItems::Int = minItems)
    shadow_test = parms.shadow_test

    if shadow_test > 0
        x = model[:x]
        zcol = size(x, 2)

        if minItems == maxItems && minItems >= 0
            @constraint(model, sum(x[i, zcol] for i in selected)==minItems * shadow_test)
        elseif minItems < maxItems
            @constraint(model, sum(x[i, zcol] for i in selected)>=minItems * shadow_test)
            @constraint(model, sum(x[i, zcol] for i in selected)<=maxItems * shadow_test)
        end
    end

    return model
end

## Incluye un número de reactivos ente minItems y maxItems] de la lista "selected" en cada prueba(F)
function constraint_item_count_aux(model::Model, parms::Parameters, selected,
                                   minItems::Int, maxItems::Int = minItems)
    x = model[:x]
    forms = size(x, 2)
    forms -= parms.shadow_test > 0 ? 1 : 0

    if minItems == maxItems && minItems >= 0
        @constraint(model, [f = 1:forms], sum(x[i, f] for i in selected)==minItems)
    elseif minItems < maxItems
        @constraint(model, [f = 1:forms], sum(x[i, f] for i in selected)<=maxItems)
        @constraint(model, [f = 1:forms], sum(x[i, f] for i in selected)>=minItems)
    else
        println("No contraints created in constraint_item_count")
    end

    return model
end

function constraint_item_count(model::Model, parms::Parameters, selected, minItems::Int,
                               maxItems::Int = minItems)
    constraint_item_count_aux(model::Model, parms::Parameters, selected, minItems::Int,
                              maxItems::Int)
    constraint_item_count_shadow_aux(model::Model, parms::Parameters, selected,
                                     minItems::Int, maxItems::Int)
    return model
end

## Suma de los valores de los reactivos entre [minVal, maxVal] en la prueba sombra
function constraint_item_sum_shadow_aux(model::Model, parms::Parameters, vals, minVal,
                                        maxVal = minVal)
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

        if minVal == maxVal && minVal >= 0
            @constraint(model,
                        sum([x[i, zcol] * vals[i] for i in 1:items if cond[1]])==
                        verminVal * shadow_test)
        elseif minVal < maxVal
            @constraint(model,
                        sum([x[i, zcol] * vals[i] for i in 1:items if cond[1]])>=
                        minVal * shadow_test)
            @constraint(model,
                        sum([x[i, zcol] * vals[i] for i in 1:items if cond[1]])<=
                        maxVal * shadow_test)
        end
    end

    return model
end

## Suma de los valores de los reactivos entre [minVal, maxVal] en cada prueba(F)
function constraint_item_sum_aux(model::Model, parms::Parameters, vals, minVal,
                                 maxVal = minVal)
    x = model[:x]
    if size(vals, 2) == 1
        cond = fill(true, length(vals))
        val = vals
    else
        cond, val = eachcol(vals)
    end

    items, forms = size(x)
    forms -= parms.shadow_test > 0 ? 1 : 0

    if minVal == maxVal && minVal >= 0
        @constraint(model,
                    [f = 1:forms],
                    sum([x[i, f] * val[i] for i in 1:items if cond[1]])==maxVal)
    elseif minVal < maxVal
        @constraint(model,
                    [f = 1:forms],
                    sum([x[i, f] * val[i] for i in 1:items if cond[1]])<=maxVal)
        @constraint(model,
                    [f = 1:forms],
                    sum([x[i, f] * val[i] for i in 1:items if cond[1]])>=minVal)
    else
        println("No contraints created in constraint_item_sum")
    end

    return model
end

function constraint_item_sum(model::Model, parms::Parameters, vals, minVal, maxVal = minVal)
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
                sum([P[i, k]^r * x[i, f] for i in 1:items])<=tau[r, k] + (w[r] * y))

    @constraint(model,
                [f = 1:forms, k = K, r = R],
                sum([P[i, k]^r * x[i, f] for i in 1:items])>=tau[r, k] - (w[r] * y))

    # Constraints for shadow test only
    if parms.shadow_test > 0
        shadow_test = parms.shadow_test

        #Stricter weights for the shadow test?
        w = [1.0, 0.8, 0.7, 0.75]

        @constraint(model, [k = K, r = R],
                    sum([P[i, k]^r * x[i, zcol] for i in 1:items])<=
                    ((tau[r, k] + (w[r] * y)) * shadow_test))

        @constraint(model, [k = K, r = R],
                    sum([P[i, k]^r * x[i, zcol] for i in 1:items])>=
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
                sum([info[i, k] * x[i, f] for i in 1:items])<=tau_info[k] + y)

    @constraint(model,
                [f = 1:forms, k = 1:K],
                sum([info[i, k] * x[i, f] for i in 1:items])>=tau_info[k] - y)

    if parms.shadow_test > 0
        shadow_test = parms.shadow_test

        @constraint(model,
                    [k = 1:K],
                    sum([info[i, k] * x[i, zcol] for i in 1:items])<=
                    (tau_info[k] + y) * shadow_test)

        @constraint(model,
                    [k = 1:K],
                    sum([info[i, k] * x[i, zcol] for i in 1:items])>=
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
                sum([info[i, k] * x[i, f] for i in 1:items])>=R[k] * y)

    shadow > 0 && @constraint(model,
                [k = 1:K],
                sum([info[i, k] * x[i, forms + 1] for i in 1:items])>=
                R[k] * y * shadow)

    return model
end

## Maximiza info en puntos k
function objective_info_relative2(model::Model, parms::Parameters,
                                  original_parms::Parameters)
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

# ## Maximiza info en puntos k
# function objective_info_relative2(model::Model, parms::Parameters,
#                                   original_parms::Parameters)
#     R = parms.relative_target_weights

#     # alterna k para esta versión
#     k = mod(parms.k, original_parms.k) + 1
#     parms.k = k
#     info = parms.info
#     x, y = model[:x], model[:y]
#     items, forms = size(x)
#     xs = forms
#     shadow = parms.shadow_test
#     forms -= shadow > 0 ? 1 : 0

#     @constraint(model, [f = 1:forms],
#                 sum([info[i, k] * x[i, f] for i in 1:items])>=R[k] * y)

#     shadow > 0 && @constraint(model,
#                 sum([info[i, k] * x[i, xs] for i in 1:items])>=R[k] * y * shadow)

#     return model
# end

function constraint_add_anchor!(model::Model, parms::Parameters)
    if parms.anchor_tests > 0
        x = model[:x]
        all_items, forms = size(x)
        items = collect(1:all_items)

        anchor_items = items[(parms.bank.ANCHOR .== parms.anchor_tests)]
        bank_items = setdiff(items, anchor_items)

        for i in bank_items
            if parms.bank.ITEM_USE[i] + forms > parms.max_item_use
                # Avoid overlap of non-anchor items (operational and shadow)
                @constraint(model,
                            sum([x[i, f] for f in 1:forms]) +
                            parms.bank.ITEM_USE[i]<=
                            parms.max_item_use)
            end
        end

        # Force anchors in operational forms (ignore shadow)
        forms -= parms.shadow_test > 0 ? 1 : 0

        for f in 1:forms
            for i in anchor_items
                JuMP.fix(x[i, f], 1; force = true)
            end
        end
    end
    return model
end

function constraint_max_use(model::Model, parms::Parameters)
    if parms.anchor_tests == 0
        x = model[:x]
        items, forms = size(x)
        # forms -= parms.shadow_test > 0 ? 1 : 0
        @constraint(model, [i = 1:items],
                    sum([x[i, f] for f in 1:forms]) +
                    parms.bank.ITEM_USE[i]<=
                    parms.max_item_use)
    end

    return model
end

function constraint_prevent_overlap!(model::Model, parms::Parameters)
    return constraint_max_use(model, parms)
end

## TODO
# function objective_match_items(model::Model, parms::Parameters)
#     x, y = model[:x], model[:y];
#     items, forms = size(x)
#     D(I, LL=-1.5, W=3.0, N=parms.n) = W * (I-1) / ((N-1) - LL)
#     reference20_80 =

#     @constraint(model, [i=1:items, j=i:(items-1), f=1:forms],
#                 x[i, f] * x[j, f] * parms.delta[i, j] <= y)

#     @constraint(model, [i=1:items, j=i:items, f=1:forms],
#                 x[i, f] * x[j, f] * parms.delta[i, j] >= -y)

#     @constraint(model, [j=1:items, f=1:forms],
#                 sum([x[i, f] * x[j, f] for i in 1:items if i < j] == 1) == 1)

#     # @constraint(model, [j=1:items, f=1:forms],
#     #              sum([x[i, f] + x[j, f] for i in 1:items] == 1 if i < j) == 1)

# end
