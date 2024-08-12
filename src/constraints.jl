using JuMP

## Número de reactivos totales (Items) en cada versión (forms)
function constraint_items_per_version(model, parameters, minItems, maxItems=minItems)
    constraint_item_count(model, parameters, 1:size(parameters.bank, 1), minItems, maxItems)
end


function constraint_prevent_overlap!(model, parameters)
    if parameters.anchor_number == 0
        x = model[:x]
        items, forms = size(x)
        @constraint(model, [i=1:items], sum([x[i, f] for f in 1:forms]) <= 1);
    end
    return model
end


## Incluye un número de reactivos ente minItems y maxItems de la lista para la prueba sombra
function constraint_item_count_shadow_aux(model, parameters, selected, minItems, maxItems=minItems)
    shadow_test_size = parameters.shadow_test_size

    if shadow_test_size > 0
        x = model[:x]
        zcol = size(x, 2)

        if minItems == maxItems && minItems >= 0
            @constraint(model, sum(x[i, zcol] for i in selected) == minItems * shadow_test_size);
        elseif minItems < maxItems
            @constraint(model, sum(x[i, zcol] for i in selected) >= minItems * shadow_test_size);
            @constraint(model, sum(x[i, zcol] for i in selected) <= maxItems * shadow_test_size);
        end
    end

    return model
end


## Incluye un número de reactivos ente minItems y maxItems] de la lista "selected" en cada prueba(F)
function constraint_item_count_aux(model, parameters, selected, minItems, maxItems=minItems)
    x = model[:x]
    forms = size(x, 2)
    forms -= (parameters.shadow_test_size > 0 ? 1 : 0)

    if minItems == maxItems && minItems >= 0
        @constraint(model, [f=1:forms], sum(x[i, f] for i in selected) == maxItems);
    elseif minItems < maxItems
        @constraint(model, [f=1:forms], sum(x[i, f] for i in selected) <= maxItems);
        @constraint(model, [f=1:forms], sum(x[i, f] for i in selected) >= minItems);
    else
        println("No contraints created in constraint_item_count")
    end

    return model
end


function constraint_item_count(model, parameters, selected, minItems, maxItems=minItems)
    constraint_item_count_aux(model, parameters, selected, minItems, maxItems)
    constraint_item_count_shadow_aux(model, parameters, selected, minItems, maxItems)
    return model
end


## Suma de los valores de los reactivos entre [minVal, maxVal] en la prueba sombra
function constraint_item_sum_shadow_aux(model, parameters, vals, minVal, maxVal=minVal)
    shadow_test_size = parameters.shadow_test_size
    if shadow_test_size > 0
        x = model[:x]
        items, zcol = size(x)

        if size(vals, 2) == 1
            cond = fill(true, length(vals))
            val = vals
        else
            cond, val = eachcol(vals)
        end

        if minVal == maxVal && minVal >= 0
            @constraint(model, sum([x[i, zcol] * vals[i] for i in 1:items if cond[1]]) == minVal * shadow_test_size);
        elseif minVal < maxVal
            @constraint(model, sum([x[i, zcol] * vals[i] for i in 1:items if cond[1]]) >= minVal * shadow_test_size);
            @constraint(model, sum([x[i, zcol] * vals[i] for i in 1:items if cond[1]]) <= maxVal * shadow_test_size);
        end
    end

    return model
end


## Suma de los valores de los reactivos entre [minVal, maxVal] en cada prueba(F)
function constraint_item_sum_aux(model, parameters, vals, minVal, maxVal=minVal)
    x = model[:x]
    if size(vals, 2) == 1
        cond = fill(true, length(vals))
        val = vals
    else
        cond, val = eachcol(vals)
    end

    items, forms = size(x)
    forms -= (parameters.shadow_test_size > 0 ? 1 : 0)

    if minVal == maxVal && minVal >= 0
        @constraint(model, [f=1:forms], sum([x[i, f] * val[i] for i in 1:items if cond[1]]) == maxVal);
    elseif minVal < maxVal
        @constraint(model, [f=1:forms], sum([x[i, f] * val[i] for i in 1:items if cond[1]]) <= maxVal);
        @constraint(model, [f=1:forms], sum([x[i, f] * val[i] for i in 1:items if cond[1]]) >= minVal);
    else
        println("No contraints created in constraint_item_sum")
    end

    return model
end


function constraint_item_sum(model, parameters, vals, minVal, maxVal=minVal)
    constraint_item_sum_aux(model, parameters, vals, minVal, maxVal)
    constraint_item_sum_shadow_aux(model, parameters, vals, minVal, maxVal)
    return model
end


## Tolerancia de y en todos los puntos k de la curva característica
function constraint_TCC_shadow_aux(model, parameters)
    if parameters.shadow_test_size > 0
        R = 1:parameters.r;
        K = 1:parameters.k;
        P = parameters.p;
        tau = parameters.tau;

        x, y = model[:x], model[:y]

        items, zcol = size(x);
        shadow_test_size = parameters.shadow_test_size

        @assert(length(P[:, 1]) == items)
        @assert(length(P[1, :]) == parameters.k)
        @assert(parameters.r == length(tau[:, 1]))

        # w = [1.0 for _ in R]
        # w = [1.15 - (0.15 * r) for r in R]
        w = [1.0, 0.8, 0.7, 0.75]

        @constraint(model, [k=K, r=R],
                    sum([P[i, k] ^ r * x[i, zcol] for
                             i in 1:items]) <= ((tau[r, k] + (w[r] * y)) * shadow_test_size));
        @constraint(model, [k=K, r=R],
                    sum([P[i, k] ^ r * x[i, zcol] for
                             i in 1:items]) >= ((tau[r, k] - (w[r] * y)) * shadow_test_size));
    end
    return model
end


## Tolerancia de y en todos los puntos k de la curva característica
function constraint_TCC_aux(model, parameters)
    R = 1:parameters.r;
    K = 1:parameters.k;
    P = parameters.p;
    tau = parameters.tau;
    x, y = model[:x], model[:y];
    items, forms = size(x)
    forms -= (parameters.shadow_test_size > 0 ? 1 : 0)

    @assert(length(P[:, 1]) == items)
    @assert(length(P[1, :]) == parameters.k)
    @assert(parameters.r == length(tau[:, 1]))

    # w = [1.0 for _ in R]
    # w = [1.15 - (0.15 * (r - 1)) for r in R]
    w = [1.0, 0.85, 0.75, 0.70]

    @constraint(model, [f=1:forms, k=K, r=R],
                sum([P[i, k] ^ r * x[i, f] for i in 1:items]) <= tau[r, k] + (w[r] * y));
    @constraint(model, [f=1:forms, k=K, r=R],
                sum([P[i, k] ^ r * x[i, f] for i in 1:items]) >= tau[r, k] - (w[r] * y));

    return model
end


function objective_match_characteristic_curve!(model, parameters)
    constraint_TCC_aux(model, parameters)
    constraint_TCC_shadow_aux(model, parameters)
end



function objective_match_items(model, parameters)
    x, y = model[:x], model[:y];
    items, forms = size(x)

    @constraint(model, [i=1:items, j=i:items, f=1:forms],
                x[i, f] * x[j, f] * parameters.delta[i, j] <= y)

    @constraint(model, [i=1:items, j=i:items, f=1:forms],
                x[i, f] * x[j, f] * parameters.delta[i, j] >= -y)

    @constraint(model, [j=1:items, f=1:forms],
                sum([x[i, f] * x[j, f] for i in 1:items if i < j] == 1) == 1)

    # @constraint(model, [j=1:items, f=1:forms],
    #              sum([x[i, f] + x[j, f] for i in 1:items] == 1 if i < j) == 1)

end

function constraint_add_anchor!(model, parameters)
    if parameters.anchor_number > 0
        x = model[:x]
        all_items, forms = size(x)
        items = collect(1:all_items)

        anchor_items = items[(parameters.bank.ANCHOR .== parameters.anchor_number)]
        bank_items = setdiff(items, anchor_items)

        @constraint(model, [i=bank_items], sum([x[i, f] for f in 1:forms]) <= 1);

        forms -= (parameters.shadow_test_size > 0 ? 1 : 0)

        for f in 1:forms
            for i in anchor_items
                JuMP.fix(x[i, f], 1; force=true)
            end
        end
    end
    return model
end



function constraint_max_use(model, parameters, overlap=0)
    if parameters.anchor_number == 0
        x = model[:x]
        items, forms = size(x)
        forms -= (parameters.shadow_test_size > 0 ? 1 : 0)

        if forms > 1
            @constraint(model, [i=1:items], sum([x[i, f] for f in 1:forms]) <= 1 + overlap);
        end
    end
    return model
end



# ## Tolerancia de y en todos los puntos k de la función de información
# function constraintInfo(model, parameters)
#     info = parameters.info;
#     infoTau = parameters.tau_info;
#     K = 1:parameters.k;
#     x = model[:x]
#     y = model[:y]
#     (Items, forms)  = size(x)

#     @assert(length(info[:, 1]) == length(x) == Items)

#     @constraint(model, [f=1:forms, k=K],
#                 sum([info[i, k] * x[i, f] for i in 1:Items]) <= infoTau[k] + y);
#     @constraint(model, [f=1:forms, k=K],
#                 sum([info[i, k] * x[i, f] for i in 1:Items]) >= infoTau[k] - y);
#     return model
# end
