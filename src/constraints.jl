using JuMP

# Define a struct for holding constraint information
struct Constraint
    id::String
    type::InlineString
    condition::Function
    lb::Int
    ub::Int
end

function apply_individual_constraint!(model::Model, parameters::Params, constraint::Constraint)
    lb, ub = constraint.lb, constraint.ub
    bank = parameters.bank
    items = 1:size(bank, 1)

    if constraint.type == "TEST"
        constraint_items_per_version(model, parameters, lb, ub)
    elseif constraint.type == "NUMBER"
        condition = Base.invokelatest(constraint.condition, bank)
        selected_items = items[condition]
        constraint_item_count(model, parameters, selected_items, lb, ub)
    elseif constraint.type == "SUM"
        item_vals = Base.invokelatest(constraint.condition, bank)
        constraint_item_sum(model, parameters, item_vals, lb, ub)
    else
        error("Unknown constraint type: ", constraint.type)
    end
end


## Número de reactivos totales (Items) en cada versión (forms)
function constraint_items_per_version(model::Model, parameters::Params, minItems::Int64, maxItems::Int64=minItems)
    constraint_item_count(model::Model, parameters::Params, 1:size(parameters.bank, 1), minItems::Int64, maxItems::Int64)
end


## Incluye un número de reactivos ente minItems y maxItems de la lista para la prueba sombra
function constraint_item_count_shadow_aux(model::Model, parameters::Params, selected, minItems::Int, maxItems::Int=minItems)
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
function constraint_item_count_aux(model::Model, parameters::Params, selected, minItems::Int64, maxItems::Int64=minItems)
    x = model[:x]
    forms = size(x, 2)
    forms -= parameters.shadow_test_size > 0 ? 1 : 0

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


function constraint_item_count(model::Model, parameters::Params, selected, minItems::Int64, maxItems::Int64=minItems)
    constraint_item_count_aux(model::Model, parameters::Params, selected, minItems::Int64, maxItems::Int64)
    constraint_item_count_shadow_aux(model::Model, parameters::Params, selected, minItems::Int64, maxItems::Int64)
    return model
end


## Suma de los valores de los reactivos entre [minVal, maxVal] en la prueba sombra
function constraint_item_sum_shadow_aux(model::Model, parameters::Params, vals, minVal, maxVal=minVal)
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
function constraint_item_sum_aux(model::Model, parameters::Params, vals, minVal, maxVal=minVal)
    x = model[:x]
    if size(vals, 2) == 1
        cond = fill(true, length(vals))
        val = vals
    else
        cond, val = eachcol(vals)
    end

    items, forms = size(x)
    forms -= parameters.shadow_test_size > 0 ? 1 : 0

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


function constraint_item_sum(model::Model, parameters::Params, vals, minVal, maxVal=minVal)
    constraint_item_sum_aux(model::Model, parameters::Params, vals, minVal, maxVal)
    constraint_item_sum_shadow_aux(model::Model, parameters::Params, vals, minVal, maxVal)
    return model
end


## Tolerancia de y en todos los puntos k de la curva característica
function constraint_TCC_shadow_aux(model::Model, parameters::Params)
    if parameters.shadow_test_size > 0
        R, K = 1:parameters.r, 1:parameters.k
        P, tau = parameters.p, parameters.tau
        x, y = model[:x], model[:y]
        items, zcol = size(x)

        shadow_test_size = parameters.shadow_test_size

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
function constraint_TCC_aux(model::Model, parameters::Params)
    R, K = 1:parameters.r, 1:parameters.k
    P, tau = parameters.p, parameters.tau
    x, y = model[:x], model[:y]
    items, forms = size(x)
    forms -= parameters.shadow_test_size > 0 ? 1 : 0

    # w = [1.0 for _ in R]
    # w = [1.15 - (0.15 * (r - 1)) for r in R]
    w = [1.0, 0.85, 0.75, 0.70]

    @constraint(model, [f=1:forms, k=K, r=R],
                sum([P[i, k] ^ r * x[i, f] for i in 1:items]) <= tau[r, k] + (w[r] * y));

    @constraint(model, [f=1:forms, k=K, r=R],
                sum([P[i, k] ^ r * x[i, f] for i in 1:items]) >= tau[r, k] - (w[r] * y));

    return model
end


function objective_match_characteristic_curve!(model::Model, parameters::Params)
    constraint_TCC_aux(model::Model, parameters::Params)
    constraint_TCC_shadow_aux(model::Model, parameters::Params)
end


## Tolerancia de y en todos los puntos k de la curva característica
function constraint_ICC_shadow_aux(model::Model, parameters::Params)
    if parameters.shadow_test_size > 0
        K = parameters.k;
        info = parameters.info;
        tau_info = parameters.tau_info;

        x, y = model[:x], model[:y]
        items, zcol = size(x);
        shadow_test_size = parameters.shadow_test_size

        @constraint(model, [k=1:K],
                    sum([info[i, k] * x[i, zcol] for
                             i in 1:items]) <= (tau_info[k] + y) * shadow_test_size);

        @constraint(model, [k=1:K],
                    sum([info[i, k] * x[i, zcol] for
                             i in 1:items]) >= (tau_info[k] - y) * shadow_test_size);
    end
    return model
end


## Tolerancia de y en todos los puntos k de la curva característica
function constraint_ICC_aux(model::Model, parameters::Params)
    K = parameters.k;
    info = parameters.info;
    tau_info = parameters.tau_info;

    x, y = model[:x], model[:y];
    items, forms = size(x)
    forms -= parameters.shadow_test_size > 0 ? 1 : 0

    @assert(length(info[:, 1]) == items)
    @assert(length(info[1, :]) == parameters.k)

    @constraint(model, [f=1:forms, k=1:K],
                sum([info[i, k] * x[i, f] for i in 1:items]) <= tau_info[k] + y);

    @constraint(model, [f=1:forms, k=1:K],
                sum([info[i, k] * x[i, f] for i in 1:items]) >= tau_info[k] - y);

    return model
end


function objective_match_information_curve!(model::Model, parameters::Params)
    constraint_ICC_aux(model::Model, parameters::Params)
    constraint_ICC_shadow_aux(model::Model, parameters::Params)
end

## Tolerancia de y en todos los puntos k de la función de información
function objective_info_relative(model::Model, parameters::Params)
    R = parameters.relative_target_weights;
    K = parameters.k
    info = parameters.info
    x, y = model[:x], model[:y]
    (items, forms)  = size(x)

    shadow = parameters.shadow_test_size
    forms -= shadow > 0 ? 1 : 0

    @constraint(model, [f=1:forms, k=1:K],
                sum([info[i, k] * x[i, f] for i in 1:items]) >= R[k] * y);

    shadow > 0 &&  @constraint(model, [k=1:K],
                               sum([info[i, k] * x[i, forms + 1]
                                    for i in 1:items]) >= R[k] * y * shadow);

    return model
end

## Tolerancia de y en puntos variables k de la función de información
function objective_info_relative2(model::Model,
                                  parameters::Params,
                                  original_parameters::Params)
    R = parameters.relative_target_weights;
    k = mod(parameters.k, original_parameters.k) + 1

    parameters.k = k
    info = parameters.info
    x, y = model[:x], model[:y]
    (items, forms)  = size(x)
    xs = forms
    shadow = parameters.shadow_test_size
    forms -= shadow > 0 ? 1 : 0

    @constraint(model, [f=1:forms],
                sum([info[i, k] * x[i, f] for i in 1:items]) >= R[k] * y);

    shadow > 0 &&  @constraint(model,
                               sum([info[i, k] * x[i, xs]
                                    for i in 1:items]) >= R[k] * y * shadow);

    return model
end

function constraint_add_anchor!(model::Model, parameters::Params)
    if parameters.anchor_number > 0
        x = model[:x]
        all_items, forms = size(x)
        items = collect(1:all_items)

        anchor_items = items[(parameters.bank.ANCHOR .== parameters.anchor_number)]
        bank_items = setdiff(items, anchor_items)

        if forms > 1
            @constraint(model, [i=bank_items], sum([x[i, f] for f in 1:forms]) <= 1);
        end

        forms -= parameters.shadow_test_size > 0 ? 1 : 0

        for f in 1:forms
            for i in anchor_items
                JuMP.fix(x[i, f], 1; force=true)
            end
        end
    end
    return model
end


function constraint_prevent_overlap!(model::Model, parameters::Params)
    if parameters.anchor_number == 0
        x = model[:x]
        items, forms = size(x)
        if forms > 1
            @constraint(model, [i=1:items], sum([x[i, f] for f in 1:forms]) <= 1);
        end
    end
    return model
end


function constraint_max_use(model::Model, parameters::Params, overlap=1)
    if parameters.anchor_number == 0
        x = model[:x]
        items, forms = size(x)
        forms -= parameters.shadow_test_size > 0 ? 1 : 0

        if forms > 1
            @constraint(model, [i=1:items], sum([x[i, f] for f in 1:forms]) <= overlap);
        end
    end
    return model
end


# function constraint_sets(model::Model, parameters::Params, group_ids, selected, minItems::Int, maxItems::Int=minItems)


# function objective_match_items(model::Model, parameters::Params)
#     x, y = model[:x], model[:y];
#     items, forms = size(x)
#     D(I, LL=-1.5, W=3.0, N=parameters.n) = W * (I-1) / ((N-1) - LL)
#     reference20_80 =

#     @constraint(model, [i=1:items, j=i:(items-1), f=1:forms],
#                 x[i, f] * x[j, f] * parameters.delta[i, j] <= y)

#     @constraint(model, [i=1:items, j=i:items, f=1:forms],
#                 x[i, f] * x[j, f] * parameters.delta[i, j] >= -y)

#     @constraint(model, [j=1:items, f=1:forms],
#                 sum([x[i, f] * x[j, f] for i in 1:items if i < j] == 1) == 1)

#     # @constraint(model, [j=1:items, f=1:forms],
#     #              sum([x[i, f] + x[j, f] for i in 1:items] == 1 if i < j) == 1)

# end
