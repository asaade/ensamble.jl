using CSV
using DataFrames

include("utils.jl")
include("expression_parser.jl")


# Define a struct for holding constraint information
struct Constraint
    id::String
    type::InlineString
    condition::Function
    lb::Int
    ub::Int
end



# Function to read constraints from a CSV file
function read_constraints(file_path::String)
    df = CSV.read(file_path, DataFrame, missingstring=nothing)
    constraints = Dict{String, Constraint}()

    for row in eachrow(df)
        if row[:ONOFF] != "OFF"
            row = map(up!, row)
            cond_id = row[:CONSTRAINT_ID]
            type = row[:TYPE]
            condition_expr = row[:CONDITION]
            lb = row[:LB]
            ub = row[:UB]
            if strip(condition_expr) == ""
                condition = Meta.parse("df -> true")
            else
                try
                    condition = parse_criteria(condition_expr)
                catch e
                    println("Error parsing condition. ", e)
                    condition = eval(Meta.parse("df -> true"))
                end
            end
            #  println(condition)
            constraints[cond_id] = Constraint(cond_id, type, eval(condition), lb, ub)
        end
    end

    return constraints
end


# # Example usage
# file_path = "data/constraints.csv"
# constraints = read_constraints(file_path)

# # Assuming parms is already defined and contains necessary data
# model = nothing # Replace with your actual model

# for (cond_id, constraint) in constraints
#     condition = constraint.condition
#     lb = constraint.lb
#     ub = constraint.ub
#     if cond_id == "Number"
#         constraintCount(model, parms, condition, lb, ub)
#     elseif cond_id == "Sum"
#         constraint_sum(model, parms, condition, lb, ub)
#     end
# end



# function translateConstraints(data::DataFrame)
# in(["CONSTRAINT_ID", "TYPE", "WHAT", "CONDITION", "LB", "UB"]).(["CONSTRAINT_ID", "TYPE", "WHAT", "CONDITION", "LB", "UB"])
# in(results[!, i]).(results[!, j])
