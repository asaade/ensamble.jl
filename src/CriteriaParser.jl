module CriteriaParser
    export parse_criteria

    using DataFrames

    # Helper function to normalize the input string
    function normalize(input_str::String)
        input_str = replace(input_str, r"\s*([!=<>]=?|in|\&|\|)\s*" => s" \1 ")
        input_str = replace(input_str, r"\s*,\s*" => ",")
        return input_str
    end

    # Function to handle different operations in a regular function
    function apply_condition(op::String, lhs::Symbol, rhs, df::DataFrame)
        if lhs ∉ Symbol.(names(df))
            throw(ArgumentError("Column $lhs does not exist in the DataFrame"))
        end

        lhs_col = df[!, lhs]

        if op == "=="
            return lhs_col .== rhs
        elseif op == "<="
            return lhs_col .<= rhs
        elseif op == "<"
            return lhs_col .< rhs
        elseif op == ">="
            return lhs_col .>= rhs
        elseif op == ">"
            return lhs_col .> rhs
        elseif op == "!="
            return lhs_col .!= rhs
        elseif op == "in"
            return lhs_col .∈ Ref(rhs)
        elseif op == "!in"
            return lhs_col .∉ Ref(rhs)
        else
            throw(ArgumentError("Unsupported operator: $op"))
        end
    end

    # Function to parse the criteria string and apply it to a DataFrame
    function parse_criteria(input_str::String)
        if strip(input_str) == ""
            return df -> trues(size(df, 1))
        end

        normalized_str = normalize(input_str)

        if contains(normalized_str, ",")
            col_expr, condition_expr = split(normalized_str, ",", limit=2)
            col_expr = strip(col_expr)

            condition_parts = split(condition_expr, r"\s+")

            if length(condition_parts) == 3
                lhs, op, rhs = condition_parts
                op = String(op)

                # Parse the right-hand side (rhs)
                if occursin(r"[\[\]]", rhs)  # Check if rhs is a collection
                    rhs = eval(Meta.parse(rhs))  # Safely evaluate as a literal
                elseif occursin(r"^-?\d+(\.\d+)?$", rhs)  # Check if rhs is a number
                    rhs = parse(Float64, rhs)
                elseif occursin(r"^\".*\"$", rhs)  # Check if rhs is a string with double quotes
                    rhs = strip(rhs, ['\"'])  # Remove double quotes
                else  # Assume rhs is a string literal without quotes
                    rhs = rhs
                end

                return df -> df[apply_condition(op, Symbol(lhs), rhs, df), Symbol(col_expr)]
            else
                throw(ArgumentError("Invalid criteria string format"))
            end
        else
            parts = split(normalized_str, r"\s+")

            if length(parts) == 3
                lhs, op, rhs = parts
                op = String(op)  # Convert SubString to String

                # Parse the right-hand side (rhs)
                if occursin(r"[\[\]]", rhs)  # Check if rhs is a collection
                    rhs = eval(Meta.parse(rhs))  # Safely evaluate as a literal
                elseif occursin(r"^-?\d+(\.\d+)?$", rhs)  # Check if rhs is a number
                    rhs = parse(Float64, rhs)
                elseif occursin(r"^\".*\"$", rhs)  # Check if rhs is a string with double quotes
                    rhs = strip(rhs, ['\"'])  # Remove double quotes
                else  # Assume rhs is a string literal without quotes
                    rhs = rhs
                end

                return df -> apply_condition(op, Symbol(lhs), rhs, df)

            elseif length(parts) == 1
                lhs = Symbol(parts[1])
                return df -> df[!, lhs]

            else
                throw(ArgumentError("Invalid criteria string format"))
            end
        end
    end
end

# # Example usage
# using .CriteriaParser
# using DataFrames

# df = DataFrame(AREA = [1, 2, 3, 4], CLAVE = ["A", "B", "C", "D"], A = [0.6, 0.4, 0.9, 0.2])

# # Test cases
# expr1 = CriteriaParser.parse_criteria("AREA == 1")
# result1 = expr1(df)
# println(result1)  # Expected: Bool[1, 0, 0, 0]

# expr1_1 = CriteriaParser.parse_criteria("CLAVE == A")
# result1_1 = expr1_1(df)
# println(result1_1)  # Expected: Bool[1, 0, 0, 0]

# expr2 = CriteriaParser.parse_criteria("AREA in [1, 2, 3]")
# result2 = expr2(df)
# println(result2)  # Expected: Bool[1, 1, 1, 0]

# expr3 = CriteriaParser.parse_criteria("CLAVE, AREA == 1")
# result3 = expr3(df)
# println(result3)  # Expected: the values in CLAVE where AREA == 1 (here is "A")

# expr4 = CriteriaParser.parse_criteria("AREA")
# result4 = expr4(df)
# println(result4)  # Expected: df[!, :AREA]

# expr5 = CriteriaParser.parse_criteria("CLAVE in [\"A\", \"D\"]")
# result5 = expr5(df)
# println(result5)  # Expected: Bool[1, 0, 0, 1]

# expr6 = CriteriaParser.parse_criteria("")
# result6 = expr6(df)
# println(result6)  # Expected: Bool[true, true, true, true]
