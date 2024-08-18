module CriteriaParser
    export parse_criteria

    using DataFrames

    # Helper function to normalize the input string
    function normalize_and_parse(input_str::String)
        input_str = replace(input_str, r"\s*([!=<>]=?|in|\&|\|)\s*" => s" \1 ")
        input_str = replace(input_str, r"\s*,\s*" => ",")
        return Meta.parse(input_str)
    end

    # Function to handle different operations and their vectorized forms
    function transform_condition(op, lhs, rhs)
        if !isa(lhs, Symbol)
            throw(ArgumentError("Left-hand side (lhs) must be a valid column name as a symbol"))
        end

        # Handle rhs as an integer, string, or collection
        if isa(rhs, Int) || isa(rhs, Float64)
            # Do nothing, rhs is already a valid number
        elseif isa(rhs, Char)
            rhs = string(rhs)
        elseif isa(rhs, Symbol)
            rhs = string(rhs)
        elseif rhs isa Expr && (rhs.head == :vect || rhs.head == :tuple)
            rhs = Expr(:vect, [isa(arg, Char) ? string(arg) : arg for arg in rhs.args]...)
        elseif !(isa(rhs, AbstractString) || isa(rhs, Number))
            throw(ArgumentError("Right-hand side (rhs) must be a number, string, or collection"))
        end

        op_map = Dict(
            "==" => ".==",
            "<=" => ".<=",
            "<"  => ".<",
            ">=" => ".>=",
            ">"  => ".>",
            "!=" => ".!=",
            "in" => ".∈",
            "!in" => ".∉"
        )

        op_str = string(op)
        if !haskey(op_map, op_str)
            throw(ArgumentError("Unsupported operator: $op"))
        end

        lhs = Expr(:., :df, QuoteNode(lhs))
        vectorized_op = Symbol(op_map[op_str])

        # Ensure that rhs is correctly parsed as a collection
        if (op_str == "in" || op_str == "!in") && (rhs isa Expr && (rhs.head == :vect || rhs.head == :tuple))
            rhs = Expr(:call, :Ref, rhs)
        end

        return Expr(:call, vectorized_op, lhs, rhs)
    end

    # Function to recursively transform parsed expressions with AND and OR logic
    function transform_logical(expr)
        if expr.head == :call
            op, lhs, rhs = expr.args
            return transform_condition(op, lhs, rhs)
        elseif expr isa Symbol
            return Expr(:., :df, QuoteNode(expr))
        else
            return expr
        end
    end

    # Main function to parse the criteria string and build the final Expr
    function parse_criteria(input_str::String)
        if strip(input_str) == ""
            return Meta.parse("df -> trues(size(df, 1))")
        end

        parsed_expr = normalize_and_parse(input_str)
        if isa(parsed_expr, Symbol)
            return Meta.parse("df -> df[:, $(QuoteNode(parsed_expr))]")
        end

        if parsed_expr.head == :tuple && length(parsed_expr.args) == 2
            column_expr, condition_expr = parsed_expr.args

            if !(column_expr isa Symbol)
                throw(ArgumentError("Column selection must be a valid column name"))
            end

            condition = transform_logical(condition_expr)

            return quote
                df -> begin
                    hcat($condition, df[!, $(QuoteNode(column_expr))])
                end
            end
        else
            condition = transform_logical(parsed_expr)

            return quote
                df -> begin
                    $condition
                end
            end
        end
    end
end

# # Example usage
# using .CriteriaParser

# df = DataFrame(AREA = [1, 2, 3, 4], CLAVE = ["A", "B", "C", "D"], A = [0.6, 0.4, 0.9, 0.2])

# # Test cases
# expr1 = CriteriaParser.parse_criteria("AREA == 1")
# result1 = eval(expr1)(df)
# println(result1)  # Expected: Bool[1, 0, 0, 0]

# expr1_1 = CriteriaParser.parse_criteria("CLAVE == A")
# result1_1 = eval(expr1_1)(df)
# println(result1_1)  # Expected: Bool[1, 0, 0, 0]

# expr2 = CriteriaParser.parse_criteria("AREA in [1, 2, 3]")
# result2 = eval(expr2)(df)
# println(result2)  # Expected: Bool[1, 1, 1, 0]

# expr3 = CriteriaParser.parse_criteria("CLAVE, AREA == 1")
# result3 = eval(expr3)(df)
# println(result3)  # Expected: hcat(Bool[1, 0, 0, 0], df[!, :CLAVE])

# expr4 = CriteriaParser.parse_criteria("AREA")
# result4 = eval(expr4)(df)
# println(result4)  # Expected: df[!, :AREA]

# expr5 = CriteriaParser.parse_criteria("CLAVE in ['A', 'D']")
# result5 = eval(expr5)(df)
# println(result5)  # Expected: Bool[1, 0, 0, 1]

# expr6 = CriteriaParser.parse_criteria("")
# result6 = eval(expr6)(df)
# println(result6)  # Expected: Bool[true, true, true, true]
