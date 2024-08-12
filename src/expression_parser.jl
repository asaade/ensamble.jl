using DataFrames, MacroTools

# Function to normalize the input string and parse it
function normalize_and_parse(input_str::String)
    # Normalize spaces around operators and commas
    input_str = replace(input_str, r"\s*([!=<>]=?|in|\&|\|)\s*" => s" \1 ")
    input_str = replace(input_str, r"\s*,\s*" => ",")
    return Meta.parse(input_str)
end

# Function to transform a parsed expression into the required Expr
function transform_condition(op, lhs, rhs)
    # Check if lhs is a valid column name (symbol)
    if !isa(lhs, Symbol)
        throw(ArgumentError("Left-hand side (lhs) must be a valid column name as a symbol"))
    end

    # Handle case when rhs is mistakenly parsed as a Symbol (but should be a string)
    if isa(rhs, Symbol)
        rhs = string(rhs)
    elseif !(isa(rhs, Number) || isa(rhs, AbstractString) || (rhs.head == :vect) || (rhs.head == :tuple))
        throw(ArgumentError("Right-hand side (rhs) must be a number, string, or collection"))
    end

    # Define the mapping for vectorized operations
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

    # Ensure the operator is supported
    if !haskey(op_map, string(op))
        throw(ArgumentError("Unsupported operator: $op"))
    end

    # Convert lhs to a full DataFrame column reference wrapped in a QuoteNode
    lhs = Expr(:., :df, QuoteNode(lhs))

    # Map the operator to its vectorized form
    vectorized_op = Symbol(op_map[string(op)])

    # Wrap rhs in Ref if it's a collection to prevent broadcasting
    if (op == :in || op == Symbol("!in")) && ((rhs.head == :vect) || (rhs.head == :tuple))
        rhs = Expr(:call, :Ref, rhs)
    end

    # Return the transformed expression
    return Expr(:call, vectorized_op, lhs, rhs)
end

# Function to recursively transform a parsed expression with AND and OR logic
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
        return Meta.parse("df -> true")
    end

    parsed_expr = normalize_and_parse(input_str)
    if isa(parsed_expr, Symbol)
        return Meta.parse("df -> df[:, $(QuoteNode(parsed_expr))]")
    end

    # Handle case with a column selection (comma-separated)
    if parsed_expr.head == :tuple && length(parsed_expr.args) == 2
        # Split the parsed expression into column selection and condition
        column_expr, condition_expr = parsed_expr.args

        # Ensure the column_expr is a valid column name (symbol)
        if !(column_expr isa Symbol)
            throw(ArgumentError("Column selection must be a valid column name"))
        end

        # Transform the condition expression
        condition = transform_logical(condition_expr)

        # Build and return the final Expr
        return quote
            df -> begin
                hcat($condition, df[!, $(QuoteNode(column_expr))])
            end
        end
    else
        # If there's no column selection, handle as a simple condition
        condition = transform_logical(parsed_expr)

        # Build and return the final Expr
        return quote
            df -> begin
                $condition
            end
        end
    end
end

# # Example usage
# df = DataFrame(AREA = [1, 2, 3, 4], CLAVE = ["A", "B", "C", "D"], A = [0.6, 0.4, 0.9, 0.2])

# expr1 = parse_criteria("AREA == 1")
# result1 = eval(expr1)(df)
# println(result1)

# expr2 = parse_criteria("AREA in [1, 2, 3]")
# result2 = eval(expr2)(df)
# println(result2)

# expr3 = parse_criteria("CLAVE, AREA == 1")
# result3 = eval(expr3)(df)
# println(result3)

# expr4 = parse_criteria("AREA")
# result4 = eval(expr4)(df)
# println(result5)


# expr5 = parse_criteria("CLAVE in ['A', 'B']")
# result5 = eval(expr5)(df)
# println(result5)

# expr6 = parse_criteria("")
# result6 = eval(expr6)(df)
# println(result6)  # Should always print true
