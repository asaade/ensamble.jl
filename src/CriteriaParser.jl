module CriteriaParser
export parse_criteria

using DataFrames

# Normalize input string
function normalize(input_str::AbstractString)
    return uppercase(replace(input_str, r"\s*([!=<>]=?|in|\&|\|)\s*" => s" \1 ", r"\s*,\s*" => ","))
end

function sanitize_input(input_str::AbstractString)
    # Only allow alphanumeric, spaces, and certain symbols
    if isempty(input_str)
        return input_str
    elseif !isvalidinput(input_str)
        throw(ArgumentError("Input string contains invalid characters"))
    end
    return input_str
end

function isvalidinput(input_str::AbstractString)
    return occursin(r"^[a-zA-Z0-9\[\]\,\s\<\>\=\!\|&\-\.]+$", input_str)
end


# Refactored condition application using a dictionary for operator functions
const OPERATOR_MAP = Dict(
    "!=" => (lhs, rhs) -> lhs .!= rhs,
    "<"  => (lhs, rhs) -> lhs .< rhs,
    "<=" => (lhs, rhs) -> lhs .<= rhs,
    "==" => (lhs, rhs) -> lhs .== rhs,
    ">"  => (lhs, rhs) -> lhs .> rhs,
    ">=" => (lhs, rhs) -> lhs .>= rhs,
    "IN" => (lhs, rhs) -> lhs .∈ Ref(rhs),
    "!IN" => (lhs, rhs) -> lhs .∉ Ref(rhs)
)

function validate_operator(op::AbstractString)
    if op ∉ keys(OPERATOR_MAP)
        throw(ArgumentError("Unsupported operator: $op"))
    end
end

# Apply condition on DataFrame
function apply_condition(op::AbstractString, lhs::Symbol, rhs, df::DataFrame)
    if lhs ∉ Symbol.(names(df))
        throw(ArgumentError("Column $lhs does not exist in the DataFrame"))
    end
    lhs_col = df[!, lhs]
    return OPERATOR_MAP[op](lhs_col, rhs)
end

# Check if string is numeric
is_numeric(s::AbstractString) =
    try
        parse(Float64, s)
        return true
    catch
        return false
    end

# Process the right-hand side (rhs)
function process_rhs(rhs::AbstractString)
    if occursin(r"^\[.*\]$", rhs)
        elements = split(strip(rhs, ['[', ']']), ",")
        return [is_numeric(el) ? parse(Float64, el) : strip(el, '"') for el in elements]
    elseif is_numeric(rhs)
        return parse(Float64, rhs)
    else
        return strip(rhs, '"')
    end
end

# Parse individual condition (e.g., AREA == 1)
function parse_condition(condition_expr::AbstractString)
    parts = split(condition_expr, r"\s+")
    if length(parts) == 3
        lhs, op, rhs = parts
        return (Symbol(lhs), String(op), process_rhs(rhs))
    else
        throw(ArgumentError("Invalid condition format"))
    end
end

# Handle column selection with a condition
function handle_column_and_condition(col_expr::AbstractString, condition_expr::AbstractString)
    lhs, op, rhs = parse_condition(condition_expr)
    validate_operator(op)
    return df -> df[apply_condition(op, Symbol(lhs), rhs, df), Symbol(col_expr)]
end

# Handle basic condition
function handle_basic_condition(condition_expr::AbstractString)
    lhs, op, rhs = parse_condition(condition_expr)
    validate_operator(op)
    return df -> apply_condition(op, Symbol(lhs), rhs, df)
end

# Handle column-only selection (no condition)
function handle_column_only(col_expr::AbstractString)
    return df -> df[!, Symbol(col_expr)]
end

# Main function to parse the entire criteria string
function parse_criteria(input_str::AbstractString; max_length::Int=100)
    if length(input_str) > max_length
        throw(ArgumentError("Input string exceeds maximum allowed length of $max_length characters"))
    end

    input_str = strip(input_str)
    input_str = sanitize_input(input_str)

    # Return all `true` if input is empty
    if isempty(input_str)
        return df -> trues(size(df, 1))
    end

    normalized_str = normalize(input_str)

    # Check for column selection with a condition
    if contains(normalized_str, r",") && !occursin(r"\[.*\]", normalized_str)
        col_expr, condition_expr = split(normalized_str, ","; limit = 2)
        return handle_column_and_condition(strip(col_expr), strip(condition_expr))
    else
        # Check if it's a basic condition or just a column
        parts = split(normalized_str, r"\s+")
        if length(parts) == 3
            return handle_basic_condition(normalized_str)
        elseif length(parts) == 1
            return handle_column_only(normalized_str)
        else
            throw(ArgumentError("Invalid criteria string format"))
        end
    end
end

end

# # # Example usage
# using .CriteriaParser
# using DataFrames

# df = DataFrame(AREA = [1, 2, 3, 4], ID = ["A", "B", "C", "D"], A = [0.6, 0.4, 0.9, 0.2])

# # Test cases
# expr1 = CriteriaParser.parse_criteria("AREA == 1")
# result1 = expr1(df)
# println(result1)  # Expected: Bool[1, 0, 0, 0]

# expr1_1 = CriteriaParser.parse_criteria("ID == A")
# result1_1 = expr1_1(df)
# println(result1_1)  # Expected: Bool[1, 0, 0, 0]

# expr2 = CriteriaParser.parse_criteria("AREA in [1, 2, 3]")
# result2 = expr2(df)
# println(result2)  # Expected: Bool[1, 1, 1, 0]

# expr3 = CriteriaParser.parse_criteria("ID, AREA == 1")
# result3 = expr3(df)
# println(result3)  # Expected: the values in ID where AREA == 1 (here is "A")

# expr4 = CriteriaParser.parse_criteria("AREA")
# result4 = expr4(df)
# println(result4)  # Expected: df[!, :AREA]

# expr5 = CriteriaParser.parse_criteria("ID in [A, D]")
# result5 = expr5(df)
# println(result5)  # Expected: Bool[1, 0, 0, 1]

## ## Not available for now
# # expr5_5 = CriteriaParser.parse_criteria("ID in [\"A\", \"D\"]")
# # result5_5 = expr5_5(df)
# # println(result5_5)  # Expected: Bool[1, 0, 0, 1]

# expr6 = CriteriaParser.parse_criteria("")
# result6 = expr6(df)
# println(result6)  # Expected: Bool[true, true, true, true]
