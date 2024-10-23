module CriteriaParser

export parse_criteria

using DataFrames
using StringDistances

"""
    normalize(input_str::AbstractString) -> AbstractString

Converts the input string to uppercase and normalizes the operators by adding proper spacing
around them (e.g., `==`, `IN`, `&&`). This helps ensure that the condition is in a standard format
for further processing.

# Arguments

  - `input_str::AbstractString`: The input condition string to normalize.

# Returns

  - A normalized string with consistent formatting for operators.
"""
function normalize(input_str::AbstractString)::AbstractString
    return uppercase(replace(input_str, r"\s*([!=<>]=|!IN|IN|\&\&|\|\|)\s*" => s" \1 ",
                             r"\s*,\s*" => ","))
end

"""
    sanitize_input(input_str::AbstractString) -> AbstractString

Ensures that the input string contains only valid characters and throws an error
if invalid characters are found. If the input is empty, it returns it as is.

# Arguments

  - `input_str::AbstractString`: The input condition string to sanitize.

# Returns

  - The sanitized input string if valid; throws an error otherwise.
"""
function sanitize_input(input_str::AbstractString)::AbstractString
    if isempty(input_str)
        return input_str
    elseif !isvalidinput(input_str)
        throw(ArgumentError("Input string contains invalid characters"))
    end
    return input_str
end

"""
    isvalidinput(input_str::AbstractString) -> Bool

Checks whether the input string contains only valid characters. The allowed characters
include alphanumeric characters, comparison operators, logical operators, and basic punctuation.

# Arguments

  - `input_str::AbstractString`: The input string to validate.

# Returns

  - `true` if the string contains only valid characters; `false` otherwise.
"""
function isvalidinput(input_str::AbstractString)::Bool
    return match(r"^[a-zA-Z0-9\[\]\,\s\<\>\=\'\!\|\&\-\.]+$", input_str) !== nothing
end

# Operator map that defines how operators behave in condition expressions
const OPERATOR_MAP = Dict("!=" => (lhs, rhs) -> lhs .!= rhs,
                          "<" => (lhs, rhs) -> lhs .< rhs,
                          "<=" => (lhs, rhs) -> lhs .<= rhs,
                          "==" => (lhs, rhs) -> lhs .== rhs,
                          ">" => (lhs, rhs) -> lhs .> rhs,
                          ">=" => (lhs, rhs) -> lhs .>= rhs,
                          "IN" => (lhs, rhs) -> lhs .∈ Ref(rhs),
                          "!IN" => (lhs, rhs) -> lhs .∉ Ref(rhs))

"""
    validate_operator(op::AbstractString)

Checks whether the provided operator is valid. Throws an error if the operator
is not in the predefined `OPERATOR_MAP`.

# Arguments

  - `op::AbstractString`: The operator to validate.

# Returns

  - Nothing if the operator is valid; throws an error if invalid.
"""
function validate_operator(op::AbstractString)
    if op ∉ keys(OPERATOR_MAP)
        throw(ArgumentError("Unsupported operator: $op"))
    end
    return nothing
end

"""
    suggest_similar_column(input_column::Symbol, df::DataFrame) -> AbstractString

Suggests the closest matching column name from a DataFrame, using the Levenshtein distance
to find the most similar name.

# Arguments

  - `input_column::Symbol`: The column name provided in the query.
  - `df::DataFrame`: The DataFrame containing actual column names.

# Returns

  - The closest matching column name as a string.
"""
function suggest_similar_column(input_column::Symbol, df::DataFrame)::AbstractString
    existing_columns = Symbol.(names(df))
    distances = [levenshtein(string(input_column), string(col)) for col in existing_columns]
    closest_match = existing_columns[argmin(distances)]
    return string(closest_match)
end

"""
    apply_condition(op::AbstractString, lhs::Symbol, rhs, df::DataFrame) -> Vector{Bool}

Applies a condition to a specified column in a DataFrame. If the column does not exist,
suggests a similar column name. The operation is defined by the operator (e.g., `==`, `IN`).

# Arguments

  - `op::AbstractString`: The operator to apply (`==`, `!=`, etc.).
  - `lhs::Symbol`: The column name on the left-hand side of the condition.
  - `rhs`: The value(s) on the right-hand side of the condition.
  - `df::DataFrame`: The DataFrame to which the condition is applied.

# Returns

  - A boolean vector indicating whether each row satisfies the condition.
"""
function apply_condition(op::AbstractString, lhs::Symbol, rhs, df::DataFrame)
    if lhs ∉ Symbol.(names(df))
        similar_col = suggest_similar_column(lhs, df)
        throw(ArgumentError("Column $lhs does not exist in the DataFrame. Did you mean '$similar_col'?"))
    end
    lhs_col = df[!, lhs]
    return OPERATOR_MAP[op](lhs_col, rhs)
end

"""
    is_numeric(s::AbstractString) -> Bool

Determines whether a string represents a numeric value.

# Arguments

  - `s::AbstractString`: The input string to check.

# Returns

  - `true` if the string can be parsed as a number, `false` otherwise.
"""
function is_numeric(s::AbstractString)::Bool
    try
        parse(Float64, s)
        return true
    catch
        return false
    end
end

"""
    process_rhs(rhs::AbstractString) -> Union{Float64, String, Vector}

Processes the right-hand side of a condition. If it represents a list (e.g., `[1,2,3]`),
it returns a vector. Otherwise, it attempts to convert the value into a float or string.

# Arguments

  - `rhs::AbstractString`: The right-hand side of the condition expression.

# Returns

  - A vector if it's a list, a float if numeric, or a string otherwise.
"""
function process_rhs(rhs::AbstractString) #::Union{Float64, AbstractString}
    if match(r"^\[.*\]$", rhs) !== nothing
        elements = map(String, split(strip(rhs, ['[', ']']), ","))
        return [is_numeric(el) ? parse(Float64, el) : strip(el, ''') for el in elements]
    elseif is_numeric(rhs)
        return parse(Float64, rhs)
    else
        return strip(rhs, '"')
    end
end

"""
    parse_condition(condition_expr::AbstractString) -> Tuple{Symbol, String, Any}

Parses a condition expression into a tuple of its components: left-hand side column,
operator, and right-hand side value(s).

# Arguments

  - `condition_expr::AbstractString`: The condition expression (e.g., `AGE >= 30`).

# Returns

  - A tuple containing the column name, operator, and the processed right-hand side value(s).
"""
function parse_condition(condition_expr::AbstractString)::Tuple{Symbol, String, Any}
    parts = split(condition_expr, r"\s+")
    if length(parts) == 3
        lhs, op, rhs = parts
        validate_operator(op)
        return (Symbol(lhs), op, process_rhs(rhs))
    else
        throw(ArgumentError("Invalid condition format: $condition_expr"))
    end
end

"""
    handle_basic_condition(condition_expr::AbstractString) -> Function

Creates a function that applies a basic condition (e.g., `AGE >= 30`) to a DataFrame.

# Arguments

  - `condition_expr::AbstractString`: The condition expression.

# Returns

  - A function that, when given a DataFrame, applies the condition.
"""
function handle_basic_condition(condition_expr::AbstractString)::Function
    lhs, op, rhs = parse_condition(condition_expr)
    return df -> apply_condition(op, lhs, rhs, df)
end

"""
    handle_column_only(col_expr::AbstractString) -> Function

Creates a function to select a column from a DataFrame by name.

# Arguments

  - `col_expr::AbstractString`: The column name to select.

# Returns

  - A function that, when given a DataFrame, returns the column values.
"""
handle_column_only(col_expr::AbstractString)::Function = df -> df[!, Symbol(col_expr)]

"""
    split_outside_brackets(input_str::AbstractString) -> Tuple{AbstractString, AbstractString}

Splits the input string at the first comma found outside any brackets.
Useful for separating column names from conditions.

# Arguments

  - `input_str::AbstractString`: The input string to split.

# Returns

  - A tuple containing the part before and after the comma.
"""
function split_outside_brackets(input_str::AbstractString)::Tuple{AbstractString,
                                                                  AbstractString}
    level = 0
    split_pos = nothing

    for (i, c) in enumerate(input_str)
        if c == '['
            level += 1
        elseif c == ']'
            level -= 1
        elseif c == ',' && level == 0
            split_pos = i
            break
        end
    end

    if split_pos === nothing
        return input_str, ""
    else
        return input_str[1:(split_pos - 1)], input_str[(split_pos + 1):end]
    end
end

"""
    handle_column_and_condition(col_expr::AbstractString, condition_expr::AbstractString) -> Function

Creates a function that selects a column and applies a condition to it.

# Arguments

  - `col_expr::AbstractString`: The column to select.
  - `condition_expr::AbstractString`: The condition to apply.

# Returns

  - A function that applies the condition and selects the column from a DataFrame.
"""
function handle_column_and_condition(col_expr::AbstractString,
                                     condition_expr::AbstractString)::Function
    lhs, op, rhs = parse_condition(condition_expr)
    return df -> df[apply_condition(op, lhs, rhs, df), Symbol(col_expr)]
end

"""
    handle_logical_expression(expr::AbstractString) -> Function

Handles logical expressions like `&&` and `||`, combining multiple conditions.

# Arguments

  - `expr::AbstractString`: The logical expression to process.

# Returns

  - A function that applies the combined logical conditions to a DataFrame.
"""
function handle_logical_expression(expr::AbstractString)
    if contains(expr, "&&")
        conditions = split(expr, "&&")
        condition_funcs = [parse_criteria(strip(cond)) for cond in conditions]
        return df -> reduce((a, b) -> a .& b, [cond(df) for cond in condition_funcs])
    elseif contains(expr, "||")
        conditions = split(expr, "||")
        condition_funcs = [parse_criteria(strip(cond)) for cond in conditions]
        return df -> reduce((a, b) -> a .| b, [cond(df) for cond in condition_funcs])
    else
        return parse_criteria(expr)
    end
end

"""
    parse_criteria(input_str::AbstractString; max_length::Int=100) -> Function

Parses a criteria string into a function that can be applied to a DataFrame.
Handles conditions, logical operators, and column selection.

# Arguments

  - `input_str::AbstractString`: The criteria string to parse.
  - `max_length::Int`: The maximum allowed length of the input string.

# Returns

  - A function that applies the parsed criteria to a DataFrame.
"""
function parse_criteria(input_str::AbstractString; max_length::Int=100)::Function
    if length(input_str) > max_length
        throw(ArgumentError("Input string exceeds maximum allowed length of $max_length characters"))
    end

    input_str = sanitize_input(strip(input_str))

    if isempty(input_str)
        return df -> trues(size(df, 1))
    end

    normalized_str = normalize(input_str)

    if contains(normalized_str, "&&") || contains(normalized_str, "||")
        return handle_logical_expression(normalized_str)
    end

    col_expr, condition_expr = split_outside_brackets(normalized_str)
    if !isempty(condition_expr)
        return handle_column_and_condition(strip(col_expr), strip(condition_expr))
    else
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

end  # module CriteriaParser

# # Example usage
# df = DataFrame(CLAVE = [1, 2, 3, 4], AREA = ["A", "B", "C", "D"], A = [0.6, 0.4, 0.9, 0.2])

# expr1 = parse_criteria("CLAVE == 1")
# result1 = eval(expr1)(df)
# println(result1)

# expr2 = parse_criteria("CLAVE in [1, 2, 3]")
# result2 = eval(expr2)(df)
# println(result2)

# expr3 = parse_criteria("AREA, CLAVE >= 1")
# result3 = eval(expr3)(df)
# println(result3)

# expr4 = parse_criteria("AREA")
# result4 = eval(expr4)(df)
# println(result4)

# expr5 = parse_criteria("AREA in ['A', 'B']")
# result5 = eval(expr5)(df)
# println(result5)

# expr6 = parse_criteria("")
# result6 = eval(expr6)(df)
# println(result6)  # Should always print true
