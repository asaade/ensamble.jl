module CriteriaParser

using DataFrames
using StringDistances

# Normalize input string
function normalize(input_str::AbstractString)::AbstractString
    return uppercase(replace(input_str, r"\s*([!=<>]=|!IN|IN|\&\&|\|\|)\s*" => s" \1 ",
                             r"\s*,\s*" => ","))
end

# Sanitize input and ensure only valid characters
function sanitize_input(input_str::AbstractString)::AbstractString
    if isempty(input_str)
        return input_str
    elseif !isvalidinput(input_str)
        throw(ArgumentError("Input string contains invalid characters"))
    end
    return input_str
end

function isvalidinput(input_str::AbstractString)::Bool
    return match(r"^[a-zA-Z0-9\[\]\,\s\<\>\=\!\|\&\-\.]+$", input_str) !== nothing
end

# Operator map
const OPERATOR_MAP = Dict("!=" => (lhs, rhs) -> lhs .!= rhs,
                          "<" => (lhs, rhs) -> lhs .< rhs,
                          "<=" => (lhs, rhs) -> lhs .<= rhs,
                          "==" => (lhs, rhs) -> lhs .== rhs,
                          ">" => (lhs, rhs) -> lhs .> rhs,
                          ">=" => (lhs, rhs) -> lhs .>= rhs,
                          "IN" => (lhs, rhs) -> lhs .∈ Ref(rhs),
                          "!IN" => (lhs, rhs) -> lhs .∉ Ref(rhs))

function validate_operator(op::AbstractString)
    if op ∉ keys(OPERATOR_MAP)
        throw(ArgumentError("Unsupported operator: $op"))
    end
end

# Function to find the closest matching column name
function suggest_similar_column(input_column::Symbol, df::DataFrame)::AbstractString
    existing_columns = Symbol.(names(df))
    distances = [levenshtein(string(input_column), string(col)) for col in existing_columns]
    closest_match = existing_columns[argmin(distances)]
    return string(closest_match)
end

# Apply condition on a DataFrame
function apply_condition(op::AbstractString, lhs::Symbol, rhs, df::DataFrame)
    if lhs ∉ Symbol.(names(df))
        similar_col = suggest_similar_column(lhs, df)
        throw(ArgumentError("Column $lhs does not exist in the DataFrame. Did you mean '$similar_col'?"))
    end
    lhs_col = df[!, lhs]
    return OPERATOR_MAP[op](lhs_col, rhs)
end

# Check if string is numeric
function is_numeric(s::AbstractString)::Bool
    try
        parse(Float64, s)
        return true
    catch
        return false
    end
end

# Process the right-hand side (rhs)
function process_rhs(rhs::AbstractString)
    if match(r"^\[.*\]$", rhs) !== nothing
        elements = split(strip(rhs, ['[', ']']), ",")
        return [is_numeric(el) ? parse(Float64, el) : strip(el, '"') for el in elements]
    elseif is_numeric(rhs)
        return parse(Float64, rhs)
    else
        return strip(rhs, '"')
    end
end

# Parse individual condition
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

# Handle basic condition
function handle_basic_condition(condition_expr::AbstractString)
    lhs, op, rhs = parse_condition(condition_expr)
    return df -> apply_condition(op, lhs, rhs, df)
end

# Handle column-only selection
handle_column_only(col_expr::AbstractString) = df -> df[!, Symbol(col_expr)]

# Split the string at the first comma outside of brackets
function split_outside_brackets(input_str::AbstractString)::Tuple{AbstractString,
                                                                  AbstractString}
    level = 0  # Track bracket nesting
    split_pos = nothing

    for i in 1:length(input_str)
        c = input_str[i]

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

# Handle column selection with condition
function handle_column_and_condition(col_expr::AbstractString,
                                     condition_expr::AbstractString)
    lhs, op, rhs = parse_condition(condition_expr)
    return df -> df[apply_condition(op, lhs, rhs, df), Symbol(col_expr)]
end

# Handle logical expressions: parse and combine using && and ||
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

# Main function to parse criteria
function parse_criteria(input_str::AbstractString; max_length::Int = 100)
    if length(input_str) > max_length
        throw(ArgumentError("Input string exceeds maximum allowed length of $max_length characters"))
    end

    input_str = sanitize_input(strip(input_str))

    if isempty(input_str)
        return df -> trues(size(df, 1))
    end

    normalized_str = normalize(input_str)

    # Check for logical operators (&& or ||)
    if contains(normalized_str, "&&") || contains(normalized_str, "||")
        return handle_logical_expression(normalized_str)
    end

    # Check for column selection with a condition
    col_expr, condition_expr = split_outside_brackets(normalized_str)
    if !isempty(condition_expr)
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

end  # module CriteriaParser
