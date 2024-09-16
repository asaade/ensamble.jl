module CriteriaParser

export parse_criteria

using DataFrames

# Normalize input string
function normalize(input_str::AbstractString)::AbstractString
    return uppercase(replace(input_str, r"\s*([!=<>]=?|in|\&\&|\|\|)\s*" => s" \1 ",
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

# Apply condition on a DataFrame
function apply_condition(op::AbstractString, lhs::Symbol, rhs, df::DataFrame)
    if lhs ∉ Symbol.(names(df))
        throw(ArgumentError("Column $lhs does not exist in the DataFrame"))
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

# Handle basic condition
function handle_basic_condition(condition_expr::AbstractString)
    lhs, op, rhs = parse_condition(condition_expr)
    return df -> apply_condition(op, lhs, rhs, df)
end

# Handle column-only selection
handle_column_only(col_expr::AbstractString) = df -> df[!, Symbol(col_expr)]

# Parse individual condition
function parse_condition(condition_expr::AbstractString)::Tuple{Symbol, String, Any}
    parts = split(condition_expr, r"\s+")
    if length(parts) == 3
        lhs, op, rhs = parts
        validate_operator(op)
        return (Symbol(lhs), op, process_rhs(rhs))
    else
        throw(ArgumentError("Invalid condition format"))
    end
end

# Handle column selection with condition
function handle_column_and_condition(col_expr::AbstractString,
                                     condition_expr::AbstractString)
    lhs, op, rhs = parse_condition(condition_expr)
    return df -> df[apply_condition(op, lhs, rhs, df), Symbol(col_expr)]
end

# Main function to parse criteria
function parse_criteria(input_str::AbstractString; max_length::Int = 100)
    if length(input_str) > max_length
        throw(ArgumentError("Input string exceeds maximum allowed length of $max_length characters"))
    end

    input_str = sanitize_input(strip(input_str))

    # Return all `true` if input is empty
    if isempty(input_str)
        return df -> trues(size(df, 1))
    end

    normalized_str = normalize(input_str)

    # Check for column selection with a condition
    if contains(normalized_str, ",") && match(r"\[.*\]", normalized_str) === nothing
        col_expr, condition_expr = split(normalized_str, ","; limit = 2)

        # Handle conditions like "ID, AREA in [2]"
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
