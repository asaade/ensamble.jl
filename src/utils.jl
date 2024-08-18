"""
Some utility functions, mostly adapted from the Advent of Code
utilities by Peter Norvig
"""

isiterable(f) = applicable(foreach, f)
isnumeric(s)  = tryparse(Float64, s) !== nothing
isint(s)      = tryparse(Int64, s) !== nothing
to_int(s) = parse(Int64, s)


# char = String # Intended as the type of a one-character string
Atom = Union{String, Float64, Int64} # The type of a string or number

##"""A tuple of all the integers in text, ignoring non-number characters."""
ints(text) = map(x -> tryparse(Int64, x.match), eachmatch(r"-?\d+", text))

## """A tuple of all the integers in text, ignoring non-number characters.## """
pos_ints(text) = map(x -> tryparse(Int64, x.match), eachmatch(r"[0-9]+", text))

## """A tuple of all the digits in text (as ints 0–9), ignoring non-digit characters.## """
digits(text) = map(x -> tryparse(Int64, x.match), eachmatch(r"[0-9]", text))

## """A tuple of all the alphabetic words in text, ignoring non-letters.## """
words(text) = map(x -> string(x.match), eachmatch(r"\w+", text))

function atom(text)
    ## """Parse text into a single float or int or str.## """
    try
        r = parse(Float64, text)
        return isinteger(r) >= 0 ? Int(r) : r
    catch _
        return(String(strip(text)))
    end
end

function atom_constraint(text)::String
    ## """Parse text into a single float or int or str.## """
    try
        r = parse(Float64, text)
        if isinteger(r) r = Int(r) end
        return r
    catch _
        r = string("\"", text, "\"")::String
        return(r)
    end
end

function atoms(text)
    ## """A tuple of all the atoms (numbers or identifiers) in text. Skip punctuation.## """
    return map(x -> atom(x.match), eachmatch(r"[+-]?\d+\.?\d*|\w+", strip(text)))
end

function cover(coll)
    ## """A `range` that covers all the given integers, and any in between them.
    ## cover(lo, hi) is a an inclusive (or closed) range, equal to range(lo, hi + 1)."""
    return (min(coll...):max(coll...))
end

function the(seq)
    ## """Return the one item in a sequence. Raise error if not exactly one."""
    return seq[1]
end

upSymbol(y::Symbol) = Symbol(uppercase(string(y)))


function upcase!(df::DataFrame)
    _helper(c) = nothing
    _helper(c::AbstractVector{T} where T<:AbstractString) = c .= strip.(uppercase.(c))
    foreach(_helper, eachcol(df))
    return df
end


up!(x) = try uppercase(strip(x)) catch _ return x end
