module BankDataLoader
export read_bank_file

using DataFrames

using ..Utils

"""
    read_bank_file(items_file::AbstractString)::DataFrame

Reads the item bank file and returns a DataFrame.
"""
function read_bank_file(itemsfile::AbstractString, anchorsfile::AbstractString)::DataFrame
    bank = read_items_file(itemsfile)
    bank_size = size(bank, 1)
    @info "Loaded $bank_size items from $itemsfile"
    anchor = read_anchor_file(anchorsfile)
    anchor_forms = size(anchor, 2)
    @info "Loaded $anchor_forms anchor forms from $anchorsfile"
    bank = add_anchor_labels!(bank, anchor)
    bank = select_valid_items!(bank)
    bank = unique!(bank, :ID)
    bank = add_aux_labels!(bank)
    return bank
end

"""
    read_items_file(items_file::AbstractString)::DataFrame

Reads the item file and returns a DataFrame.
"""
function read_items_file(items_file::AbstractString)::DataFrame
    try
        bank = safe_read_csv(items_file)
        rename!(bank, uppercase.(names(bank)))
        bank = uppercase_dataframe!(bank)
        return bank
    catch e
        error("Error loading item bank: $e")
    end
end

"""
    read_anchor_file(anchor_file::AbstractString)::DataFrame

Reads the anchor items file and returns a DataFrame.
"""
function read_anchor_file(anchor_file::AbstractString)::DataFrame
    if !isempty(anchor_file)
        try
            anchor_data = safe_read_csv(anchor_file)
            rename!(anchor_data, uppercase.(names(anchor_data)))
            return uppercase_dataframe!(anchor_data)
        catch e
            error("Error loading item bank: $e")
        end
    end
end

"""
    add_anchor_labels!(bank::DataFrame, anchor_tests::DataFrame)

Adds a column to the DataFrame with and ID number for the anchor test or zero.
"""
function add_anchor_labels!(bank::DataFrame, anchor_tests::DataFrame)::DataFrame
    anchor_forms = size(anchor_tests, 2)
    bank.ANCHOR = Vector{Union{Missing, Int}}(missing, nrow(bank))

    for i in 1:anchor_forms
        dfv = view(bank, bank.ID .∈ Ref(anchor_tests[:, i]), :)
        @. dfv.ANCHOR = i
    end
    return bank
end

"""
    add_aux_labels!(bank::DataFrame)

Adds ID, INDEX and ITEM_USE columns to the DataFrame to be used by some constraints later.
"""
function add_aux_labels!(bank::DataFrame)::DataFrame
    bank.ITEM_USE = zeros(size(bank, 1))
    bank.ID = string.(bank.ID)
    bank.INDEX = rownumber.(eachrow(bank))
    return bank
end

"""
    select_valid_items!(config::Config, bank::DataFrame)::DataFrame

Cleans the item bank based on IRT limits in a, b, and ceor Classical.
"""
function select_valid_items!(bank::DataFrame)::DataFrame
    original_size = size(bank, 1)
    dropmissing!(bank, [:B, :ID])
    # unique!(bank, :ID)
    bank.A = coalesce.(bank.A, 1.0)
    bank.C = coalesce.(bank.C, 0.0)
    filter!(row -> (0.4 <= row.A <= 2.0 && -3.5 <= row.B <= 3.5 && 0.0 <= row.C <= 0.5), bank)
    invalid_items = original_size - size(bank, 1)
    invalid_items > 0 && @warn string(invalid_items, " invalid items in bank.")
    return bank
end

end # module BankDataLoader
