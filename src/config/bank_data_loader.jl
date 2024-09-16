module BankDataLoader

using Main.DataFrames
# using Logging
using ..StringUtils

export read_bank_file

data_path = "data"

"""
    read_bank_file(items_file::AbstractString)::DataFrame

Reads the item bank file and returns a DataFrame.
"""
function read_bank_file(itemsfile::AbstractString, anchorsfile::AbstractString)
    bank = read_items_file(joinpath(data_path, itemsfile))
    bank_size = size(bank, 1)
    @info "Loaded $bank_size items from $itemsfile"
    anchor = read_anchor_file(joinpath(data_path, anchorsfile))
    anchor_forms = size(anchor, 2)
    @info "Loaded $anchor_forms anchor forms from $anchorsfile"
    add_anchor_labels!(bank, anchor)
    add_aux_labels!(bank)
    bank_items_before = bank.ID
    select_valid_items!(bank)
    bank_items_after = bank.ID
    items_removed = setdiff(bank_items_before, bank_items_after)
    length(items_removed) > 0 && @info "Removed $items_removed not valid item(s)"
    return bank
end

"""
    read_items_file(items_file::AbstractString)::DataFrame

Reads the item file and returns a DataFrame.
"""
function read_items_file(items_file::AbstractString)::DataFrame
    try
        bank = StringUtils.safe_read_csv(items_file)
        rename!(bank, uppercase.(names(bank)))
        bank = StringUtils.upcase.(bank)
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
            anchor_data = StringUtils.safe_read_csv(anchor_file)
            rename!(anchor_data, uppercase.(names(anchor_data)))
            anchor_data = StringUtils.upcase.(anchor_data)
            return anchor_data
        catch e
            error("Error loading item bank: $e")
        end
    end
end

"""
    add_anchor_labels!(bank::DataFrame, anchor_tests::DataFrame)

Adds a column to the DataFrame with and ID number for the anchor test or zero.
"""
function add_anchor_labels!(bank::DataFrame, anchor_tests::DataFrame)
    anchor_forms = size(anchor_tests, 2)
    bank[!, "ANCHOR"] = fill(0, size(bank, 1))

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
function add_aux_labels!(bank::DataFrame)
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
    bank[!, :A] .= coalesce.(bank[!, :A], 1.0)
    bank[!, :C] .= coalesce.(bank[!, :C], 0.0)
    filter!(row -> (0.4 <= row.A <= 2.0 && -4.0 <= row.B <= 4.0 && 0.0 <= row.C <= 0.5),
            bank)
    invalid_items = original_size - size(bank, 1)
    invalid_items > 0 && @warn string(invalid_items, " invalid items in bank.")
    return bank
end

end # module BankDataLoader
