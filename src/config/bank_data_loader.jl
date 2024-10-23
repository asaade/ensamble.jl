module BankDataLoader

export read_bank_file

using DataFrames
using ..Utils

"""
    read_bank_file(items_file::String)::DataFrame

Reads the item bank file and returns a DataFrame.
"""
function read_bank_file(itemsfile::String, anchorsfile::String)::DataFrame
    bank = read_items_file(itemsfile)
    bank_size = size(bank, 1)
    @info "Loaded $bank_size items from $itemsfile"

    anchor = read_anchor_file(anchorsfile)
    anchor_forms = size(anchor, 2)
    @info "Loaded $anchor_forms anchor forms from $anchorsfile"

    # Add anchor labels and clean the bank
    bank = add_anchor_labels!(bank, anchor)
    bank = select_valid_items!(bank)
    bank = unique!(bank, :ID)
    bank = add_aux_labels!(bank)

    return bank
end

"""
    read_items_file(items_file::String)::DataFrame

Reads the item file and returns a DataFrame.
"""
function read_items_file(items_file::String)::DataFrame
    try
        bank = safe_read_csv(items_file)
        rename!(bank, uppercase.(names(bank)))
        bank = uppercase_dataframe!(bank)

        # Set missing A values to 1.0 for all models
        bank.A = coalesce.(bank.A, 1.0)

        # Set missing C values to 0.0 for all models
        bank.C = coalesce.(bank.C, 0.0)

        return bank
    catch e
        error("Error loading item bank: $e")
    end
end

"""
    read_anchor_file(anchor_file::String)::DataFrame

Reads the anchor items file and returns a DataFrame.
"""
function read_anchor_file(anchor_file::String)::DataFrame
    if !isempty(anchor_file)
        try
            anchor_data = safe_read_csv(anchor_file)
            rename!(anchor_data, uppercase.(names(anchor_data)))
            return uppercase_dataframe!(anchor_data)
        catch e
            error("Error loading anchor items: $e")
        end
    else
        @warn "Anchor file is empty or not provided."
        return DataFrame()  # Return an empty DataFrame if no anchor file is provided
    end
end

"""
    add_anchor_labels!(bank::DataFrame, anchor_tests::DataFrame)

Adds a column to the DataFrame with an ID number for the anchor test or missing.
"""
function add_anchor_labels!(bank::DataFrame, anchor_tests::DataFrame)::DataFrame
    bank.ANCHOR = Vector{Union{Missing, Int}}(missing, nrow(bank))

    for i in 1:size(anchor_tests, 2)
        dfv = view(bank, bank.ID .∈ Ref(anchor_tests[:, i]), :)
        @. dfv.ANCHOR = i
    end

    return bank
end

"""
    add_aux_labels!(bank::DataFrame)

Adds INDEX and ITEM_USE columns to the DataFrame to be used by some constraints later.
"""
function add_aux_labels!(bank::DataFrame)::DataFrame
    bank.ITEM_USE .= 0  # Initialize ITEM_USE column with zeros
    bank.INDEX = rownumber.(eachrow(bank))  # Assign index numbers
    return bank
end

"""
    select_valid_items!(bank::DataFrame)::DataFrame

Cleans the item bank based on IRT limits for A, B, and C.
"""
function select_valid_items!(bank::DataFrame)::DataFrame
    original_size = size(bank, 1)
    dropmissing!(bank, [:B, :ID])  # Remove rows with missing B or ID values

    # Fill missing A and C values for dichotomous items
    bank.A = coalesce.(bank.A, 1.0)
    bank.C = coalesce.(bank.C, 0.0)

    # Filter items based on A, B, C limits for dichotomous items
    filter!(row -> (0.4 <= row.A <= 2.0 && -3.5 <= row.B <= 3.5 && 0.0 <= row.C <= 0.5),
            bank)

    invalid_items = original_size - size(bank, 1)
    invalid_items > 0 &&
        @warn string(invalid_items, " invalid items were filtered out of the bank.")

    # Recompute the INDEX column after filtering
    bank.INDEX = rownumber.(eachrow(bank))

    return bank
end

end # module BankDataLoader
