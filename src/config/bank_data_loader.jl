# src/data/bank_data_loader.jl

module BankDataLoader

export read_bank_file, BankLoadError

using DataFrames
using ..Utils
using ..ATAErrors
using ..ConfigValidation

"""
Custom errors for bank loading operations
"""
struct BankLoadError <: ATAError
    message::String
    operation::String
    details::Any
end

"""
Reads and processes the item bank file
"""
function read_items_file(items_file::String)::DataFrame
    try
        # Read and standardize column names
        bank = safe_read_csv(items_file)
        rename!(bank, uppercase.(names(bank)))
        bank = uppercase_dataframe!(bank)

        # Validate required columns
        validate_bank_columns(bank)

        # Set default values for A and C parameters
        bank.A = coalesce.(bank.A, 1.0)
        bank.C = coalesce.(bank.C, 0.0)

        return bank
    catch e
        if e isa ATAError
            rethrow(e)
        else
            throw(BankLoadError(
                "Failed to load item bank",
                "read_items_file",
                e
            ))
        end
    end
end

"""
Reads and processes the anchor items file
"""
function read_anchor_file(anchor_file::String)::DataFrame
    if isempty(anchor_file)
        @info "No anchor file provided, proceeding without anchor items"
        return DataFrame()
    end

    try
        anchor_data = safe_read_csv(anchor_file)
        rename!(anchor_data, uppercase.(names(anchor_data)))
        return uppercase_dataframe!(anchor_data)
    catch e
        if e isa ATAError
            rethrow(e)
        else
            throw(BankLoadError(
                "Failed to load anchor items",
                "read_anchor_file",
                e
            ))
        end
    end
end

"""
Adds anchor test labels to the item bank
"""
function add_anchor_labels!(bank::DataFrame, anchor_tests::DataFrame)::DataFrame
    try
        bank.ANCHOR = Vector{Union{Missing, Int}}(missing, nrow(bank))

        for i in 1:ncol(anchor_tests)  # Using ncol() from DataFrames.jl
            dfv = view(bank, bank.ID .∈ Ref(anchor_tests[:, i]), :)
            @. dfv.ANCHOR = i
        end

        return bank
    catch e
        throw(BankLoadError(
            "Failed to add anchor labels",
            "add_anchor_labels",
            e
        ))
    end
end

"""
Adds auxiliary labels (INDEX and ITEM_USE) to the item bank
"""
function add_aux_labels!(bank::DataFrame)::DataFrame
    try
        bank.ITEM_USE .= 0
        bank.INDEX = rownumber.(eachrow(bank))
        return bank
    catch e
        throw(BankLoadError(
            "Failed to add auxiliary labels",
            "add_aux_labels",
            e
        ))
    end
end

"""
Selects valid items based on IRT parameter limits
"""
function select_valid_items!(
        bank::DataFrame, limits::IRTLimits = DEFAULT_IRT_LIMITS)::DataFrame
    try
        original_size = nrow(bank)

        # Validate and log invalid items
        invalid_items = validate_irt_parameters(bank)

        if !isempty(invalid_items)
            @warn "Found $(nrow(invalid_items)) items with invalid IRT parameters or missing ID"
            @debug "Invalid items:" invalid_items
        end

        # Remove rows with missing essential values
        dropmissing!(bank, [:ID, :CHECK])

        # Log filtering results
        items_removed = original_size - nrow(bank)
        if items_removed > 0
            @info "Removed $items_removed items with invalid parameters"
        end

        # Update indices
        bank.INDEX = rownumber.(eachrow(bank))

        return bank
    catch e
        throw(BankLoadError(
            "Failed to select valid items",
            "select_valid_items",
            e
        ))
    end
end

"""
Reads and processes the complete item bank with anchor items
"""
function read_bank_file(itemsfile::String, anchorsfile::String)::DataFrame
    try
        # Load main item bank
        bank = read_items_file(itemsfile)
        @info "Loaded $(nrow(bank)) items from $itemsfile"

        # Load anchor items if provided
        anchor = read_anchor_file(anchorsfile)
        if !isempty(anchor)
            @info "Loaded $(size(anchor, 2)) anchor forms from $anchorsfile"
        end

        # Process the bank
        bank = add_anchor_labels!(bank, anchor)
        bank = select_valid_items!(bank)
        bank = unique!(bank, :ID)
        bank = add_aux_labels!(bank)

        @info "Final bank contains $(nrow(bank)) valid items"
        return bank

    catch e
        if e isa ATAError
            rethrow(e)
        else
            throw(BankLoadError(
                "Failed to process item bank",
                "read_bank_file",
                e
            ))
        end
    end
end

end # module BankDataLoader
