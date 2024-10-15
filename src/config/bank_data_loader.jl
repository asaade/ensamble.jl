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

Reads the item file and returns a DataFrame, handling both dichotomous and polytomous items.
"""
function read_items_file(items_file::AbstractString)::DataFrame
    try
        bank = safe_read_csv(items_file)
        rename!(bank, uppercase.(names(bank)))
        bank = uppercase_dataframe!(bank)

        # Ensure MODEL_TYPE and NUM_CATEGORIES columns are present
        if "MODEL_TYPE" ∉ names(bank)
            bank.MODEL_TYPE .= "3PL"  # Default to dichotomous 3PL model if not present
        end
        if "NUM_CATEGORIES" ∉ names(bank)
            bank.NUM_CATEGORIES .= 2  # Default to dichotomous (2 categories) if not present
        end

        # Set missing A values to 1.0 for some models
        if "A" ∉ names(bank)
            bank.A .= 1.0
        else
            bank.A = coalesce.(bank.A, 1.0)  # Replace missing A with 1.0
        end

        # Set missing C values to 0.0 for Rasch-like models
        if "C" ∉ names(bank)
            bank.C .= 0.0
        else
            bank.C = coalesce.(bank.C, 0.0)  # Replace missing C with 0.0
        end

        # Convert thresholds (B1, B2, ...) to a vector of B values for polytomous items
        num_items = size(bank, 1)

        # Create a matrix for B parameters (polytomous items with thresholds)
        bank.B_THRESHOLDS = [Vector{Union{Float64, Missing}}() for _ in 1:num_items]
        for i in 1:num_items
            if bank.MODEL_TYPE[i] != "3PL"  # Assume 3PL is for dichotomous items, other values for polytomous models
                num_cat = bank.NUM_CATEGORIES[i]
                b_thresh = Vector{Union{Float64, Missing}}()  # Allow missing values
                for cat in 1:(num_cat - 1)
                    b_column = Symbol("B$cat")  # B1, B2, B3...
                    if b_column in names(bank)
                        push!(b_thresh, bank[i, b_column])
                    else
                        push!(b_thresh, missing)  # Handle missing threshold columns gracefully
                    end
                end
                bank.B_THRESHOLDS[i] = b_thresh
            end
        end

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
            error("Error loading anchor items: $e")
        end
    else
        @warn "Anchor file is empty or not provided."
        return DataFrame()  # Return an empty DataFrame if no anchor file is provided
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
    bank.ITEM_USE .= 0 # zeros(size(bank, 1))
    bank.ID = string.(bank.ID)
    bank.INDEX = rownumber.(eachrow(bank))
    return bank
end


"""
    select_valid_items!(bank::DataFrame)::DataFrame

Cleans the item bank based on IRT limits in A, B, and C or other polytomous parameters.
"""
function select_valid_items!(bank::DataFrame)::DataFrame
    original_size = size(bank, 1)
    dropmissing!(bank, [:B, :ID])

    # Fill missing A and C values for dichotomous items
    bank.A = coalesce.(bank.A, 1.0)
    bank.C = coalesce.(bank.C, 0.0)

    # Filter dichotomous items based on A, B, C limits
    filter!(row -> (0.4 <= row.A <= 2.0 && -3.5 <= row.B <= 3.5 && 0.0 <= row.C <= 0.5), bank)

    # Additional filtering for polytomous items based on the existence of B_THRESHOLDS
    filter!(row -> (row.MODEL_TYPE == "3PL" || (length(row.B_THRESHOLDS) == row.NUM_CATEGORIES - 1)), bank)

    invalid_items = original_size - size(bank, 1)
    invalid_items > 0 && @warn string(invalid_items, " invalid items in bank.")

    # Recompute the INDEX column after filtering
    # bank.INDEX = rownumber.(eachrow(bank))

    return bank
end



end # module BankDataLoader
