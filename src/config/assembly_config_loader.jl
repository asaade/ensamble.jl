# src/config/assembly_config_loader.jl

module AssemblyConfigLoader

export AssemblyConfig, load_assembly_config

using CSV, DataFrames
using ..ATAErrors
using ..ConfigValidation
using ..Utils

"""
Configuration structure for test assembly parameters
"""
struct AssemblyConfig
    form_size::Int           # Size of each form (number of items)
    num_forms::Int          # Number of forms to assemble
    f::Int                  # Fixed number of forms to assemble
    max_items::Int          # Maximum number of items in a form
    anchor_tests::Int       # Number of anchor tests to include
    anchor_size::Int        # Size of anchor test section in each form
    max_item_use::Int       # Maximum number of times an item can be used across forms
    shadow_test_size::Int   # Size of the shadow test (for iterative algorithms)

    function AssemblyConfig(
            form_size::Int,
            num_forms::Int,
            f::Int,
            max_items::Int,
            anchor_tests::Int,
            anchor_size::Int,
            max_item_use::Int,
            shadow_test_size::Int
    )
        # Validate form size and related parameters
        if form_size <= 0
            throw(ValidationError(
                "Form size must be positive",
                "FORMS.form_size",
                form_size
            ))
        end

        if num_forms <= 0
            throw(ValidationError(
                "Number of forms must be positive",
                "FORMS.num_forms",
                num_forms
            ))
        end

        if max_items < form_size
            throw(ValidationError(
                "Maximum items must be greater than or equal to form size",
                "FORMS.max_items",
                Dict("max_items" => max_items, "form_size" => form_size)
            ))
        end

        if anchor_tests < 0
            throw(ValidationError(
                "Number of anchor tests cannot be negative",
                "FORMS.anchor_tests",
                anchor_tests
            ))
        end

        if max_item_use < 1
            throw(ValidationError(
                "Maximum item use must be at least 1",
                "FORMS.max_item_use",
                max_item_use
            ))
        end

        if shadow_test_size < 0
            throw(ValidationError(
                "Shadow test size cannot be negative",
                "FORMS.shadow_test_size",
                shadow_test_size
            ))
        end

        return new(form_size, num_forms, f, max_items, anchor_tests,
            anchor_size, max_item_use, shadow_test_size)
    end
end

"""
Extracts form size from constraints file and configuration
"""
function extract_form_size(
        constraints_df::DataFrame,
        default_size::Int
)::Tuple{Int, Int}
    try
        row_index = findfirst(
            r -> upcase(r[:ONOFF]) == "ON" && upcase(r[:TYPE]) == "TEST",
            eachrow(constraints_df)
        )

        if row_index !== nothing
            row = constraints_df[row_index, :]
            lb = get(row, :LB, default_size)
            ub = get(row, :UB, default_size)
            form_size = round(Int, (lb + ub) ÷ 2)
            return (form_size, ub)
        end

        return (default_size, default_size)
    catch e
        throw(ConfigError(
            "Failed to extract form size from constraints",
            "constraints",
            e
        ))
    end
end

"""
Validates and processes assembly configuration data
"""
function process_assembly_data(
        assembly_dict::Dict{Symbol, Any}
)::Dict{Symbol, Int}
    try
        return Dict{Symbol, Int}(
            :num_forms => get(assembly_dict, :NUMFORMS, 1),
            :max_item_use => get(
                assembly_dict, :MAXITEMUSE, get(assembly_dict, :NUMFORMS, 1)),
            :shadow_test_size => get(assembly_dict, :SHADOWTEST, 1),
            :anchor_tests => get(assembly_dict, :ANCHORTESTS, 0),
            :anchor_size => 0  # Default value as per original implementation
        )
    catch e
        throw(ConfigError(
            "Failed to process assembly data",
            "FORMS",
            e
        ))
    end
end

"""
Loads assembly configuration from configuration data
"""
function load_assembly_config(config_data::Dict{Symbol, Any})::AssemblyConfig
    try
        # Validate FORMS section existence
        if !haskey(config_data, :FORMS)
            throw(ConfigError(
                "Missing FORMS section",
                "Configuration",
                nothing
            ))
        end

        assembly_dict = config_data[:FORMS]
        default_form_size = get(assembly_dict, :N, 60)

        # Load and process constraints file
        constraints_file = get(config_data, :CONSTRAINTSFILE, "data/constraints.csv")
        constraints_df = safe_read_csv(constraints_file)
        uppercase_dataframe!(constraints_df)

        # Extract form size and maximum items
        form_size, max_items = extract_form_size(constraints_df, default_form_size)

        # Process assembly configuration
        assembly_params = process_assembly_data(assembly_dict)

        # Create and return AssemblyConfig
        return AssemblyConfig(
            form_size,
            assembly_params[:num_forms],
            assembly_params[:num_forms],  # f equals num_forms as per original
            max_items,
            assembly_params[:anchor_tests],
            assembly_params[:anchor_size],
            assembly_params[:max_item_use],
            assembly_params[:shadow_test_size]
        )

    catch e
        if e isa ATAError
            rethrow(e)
        else
            throw(ConfigError(
                "Failed to load assembly configuration",
                "load_assembly_config",
                e
            ))
        end
    end
end

end # module AssemblyConfigLoader
