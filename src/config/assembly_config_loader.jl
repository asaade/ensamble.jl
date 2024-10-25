module AssemblyConfigLoader

export AssemblyConfig, load_assembly_config

using CSV, DataFrames

using ..Utils

# Define the AssemblyConfig struct
mutable struct AssemblyConfig
    form_size::Int           # Size of each form (number of items)
    num_forms::Int           # Number of forms to assemble
    f::Int                   # Fixed number of forms to assemble
    max_items::Int           # Maximum number of items in a form
    operational_items::Int   # Number of operational items in the bank
    anchor_tests::Int        # Number of anchor tests to include
    anchor_size::Int         # Size of anchor test section in each form
    max_item_use::Int        # Maximum number of times an item can be used across forms
    shadow_test_size::Int         # Size of the shadow test (for iterative algorithms)
end

"""
    load_assembly_config(config_data::Dict{Symbol, Any})::AssemblyConfig

Reads the test assembly configuration from the provided dictionary and returns an `AssemblyConfig` struct.

# Arguments

  - `config_data`: A dictionary containing the assembly configuration data.

# Returns

  - An `AssemblyConfig` struct with the test assembly configuration.
"""
function load_assembly_config(config_data::Dict{Symbol, Any})::AssemblyConfig
    # Check if the FORMS section exists

    if !haskey(config_data, :FORMS)
        @error "Configuration file is missing the FORMS section."
        throw(ArgumentError("Missing FORMS section in the configuration"))
    end

    assembly_dict = config_data[:FORMS]

    constraints_file = get(config_data, :CONSTRAINTSFILE, "data/constraints.csv")
    form_size = get(assembly_dict, :N, 60)

    df = safe_read_csv(constraints_file)
    uppercase_dataframe!(df)

    row_index = findfirst(r -> upcase(r[:ONOFF]) == "ON" && upcase(r[:TYPE]) == "TEST",
                          eachrow(df))

    if row_index !== nothing
        row = df[row_index, :]
        lb = get(row, :LB, form_size)
        ub = get(row, :UB, form_size)
        form_size = round((lb + ub) ÷ 2)
        max_items = ub
    end

    # Extract necessary values from the FORMS section and apply defaults where needed
    num_forms = get(assembly_dict, :NUMFORMS, 1)
    f = num_forms  # Assuming the fixed number of forms is the same as num_forms
    max_item_use = get(assembly_dict, :MAXITEMUSE, num_forms)  # Default to forms if not provided
    shadow_test_size = get(assembly_dict, :SHADOWTEST, 1)   # Default to 1 if not provided
    anchor_tests = get(assembly_dict, :ANCHORTESTS, 0) # Default to 0 if not provided

    # Initialize other fields with default values
    operational_items = 0
    anchor_size = 0

    # Log the loaded configuration for debugging
    @info "Loaded assembly configuration: form_size = $form_size, num_forms = $num_forms, anchor_tests = $anchor_tests"

    # Return the AssemblyConfig struct
    return AssemblyConfig(form_size,
                          num_forms,
                          f,
                          max_items,
                          operational_items,
                          anchor_tests,
                          anchor_size,
                          max_item_use,
                          shadow_test_size)
end

end # module AssemblyConfigLoader
