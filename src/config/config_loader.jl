module ConfigLoader

export BasicConfig, load_config

# Define the Config struct (from types.jl)
mutable struct BasicConfig
    items_file::String             # Path to item bank file
    anchor_items_file::String      # Path to anchor items file (empty string if not used)
    constraints_file::String       # Path to constraints file
    forms_file::String             # Path to item bank file (output)
    results_file::String           # Path to results file
    tcc_file::String               # Path to TCC file
    plot_file::String              # Path to plot file
    solver::String                 # Solver type (e.g., "CPLEX", "HiGHS", etc.)
    verbose::Int                   # Verbosity level
end

"""
    load_config(config_data::Dict{Symbol, Any})::Config

Reads configuration from a TOML dictionary and returns a `Config` struct.

# Arguments

  - `config_data`: A dictionary containing the TOML configuration data.

# Returns

  - A `Config` struct with configuration settings loaded from the dictionary.
"""
function load_config(config_data::Dict{Symbol,Any})::BasicConfig
    # Load only the corresponding part from the dictionary

    if !haskey(config_data, :FILES)
        @error "Configuration file is missing the FILES section."
        throw(ArgumentError("Missing FILES section in the configuration"))
    end

    files_data = config_data[:FILES]

    # Extract necessary fields and provide defaults if needed
    items_file = get(files_data, :ITEMSFILE, "items.csv")
    anchor_file = get(files_data, :ANCHORFILE, "anchors.csv")
    constraints_file = get(files_data, :CONSTRAINTSFILE, "")
    forms_file = get(files_data, :FORMSFILE, "results/forms.csv")
    results_file = get(files_data, :RESULTSFILE, "results/results.csv")
    tcc_file = get(files_data, :TCCFILE, "results/tcc_results.csv")
    plot_file = get(files_data, :PLOTFILE, "results/combined_plots.pdf")
    solver = get(files_data, :SOLVER, "CPLEX")  # Default solver is "CPLEX"
    verbose = get(files_data, :VERBOSE, 1)      # Default verbosity is 1

    # Log the loaded configuration for debugging purposes
    @info "Loaded configuration: items_file = $items_file, anchor_file = $anchor_file, solver = $solver"

    # Return the Config struct
    return BasicConfig(items_file, anchor_file, constraints_file, forms_file, results_file, tcc_file,
        plot_file, solver, verbose)
end

end # module ConfigLoader
