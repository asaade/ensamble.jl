# src/config/config_loader.jl
module ConfigLoader

using ..ATAErrors
using ..ConfigValidation
using ..Utils

export BasicConfig, load_config

"""
Configuration structure for basic settings
"""
struct BasicConfig
    items_file::String
    anchor_items_file::String
    constraints_file::String
    forms_file::String
    results_file::String
    tcc_file::String
    plot_file::String
    solver::String
    verbose::Int
    report_categories::Vector{String}
    report_sums::Vector{String}

    # Inner constructor with validation
    function BasicConfig(items_file::String, anchor_file::String, constraints_file::String,
                        forms_file::String, results_file::String, tcc_file::String,
                        plot_file::String, solver::String, verbose::Int,
                        report_categories::Vector{String}, report_sums::Vector{String})
        # Validate required files exist
        for (name, file) in [("items", items_file), ("constraints", constraints_file)]
            if !isfile(file)
                throw(FilePathError(name, file, "File does not exist"))
            end
        end

        # Validate solver
        valid_solvers = ["CPLEX", "HIGHS", "GLPK", "CBC", "SCIP"]
        if !(uppercase(solver) in valid_solvers)
            throw(ValidationError("Invalid solver specified", "solver", solver))
        end

        # Validate verbose level
        if !(0 ≤ verbose ≤ 3)
            throw(ValidationError("Invalid verbose level", "verbose", verbose))
        end

        new(items_file, anchor_file, constraints_file, forms_file,
            results_file, tcc_file, plot_file, uppercase(solver),
            verbose, report_categories, report_sums)
    end
end

"""
Loads configuration from TOML data
"""
function load_config(config_data::Dict{Symbol,Any})::BasicConfig
    try
        # Validate complete configuration structure
        validate_config_data(config_data)

        files_data = config_data[:FILES]

        # Create configuration with defaults
        return BasicConfig(
            get(files_data, :ITEMSFILE, "items.csv"),
            get(files_data, :ANCHORFILE, ""),
            get(files_data, :CONSTRAINTSFILE, "constraints.csv"),
            get(files_data, :FORMSFILE, "results/forms.csv"),
            get(files_data, :RESULTSFILE, "results/results.csv"),
            get(files_data, :TCCFILE, "results/tcc_results.csv"),
            get(files_data, :PLOTFILE, "results/plot_output.png"),
            get(files_data, :SOLVER, "CPLEX"),
            get(files_data, :VERBOSE, 1),
            convert(Vector{String}, get(files_data, :REPORTCATEGORIES, String[])),
            convert(Vector{String}, get(files_data, :REPORTSUMS, String[]))
        )

    catch e
        if e isa ATAError
            rethrow(e)
        else
            throw(ConfigError(
                "Failed to load configuration",
                "load_config",
                e
            ))
        end
    end
end

end # module ConfigLoader
