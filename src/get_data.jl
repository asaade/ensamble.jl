
using DataFrames, YAML, CSV

include("stats_functions.jl")
include("types.jl")
include("utils.jl")

# Function to convert a dictionary to a Config struct
function Config(config_dict::Dict{Symbol, Any})
    forms_data = Dict(Symbol(k) => v for (k, v) in config_dict[:FORMS])
    return Config(forms_data,
                  config_dict[:ITEMSFILE],
                  config_dict[:ANCHORFILE],
                  config_dict[:ANCHORNUMBER],
                  config_dict[:VERSIONSFILE],
                  config_dict[:CONSTRAINTSFILE],
                  config_dict[:RESULTSFILE],
                  config_dict[:TCCFILE],
                  config_dict[:PLOTFILE],
                  config_dict[:SOLVER],
                  config_dict[:VERBOSE])
end

# Function to convert a dictionary to a Params struct
function Params(parms_dict::Dict{Symbol, Any})
    return Params(parms_dict[:N],
                  parms_dict[:F],
                  parms_dict[:MAXN],
                  # zeros(size(parms_dict[:BANK], 1)),
                  parms_dict[:F],
                  parms_dict[:K],
                  parms_dict[:R],
                  parms_dict[:SHADOWSIZE],
                  parms_dict[:METHOD],
                  parms_dict[:BANK],
                  parms_dict[:ANCHOR_NUMBER],
                  parms_dict[:THETA],
                  parms_dict[:P],
                  get(parms_dict, :INFO, nothing),
                  parms_dict[:TAU],
                  get(parms_dict, :TAU_INFO, nothing),
                  parms_dict[:RELATIVETARGETWEIGHTS],
                  parms_dict[:RELATIVETARGETPOINTS],
                  get(parms_dict, :DELTA, nothing),
                  parms_dict[:VERBOSE])
end

# Error handling for reading CSV files
function safe_read_csv(file_path::String)::DataFrame
    try
        return DataFrame(CSV.File(file_path))
    catch e
        println("Error reading CSV file: $file_path. Error: $e")
        return DataFrame()
    end
end

# Function to clean IRT data
function clean_IRT!(bank::DataFrame)::DataFrame
    dropmissing!(bank, :B)
    bank[!, :A] .= coalesce.(bank[!, :A], 1.0)
    bank[!, :C] .= coalesce.(bank[!, :C], 0.0)
    filter!(row -> (row.A >= 0.0 &&
                    row.A < 3.0 &&
                    row.B > -4.0 &&
                    row.B < 4.0 &&
                    row.C >= 0.0 &&
                    row.C < 0.5),
            bank)
    return bank
end

# Function to clean TC data
function clean_TC!(bank::DataFrame)::DataFrame
    dropmissing!(bank, :DIF)
    dropmissing!(bank, :CORR)
    if mean(bank.DIF) > 2.0
        bank.DIF = bank.DIF / 100
    end
    filter!(row -> (row.DIF > 0.1 && row.DIF < 0.9 && row.CORR >= 0.15 && row.CORR <= 0.65),
            bank)
    return bank
end

# Function to clean the items bank based on the selected method
function clean_items_bank!(config::Config, bank::DataFrame)::DataFrame
    original_size = size(bank, 1)
    unique!(bank)

    if !("CLAVE" in names(bank))
        bank[!, "CLAVE"] = collect(1:size(bank, 1))
    end

    method = config.forms[:METHOD]

    if method in ["TCC", "TIC", "TIC2", "TIC3"]
        clean_IRT!(bank)
    elseif method == "TC"
        clean_TC!(bank)
        # else
        #     error("Unknown method: $method")
    end

    println(original_size - size(bank, 1), " items in bank are invalid.")
    return bank
end

# Function to read anchor file and handle errors
function read_anchor_file(config::Config)::DataFrame
    anchor_data = safe_read_csv(config.anchor_items_file)
    upcase!(anchor_data)

    return anchor_data
end

# Function to add anchor labels to the bank
function add_anchor_labels!(config::Config, bank::DataFrame)
    anchor_items = read_anchor_file(config)
    anchor_versions = 0
    if !isempty(anchor_items)
        println("Loaded valid anchor data file")
        anchor_tests = size(anchor_items, 2)
        if anchor_tests > 0
            anchor_versions = min(config.anchor_number, anchor_tests)

            bank[!, "ANCHOR"] = fill(0, size(bank, 1))

            for i in 1:anchor_versions
                dfv = view(bank, bank.CLAVE .∈ Ref(anchor_items[:, i]), :)
                @. dfv.ANCHOR = i
            end
        end
    else
        println("Anchor file missing or empty")
    end
    return bank, anchor_versions
end

# Function to read bank file and handle errors
function read_bank_file(config::Config)::DataFrame
    bank = safe_read_csv(config.items_file)
    if !isempty(bank)
        println("Loaded valid bank data file")
    end
    rename!(bank, uppercase.(names(bank)))
    upcase!(bank)
    return bank
end

# Function to find total items in constraints file
function find_total_items(file_path::String)::Tuple{Int, Int}
    df = safe_read_csv(file_path)

    for row in eachrow(df)
        row = map(up!, row)
        if row[:TYPE] == "TEST"
            return row[:LB], row[:UB]
        end
    end

    return 0, 0
end

# Function to load configuration from a YAML file
function load_config(inFile::String = "data/config.yaml")::Config
    config_data = YAML.load_file(inFile; dicttype = Dict{Symbol, Any})
    config_dict = Dict(upSymbol(k) => v for (k, v) in config_data)
    config_dict[:FORMS] = Dict(upSymbol(k) => v for (k, v) in config_dict[:FORMS])
    lb, ub = find_total_items(config_dict[:CONSTRAINTSFILE])
    config_dict[:FORMS][:N] = (lb + ub) ÷ 2
    config_dict[:FORMS][:MINN] = lb
    config_dict[:FORMS][:MAXN] = max(lb, ub)

    if !haskey(config_dict[:FORMS], :R)
        config_dict[:FORMS][:R] = if config_dict[:FORMS][:N] <= 25
            4
        elseif config_dict[:FORMS][:N] <= 50
            3
        else
            2
        end
    end
    return Config(config_dict)
end

# Function to get parameters based on configuration
function get_params(config::Config)::Params
    bank = up!.(read_bank_file(config))
    bank, anchor_number = add_anchor_labels!(config, bank)
    bank = clean_items_bank!(config, bank)

    bank.CLAVE = string.(bank.CLAVE)
    bank.INDEX = rownumber.(eachrow(bank))
    parms_dict = deepcopy(config.forms)
    parms_dict[:BANK] = unique(bank, [:CLAVE])
    parms_dict[:ANCHOR_NUMBER] = anchor_number
    parms_dict[:VERBOSE] = config.verbose
    method = parms_dict[:METHOD]

    if method in ["TIC2", "TIC3"]
        theta = parms_dict[:RELATIVETARGETPOINTS]
    else
        theta = parms_dict[:THETA]
    end

    if method in ["TCC", "TIC", "TIC2", "TIC3"]
        parms_dict[:THETA] = theta
        a, b, c = bank[!, :A], bank[!, :B], bank[!, :C]
        parms_dict[:K] = length(theta)
        p = [Pr(t, b, a, c; d = 1.0) for t in theta]
        parms_dict[:P] = reduce(hcat, p)

        if haskey(parms_dict, :TAU)
            parms_dict[:TAU] = eval(Meta.parse(join(["[", join(parms_dict[:TAU], " "), "]"])))
        end
        parms_dict[:TAU] = get(parms_dict,
                               :TAU,
                               calc_tau(parms_dict[:P], parms_dict[:R], parms_dict[:K],
                                        parms_dict[:N], bank))
        parms_dict[:R] = min(parms_dict[:R], size(parms_dict[:TAU], 1))

        if method in ["TIC", "TIC2", "TIC3"]
            info = [item_information(t, b, a, c) for t in theta]
            parms_dict[:INFO] = reduce(hcat, info)
            parms_dict[:TAU_INFO] = get(parms_dict,
                                        :TAU_INFO,
                                        calc_info_tau(parms_dict[:INFO], parms_dict[:K],
                                                      parms_dict[:N]))
        end

        parms_dict[:DELTA] = nothing
    elseif method == "TC"
        # parms_dict[:DELTA] = delta(bank[!, :DIF], bank[!, :CORR])
    else
        error("Nonexistent method: $method")
    end

    return Params(parms_dict)
end

# Example usage:
# config = load_config("data/config.yaml")
# parms = get_params(config)
