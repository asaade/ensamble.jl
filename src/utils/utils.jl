module Utils

export upSymbol,
       upcase,
       upcaseKeys,
       safe_read_csv,
       safe_read_csv,
       safe_read_toml,
       cleanValues,
       uppercase_dataframe!,
        calc_tau,
       calc_info_tau,
       prob_3pl,
       info_3pl,
       observed_score_continuous,
       lw_dist,
       expected_score_item,
       expected_score_matrix,
       expected_info_matrix,
       calc_scores_reference

export ensure_dir, safe_read_csv, safe_read_toml

include("string_utils.jl")      # Auxiliary string manipulation methods
include("stats_functions.jl")   # Auxiliary Statistical and IRT methods
include("file_utils.jl")
using .StringUtils
using .StatsFunctions
# using .FileUtils

end
