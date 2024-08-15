# Define a struct for the configuration
MatrixOrMissing = Union{Matrix{AbstractFloat}, Nothing}
DataFrameOrMissing = Union{DataFrame, Nothing}
VectorOrMissing = Union{Vector{AbstractFloat}, Nothing}


# Define a struct for the configuration
struct Config
    forms::Dict{Symbol, Any}
    items_file::String
    anchor_items_file::Union{String, Missing}
    anchor_number::Int
    versions_file::String
    constraints_file::String
    results_file::String
    tcc_file::String
    plot_file::String
    solver::String
end

# Define a struct for the parameters
mutable struct Params
    n::Int
    num_forms::Int
    max_items::Int
    f::Int
    k::Int
    r::Int
    shadow_test_size::Int
    method::String
    bank::DataFrame
    anchor_number::Int
    theta::VectorOrMissing
    p::MatrixOrMissing
    info::MatrixOrMissing
    tau::MatrixOrMissing
    tau_info::VectorOrMissing
    relative_target_weights::VectorOrMissing
    relative_target_points::VectorOrMissing
    delta::MatrixOrMissing
end

# struct Config
#     forms::Dict{Symbol, Any}
#     items_file::String
#     anchor_items_file
#     anchor_number
#     versions_file::String
#     constraints_file::String
#     results_file::String
#     tcc_file::String
#     plot_file::String
# end

# # Define a struct for the parameters
# mutable struct Params
#     n::Int
#     num_forms::Int
#     max_items::Int
#     f::Int
#     k::Int
#     r::Int
#     shadow_test_size::Int
#     method::String
#     bank::DataFrame
#     anchor_number::Int
#     ## For TCC and TIF
#     theta::VectorOrMissing
#     p::MatrixOrMissing
#     info::MatrixOrMissing
#     tau::MatrixOrMissing
#     tau_info::VectorOrMissing
#     ## For TC
#     delta::MatrixOrMissing
# end
