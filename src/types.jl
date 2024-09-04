# Define a struct for the configuration
MatrixOrMissing = Union{Matrix{AbstractFloat}, Nothing}
DataFrameOrMissing = Union{DataFrame, Nothing}
VectorOrMissing = Union{Vector{AbstractFloat}, Nothing}

# Define a struct for the configuration
struct Config
    forms::Dict{Symbol, Any}
    items_file::AbstractString
    anchor_items_file::Union{String, Missing}
    forms_file::AbstractString
    constraints_file::AbstractString
    results_file::AbstractString
    tcc_file::AbstractString
    plot_file::AbstractString
    solver::AbstractString
    verbose::Int
end

# Define a struct for the parms
mutable struct Parameters
    n::Int
    num_forms::Int
    max_items::Int
    operational_items::Int
    max_item_use::Int
    f::Int
    k::Int
    r::Int
    shadow_test::Int
    method::AbstractString
    bank::DataFrame
    anchor_tests::Int
    anchor_size::Int
    theta::VectorOrMissing
    p::MatrixOrMissing
    info::MatrixOrMissing
    tau::MatrixOrMissing
    tau_info::VectorOrMissing
    relative_target_weights::VectorOrMissing
    relative_target_points::VectorOrMissing
    delta::MatrixOrMissing
    verbose::Int
end

# Define a struct for holding constraint information
struct Constraint
    id::AbstractString
    type::AbstractString
    condition::Function
    lb::Number
    ub::Number
end
