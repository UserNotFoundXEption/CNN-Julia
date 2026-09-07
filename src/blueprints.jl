abstract type Operator end
abstract type Blueprint end

struct DenseSpec <: Blueprint
    dimensions::Pair{Int, Int}
    bias::Bool
end

struct ConvSpec <: Blueprint
    kernel_size::Tuple{Int, Int}
    channels::Pair{Int, Int}
    padding::Int
    bias::Bool
end

struct MaxPoolSpec <: Blueprint
    pool_size::Tuple{Int, Int}
    stride::Int
end

struct DropoutSpec <: Blueprint
    drop_probability::Float32
end

struct ReLUSpec <: Blueprint end
struct FlattenSpec <: Blueprint end

const relu = ReLUSpec()
const flatten = FlattenSpec()

Dense(dimensions::Pair{Int, Int}; bias::Bool=true) =
    DenseSpec(dimensions, bias)

Dense(dimensions::Pair{Int, Int}, ::ReLUSpec; bias::Bool=true) =
    (DenseSpec(dimensions, bias), ReLUSpec())

function Conv(
    kernel_size::Tuple{Int, Int},
    channels::Pair{Int, Int};
    padding::Int=0,
    bias::Bool=false,
)
    return ConvSpec(kernel_size, channels, padding, bias)
end

MaxPool(pool_size::Tuple{Int, Int}; stride::Int=pool_size[1]) =
    MaxPoolSpec(pool_size, stride)

Dropout(drop_probability::Real) =
    DropoutSpec(Float32(drop_probability))

Flatten() =
    FlattenSpec()

struct ChainDef{T <: Tuple}
    blueprints::T
end

function Chain(layer_definitions...)
    flattened_blueprints = Any[]

    for definition in layer_definitions
        if definition isa Tuple
            append!(flattened_blueprints, definition)
        else
            push!(flattened_blueprints, definition)
        end
    end

    return ChainDef(Tuple(flattened_blueprints))
end
