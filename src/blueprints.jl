abstract type Operator end
abstract type Blueprint end

struct DenseSpec <: Blueprint
    in_out::Pair{Int, Int}
    bias::Bool
end

struct ConvSpec <: Blueprint
    filter::Tuple{Int, Int}
    ch::Pair{Int, Int}
    pad::Int
    bias::Bool
end

struct MaxPoolSpec <: Blueprint
    pool::Tuple{Int, Int}
    stride::Int
end

struct DropoutSpec <: Blueprint
    p::Float32
end

struct ReLUSpec <: Blueprint end
struct FlattenSpec <: Blueprint end

const relu = ReLUSpec()
const flatten = FlattenSpec()

Dense(pair::Pair{Int, Int}; bias::Bool=true) =
    DenseSpec(pair, bias)

Dense(pair::Pair{Int, Int}, ::ReLUSpec; bias::Bool=true) =
    (DenseSpec(pair, bias), ReLUSpec())

Conv(filter::Tuple{Int, Int}, ch::Pair{Int, Int}; pad::Int=0, bias::Bool=false) =
    ConvSpec(filter, ch, pad, bias)

MaxPool(pool::Tuple{Int, Int}; stride::Int=pool[1]) =
    MaxPoolSpec(pool, stride)

Dropout(p::Real) =
    DropoutSpec(Float32(p))

Flatten() =
    FlattenSpec()

struct ChainDef{T <: Tuple}
    blueprints::T
end

function Chain(args...)
    flat = Any[]

    for a in args
        if a isa Tuple
            append!(flat, a)
        else
            push!(flat, a)
        end
    end

    return ChainDef(Tuple(flat))
end