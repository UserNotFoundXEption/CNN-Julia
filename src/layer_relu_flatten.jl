struct ReLULayer{O} <: Operator
    out::GraphNode{O}
end

function build_layer(::ReLUSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    out = alloc_act!(pool, in_shape..., batch_size)
    return ReLULayer(out), in_shape
end

function primal!(layer::ReLULayer, x::GraphNode)
    xd = x.data
    od = layer.out.data

    @inbounds for i in eachindex(od)
        od[i] = max(0f0, xd[i])
    end

    return nothing
end

function adjoint!(layer::ReLULayer, x::GraphNode)
    xd = x.data
    og = layer.out.grad
    xg = x.grad

    @inbounds for i in eachindex(xg)
        xg[i] += ifelse(xd[i] > 0f0, og[i], 0f0)
    end

    return nothing
end


struct FlattenLayer{O} <: Operator
    out::GraphNode{O}
    aliased::Bool
end

function build_layer(::FlattenSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    flat_size = prod(in_shape)

    out = alloc_act!(pool, flat_size, batch_size)

    return FlattenLayer(out, false), (flat_size,)
end

function build_layer(::FlattenSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int, prev_out::GraphNode)
    flat_size = prod(in_shape)

    out = GraphNode(
        reshape(prev_out.data, flat_size, batch_size),
        reshape(prev_out.grad, flat_size, batch_size),
    )

    return FlattenLayer(out, true), (flat_size,)
end

function primal!(layer::FlattenLayer, x::GraphNode)
    # Jeśli Flatten jest aliasem poprzedniego bufora, nie trzeba nic kopiować.
    if layer.aliased
        return nothing
    end

    copyto!(layer.out.data, x.data)

    return nothing
end

function adjoint!(layer::FlattenLayer, x::GraphNode)
    # Jeśli data i grad są aliasami, gradient już trafia do tego samego bufora.
    if layer.aliased
        return nothing
    end

    og = layer.out.grad
    xg = x.grad

    @inbounds for i in eachindex(xg)
        xg[i] += og[i]
    end

    return nothing
end