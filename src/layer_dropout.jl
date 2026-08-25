struct DropoutLayer{R, O} <: Operator
    p::Float32
    rand_buf::R
    out::GraphNode{O}
end

function build_layer(bp::DropoutSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    rand_buf = zeros(Float32, in_shape..., batch_size)
    out = alloc_act!(pool, in_shape..., batch_size)

    return DropoutLayer(bp.p, rand_buf, out), in_shape
end

function primal_train!(layer::DropoutLayer, x::GraphNode)
    xd = x.data
    od = layer.out.data
    mask = layer.rand_buf
    p = layer.p

    if p <= 0f0
        copyto!(od, xd)
        fill!(mask, 1f0)
        return nothing
    end

    keep_prob = 1f0 - p
    scale = 1f0 / keep_prob

    rand!(mask)

    @inbounds for i in eachindex(od)
        m = ifelse(mask[i] > p, scale, 0f0)
        mask[i] = m
        od[i] = xd[i] * m
    end

    return nothing
end

function primal_test!(layer::DropoutLayer, x::GraphNode)
    copyto!(layer.out.data, x.data)
    fill!(layer.rand_buf, 1f0)

    return nothing
end

function adjoint!(layer::DropoutLayer, x::GraphNode)
    mask = layer.rand_buf
    og = layer.out.grad
    xg = x.grad

    @inbounds for i in eachindex(xg)
        xg[i] += og[i] * mask[i]
    end

    return nothing
end