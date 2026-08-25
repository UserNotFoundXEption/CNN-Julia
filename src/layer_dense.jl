struct DenseLayer{W, B, O} <: Operator
    w::GraphNode{W}
    b::GraphNode{B}
    out::GraphNode{O}
    has_bias::Bool
end

function build_layer(bp::DenseSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    declared_in = bp.in_out.first
    outsize = bp.in_out.second

    actual_in = prod(in_shape)

    w = alloc_weight!(pool, outsize, declared_in)
    he_normal!(w.data; fan_in=declared_in)

    if bp.bias
        b = alloc_weight!(pool, outsize)
    else
        b = alloc_weight!(pool, 0)
    end

    out = alloc_act!(pool, outsize, batch_size)

    return DenseLayer(w, b, out, bp.bias), (outsize,)
end

function primal!(layer::DenseLayer, x::GraphNode)
    mul!(layer.out.data, layer.w.data, x.data)

    if layer.has_bias
        od = layer.out.data
        bd = layer.b.data

        @inbounds for batch in axes(od, 2)
            for i in axes(od, 1)
                od[i, batch] += bd[i]
            end
        end
    end

    return nothing
end

function adjoint!(layer::DenseLayer, x::GraphNode)
    mul!(layer.w.grad, layer.out.grad, x.data', 1f0, 1f0)

    if layer.has_bias
        bg = layer.b.grad
        og = layer.out.grad

        @inbounds for batch in axes(og, 2)
            for i in axes(og, 1)
                bg[i] += og[i, batch]
            end
        end
    end

    mul!(x.grad, layer.w.data', layer.out.grad, 1f0, 1f0)

    return nothing
end