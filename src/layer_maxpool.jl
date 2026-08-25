struct MaxPoolLayer{O, A} <: Operator
    pool::Tuple{Int, Int}
    stride::Int
    argmax::A
    out::GraphNode{O}
end

function build_layer(bp::MaxPoolSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    h, w, c = in_shape
    ph, pw = bp.pool
    stride = bp.stride

    oh = fld(h - ph, stride) + 1
    ow = fld(w - pw, stride) + 1

    out = alloc_act!(pool, oh, ow, c, batch_size)

    argmax = Array{Int}(undef, oh, ow, c, batch_size)

    return MaxPoolLayer(bp.pool, stride, argmax, out), (oh, ow, c)
end

function primal!(layer::MaxPoolLayer, x::GraphNode)
    xd = x.data
    od = layer.out.data
    argmax = layer.argmax

    H, W, C, B = size(xd)
    OH, OW, _, _ = size(od)

    ph, pw = layer.pool
    stride = layer.stride

    @inbounds for b in 1:B
        batch_offset = (b - 1) * H * W * C

        for ch in 1:C
            channel_offset = (ch - 1) * H * W

            for ow in 1:OW
                w_start = (ow - 1) * stride + 1

                for oh in 1:OH
                    h_start = (oh - 1) * stride + 1

                    best_val = -Inf32
                    best_idx = 1

                    for dw in 0:pw-1
                        iw = w_start + dw

                        for dh in 0:ph-1
                            ih = h_start + dh

                            val = xd[ih, iw, ch, b]

                            if val > best_val
                                best_val = val

                                best_idx = ih + (iw - 1) * H + channel_offset + batch_offset
                            end
                        end
                    end

                    od[oh, ow, ch, b] = best_val
                    argmax[oh, ow, ch, b] = best_idx
                end
            end
        end
    end

    return nothing
end

function adjoint!(layer::MaxPoolLayer, x::GraphNode)
    xg = x.grad
    og = layer.out.grad
    argmax = layer.argmax

    @inbounds for i in eachindex(og)
        xg[argmax[i]] += og[i]
    end

    return nothing
end