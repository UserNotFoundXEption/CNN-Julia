struct MaxPool
    kh::Int
    kw::Int
    stride::Int
end

MaxPool(kh::Int, kw::Int; stride::Int=kh) = MaxPool(kh, kw, stride)

function forward(l::MaxPool, x::Node{Array{Float32,4}})
    xval = x.value          ::Array{Float32,4}
    @assert ndims(xval) == 4 "MaxPool expects HxWxCxB input"

    h, w, c, bsz  = size(xval)
    kh, kw, stride = l.kh, l.kw, l.stride

    oh = fld(h - kh, stride) + 1
    ow = fld(w - kw, stride) + 1

    out_val  = Array{Float32,4}(undef, oh, ow, c, bsz)
    maxpos_u = Array{Int,4}(undef, oh, ow, c, bsz)
    maxpos_v = Array{Int,4}(undef, oh, ow, c, bsz)

    @inbounds for n in 1:bsz
        for ch in 1:c
            for i in 1:oh
                hs = (i - 1) * stride + 1
                for j in 1:ow
                    ws = (j - 1) * stride + 1

                    best  = -Inf32
                    bestu = hs
                    bestv = ws

                    for u in 0:kh-1
                        @simd for v in 0:kw-1
                            val = xval[hs+u, ws+v, ch, n]
                            if val > best
                                best  = val
                                bestu = hs + u
                                bestv = ws + v
                            end
                        end
                    end

                    out_val[i, j, ch, n] = best
                    maxpos_u[i, j, ch, n] = bestu
                    maxpos_v[i, j, ch, n] = bestv
                end
            end
        end
    end

    out = Node(out_val)
    out.parents = [x]

    out.backward_fn = () -> begin
        xgrad  = x.grad     ::Array{Float32,4}
        ograd  = out.grad   ::Array{Float32,4}

        @inbounds for n in 1:bsz
            for ch in 1:c
                for i in 1:oh
                    @simd for j in 1:ow
                        xgrad[maxpos_u[i,j,ch,n], maxpos_v[i,j,ch,n], ch, n] += ograd[i, j, ch, n]
                    end
                end
            end
        end
    end

    return out
end