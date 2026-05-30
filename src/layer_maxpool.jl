struct MaxPool
    kh::Int
    kw::Int
    stride::Int
end

MaxPool(kh::Int, kw::Int; stride::Int=kh) = MaxPool(kh, kw, stride)

function forward(l::MaxPool, x::Node{Array{Float32,4}})
    xval = x.value
    @assert ndims(xval) == 4 "MaxPool expects H×W×C×B input"

    h, w, c, bsz = size(xval)
    kh, kw, stride = l.kh, l.kw, l.stride

    oh = fld(h - kh, stride) + 1
    ow = fld(w - kw, stride) + 1

    out_val = zeros(Float32, oh, ow, c, bsz)
    maxpos = Array{NTuple{2,Int}}(undef, oh, ow, c, bsz)

    @inbounds for n in 1:bsz
        for ch in 1:c
            for i in 1:oh
                for j in 1:ow
                    hs = (i - 1) * stride + 1
                    ws = (j - 1) * stride + 1

                    best = -Inf32
                    bestu = hs
                    bestv = ws

                    for u in 0:kh-1
                        for v in 0:kw-1
                            val = xval[hs+u, ws+v, ch, n]
                            if val > best
                                best = val
                                bestu = hs + u
                                bestv = ws + v
                            end
                        end
                    end

                    out_val[i, j, ch, n] = best
                    maxpos[i, j, ch, n] = (bestu, bestv)
                end
            end
        end
    end

    out = Node(out_val)
    out.parents = [x]

    out.backward_fn = () -> begin
        @inbounds for n in 1:bsz
            for ch in 1:c
                for i in 1:oh
                    for j in 1:ow
                        u, v = maxpos[i, j, ch, n]
                        x.grad[u, v, ch, n] += out.grad[i, j, ch, n]
                    end
                end
            end
        end
    end

    return out
end