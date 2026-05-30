struct Conv
    W::Node{Array{Float32,4}}
    pad::Int
end

function Conv(kh::Int, kw::Int, cin::Int, cout::Int; pad::Int=0)
    W = Node(he_init_conv(kh, kw, cin, cout))
    return Conv(W, pad)
end

function pad_input(x::Array{Float32,4}, pad::Int)
    if pad == 0
        return x
    end
    h, w, c, b = size(x)
    xp = zeros(Float32, h + 2pad, w + 2pad, c, b)
    @inbounds xp[pad+1:pad+h, pad+1:pad+w, :, :] .= x
    return xp
end

function crop_unpad!(dx::Array{Float32,4}, dxp::Array{Float32,4}, pad::Int)
    if pad == 0
        dx .+= dxp
    else
        h, w, c, b = size(dx)
        @inbounds dx .+= dxp[pad+1:pad+h, pad+1:pad+w, :, :]
    end
end

function forward(l::Conv, x::Node{Array{Float32,4}})
    xval = x.value
    wval = l.W.value
    pad = l.pad

    kh, kw, cin, cout = size(wval)
    h, w, c, bsz = size(xval)
    @assert c == cin "Conv channel mismatch: input has $c, weights expect $cin"

    xp = pad_input(xval, pad)
    hp, wp, _, _ = size(xp)

    oh = hp - kh + 1
    ow = wp - kw + 1
    out_val = zeros(Float32, oh, ow, cout, bsz)

    @inbounds for n in 1:bsz
        for co in 1:cout
            for i in 1:oh
                for j in 1:ow
                    s = 0f0
                    for ci in 1:cin
                        for u in 1:kh
                            for v in 1:kw
                                s += xp[i+u-1, j+v-1, ci, n] * wval[u, v, ci, co]
                            end
                        end
                    end
                    out_val[i, j, co, n] = s
                end
            end
        end
    end

    out = Node(out_val)
    out.parents = [x, l.W]

    out.backward_fn = () -> begin
        dxp = zeros(Float32, size(xp))
        dw = zeros(Float32, size(wval))
        db = zeros(Float32, cout)

        @inbounds for n in 1:bsz
            for co in 1:cout
                for i in 1:oh
                    for j in 1:ow
                        go = out.grad[i, j, co, n]
                        db[co] += go
                        for ci in 1:cin
                            for u in 1:kh
                                for v in 1:kw
                                    dxp[i+u-1, j+v-1, ci, n] += wval[u, v, ci, co] * go
                                    dw[u, v, ci, co] += xp[i+u-1, j+v-1, ci, n] * go
                                end
                            end
                        end
                    end
                end
            end
        end

        crop_unpad!(x.grad, dxp, pad)
        l.W.grad .+= dw
    end

    return out
end