using BenchmarkTools

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

    @views @inbounds xp[pad+1:pad+h, pad+1:pad+w, :, :] .= x
    return xp
end

function crop_unpad!(dx::Array{Float32,4}, dxp::Array{Float32,4}, pad::Int)
    if pad == 0
        dx .+= dxp
    else
        h, w, c, b = size(dx)
        @views @inbounds dx .+= dxp[pad+1:pad+h, pad+1:pad+w, :, :]
    end
end

function im2col(xp::Array{Float32,4}, kh::Int, kw::Int)
    hp, wp, cin, bsz = size(xp)

    oh = hp - kh + 1
    ow = wp - kw + 1

    ksize = kh * kw * cin
    ncols = oh * ow * bsz

    cols = Matrix{Float32}(undef, ksize, ncols)

    @inbounds for n in 1:bsz
        for i in 1:oh
            for j in 1:ow
                col = (n - 1) * oh * ow + (i - 1) * ow + j
                row = 1

                for ci in 1:cin
                    for u in 1:kh
                        for v in 1:kw
                            cols[row, col] = xp[i+u-1, j+v-1, ci, n]
                            row += 1
                        end
                    end
                end
            end
        end
    end

    return cols, oh, ow
end

function col2im!(
    dxp::Array{Float32,4},
    dcols::Matrix{Float32},
    kh::Int,
    kw::Int,
    oh::Int,
    ow::Int
)
    hp, wp, cin, bsz = size(dxp)

    @inbounds for n in 1:bsz
        for i in 1:oh
            for j in 1:ow
                col = (n - 1) * oh * ow + (i - 1) * ow + j
                row = 1

                for ci in 1:cin
                    for u in 1:kh
                        for v in 1:kw
                            dxp[i+u-1, j+v-1, ci, n] += dcols[row, col]
                            row += 1
                        end
                    end
                end
            end
        end
    end
end

function mat_to_4d!(
    out_val::Array{Float32,4},
    out_mat::Matrix{Float32},
    oh::Int,
    ow::Int,
    cout::Int,
    bsz::Int
)
    @inbounds for n in 1:bsz
        for i in 1:oh
            for j in 1:ow
                col = (n - 1) * oh * ow + (i - 1) * ow + j

                for co in 1:cout
                    out_val[i, j, co, n] = out_mat[co, col]
                end
            end
        end
    end
end

function grad_to_mat(
    grad::Array{Float32,4},
    oh::Int,
    ow::Int,
    cout::Int,
    bsz::Int
)
    gmat = Matrix{Float32}(undef, cout, oh * ow * bsz)

    @inbounds for n in 1:bsz
        for i in 1:oh
            for j in 1:ow
                col = (n - 1) * oh * ow + (i - 1) * ow + j

                for co in 1:cout
                    gmat[co, col] = grad[i, j, co, n]
                end
            end
        end
    end

    return gmat
end

function forward(l::Conv, x::Node{Array{Float32,4}})
    xval = x.value
    wval = l.W.value
    pad = l.pad

    kh, kw, cin, cout = size(wval)
    h, w, c, bsz = size(xval)

    @assert c == cin "Conv channel mismatch: input has $c, weights expect $cin"

    xp = pad_input(xval, pad)

    xcol, oh, ow = im2col(xp, kh, kw)

    # W: kh × kw × cin × cout
    # Wmat: cout × (kh*kw*cin)
    Wmat = reshape(wval, kh * kw * cin, cout)'

    # GEMM:
    # out_mat: cout × (oh*ow*bsz)
    out_mat = Wmat * xcol

    out_val = Array{Float32,4}(undef, oh, ow, cout, bsz)
    mat_to_4d!(out_val, out_mat, oh, ow, cout, bsz)

    out = Node(out_val)
    out.parents = [x, l.W]

    out.backward_fn = () -> begin
        gmat = grad_to_mat(out.grad, oh, ow, cout, bsz)

        # dWmat: cout × ksize
        dWmat = gmat * xcol'

        # dXcol: ksize × ncols
        dXcol = Wmat' * gmat

        dxp = zeros(Float32, size(xp))
        col2im!(dxp, dXcol, kh, kw, oh, ow)

        crop_unpad!(x.grad, dxp, pad)

        # dWmat' ma rozmiar ksize × cout,
        # więc można go przekształcić z powrotem do kh × kw × cin × cout
        l.W.grad .+= reshape(dWmat', kh, kw, cin, cout)
    end

    return out
end