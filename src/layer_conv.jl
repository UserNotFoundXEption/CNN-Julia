struct ConvLayer{W, BIAS, O, C, OM, GM, DC, XP, DXP} <: Operator
    w::GraphNode{W}
    b::GraphNode{BIAS}
    out::GraphNode{O}

    pad::Int
    has_bias::Bool

    cols::C        # K × N
    out_mat::OM    # Cout × N
    grad_mat::GM   # Cout × N
    dcols::DC      # K × N

    xpad::XP       # padded input
    dxpad::DXP     # padded input gradient
end

function build_layer(bp::ConvSpec, pool::MemoryPool, in_shape::Tuple, batch_size::Int)
    h, w, cin_actual = in_shape
    kh, kw = bp.filter
    cout = bp.ch.second
    pad = bp.pad

    oh = h + 2pad - kh + 1
    ow = w + 2pad - kw + 1

    # W: kh × kw × Cin × Cout
    wnode = alloc_weight!(pool, kh, kw, cin_actual, cout)
    he_normal!(wnode.data; fan_in=kh * kw * cin_actual)

    if bp.bias
        bnode = alloc_weight!(pool, cout)
    else
        bnode = alloc_weight!(pool, 0)
    end

    out = alloc_act!(pool, oh, ow, cout, batch_size)

    ksize = kh * kw * cin_actual
    ncols = oh * ow * batch_size

    cols = zeros(Float32, ksize, ncols)
    out_mat = zeros(Float32, cout, ncols)
    grad_mat = zeros(Float32, cout, ncols)
    dcols = zeros(Float32, ksize, ncols)

    xpad = zeros(Float32, h + 2pad, w + 2pad, cin_actual, batch_size)
    dxpad = zeros(Float32, h + 2pad, w + 2pad, cin_actual, batch_size)

    layer = ConvLayer(
        wnode,
        bnode,
        out,
        pad,
        bp.bias,
        cols,
        out_mat,
        grad_mat,
        dcols,
        xpad,
        dxpad,
    )

    return layer, (oh, ow, cout)
end


# ========================= padding =========================

function copy_pad!(xpad::AbstractArray{Float32,4}, x::AbstractArray{Float32,4}, pad::Int)
    if pad == 0
        copyto!(xpad, x)
    else
        fill!(xpad, 0f0)

        h, w, _, _ = size(x)

        @views xpad[pad+1:pad+h, pad+1:pad+w, :, :] .= x
    end

    return nothing
end

function crop_unpad_add!(xgrad::AbstractArray{Float32,4}, dxpad::AbstractArray{Float32,4}, pad::Int)
    if pad == 0
        xgrad .+= dxpad
    else
        h, w, _, _ = size(xgrad)

        @views xgrad .+= dxpad[pad+1:pad+h, pad+1:pad+w, :, :]
    end

    return nothing
end


# ========================= im2col / col2im =========================

function im2col!(
    cols::Matrix{Float32},
    xpad::Array{Float32,4},
    kh::Int,
    kw::Int,
)
    hp, wp, cin, batch_size = size(xpad)

    oh = hp - kh + 1
    ow = wp - kw + 1

    @inbounds for b in 1:batch_size
        batch_col_offset = (b - 1) * oh * ow

        for out_w in 1:ow
            for out_h in 1:oh
                col = batch_col_offset + (out_w - 1) * oh + out_h
                row = 1

                for c in 1:cin
                    for kernel_w in 1:kw
                        in_w = out_w + kernel_w - 1

                        for kernel_h in 1:kh
                            in_h = out_h + kernel_h - 1

                            cols[row, col] = xpad[in_h, in_w, c, b]
                            row += 1
                        end
                    end
                end
            end
        end
    end

    return nothing
end

function col2im!(
    dxpad::Array{Float32,4},
    dcols::Matrix{Float32},
    kh::Int,
    kw::Int,
)
    hp, wp, cin, batch_size = size(dxpad)

    oh = hp - kh + 1
    ow = wp - kw + 1

    fill!(dxpad, 0f0)

    @inbounds for b in 1:batch_size
        batch_col_offset = (b - 1) * oh * ow

        for out_w in 1:ow
            for out_h in 1:oh
                col = batch_col_offset + (out_w - 1) * oh + out_h
                row = 1

                for c in 1:cin
                    for kernel_w in 1:kw
                        in_w = out_w + kernel_w - 1

                        for kernel_h in 1:kh
                            in_h = out_h + kernel_h - 1

                            dxpad[in_h, in_w, c, b] += dcols[row, col]
                            row += 1
                        end
                    end
                end
            end
        end
    end

    return nothing
end


# ========================= matrix <-> tensor =========================

function mat_to_4d!(
    out::AbstractArray{Float32,4},
    mat::Matrix{Float32},
)
    oh, ow, cout, batch_size = size(out)

    @inbounds for b in 1:batch_size
        batch_col_offset = (b - 1) * oh * ow

        for out_w in 1:ow
            for out_h in 1:oh
                col = batch_col_offset + (out_w - 1) * oh + out_h

                for c in 1:cout
                    out[out_h, out_w, c, b] = mat[c, col]
                end
            end
        end
    end

    return nothing
end

function grad_to_mat!(
    mat::Matrix{Float32},
    grad::AbstractArray{Float32,4},
)
    oh, ow, cout, batch_size = size(grad)

    @inbounds for b in 1:batch_size
        batch_col_offset = (b - 1) * oh * ow

        for out_w in 1:ow
            for out_h in 1:oh
                col = batch_col_offset + (out_w - 1) * oh + out_h

                for c in 1:cout
                    mat[c, col] = grad[out_h, out_w, c, b]
                end
            end
        end
    end

    return nothing
end


# ========================= Conv forward / backward =========================

function primal!(layer::ConvLayer, x::GraphNode)
    xd = x.data
    wd = layer.w.data
    od = layer.out.data

    kh, kw, cin, cout = size(wd)

    copy_pad!(layer.xpad, xd, layer.pad)

    im2col!(layer.cols, layer.xpad, kh, kw)

    # W_flat: K × Cout
    W_flat = reshape(wd, kh * kw * cin, cout)

    # out_mat = W_flat' * cols
    # shapes:
    #   W_flat'  = Cout × K
    #   cols     = K × N
    #   out_mat  = Cout × N
    mul!(layer.out_mat, transpose(W_flat), layer.cols)

    mat_to_4d!(od, layer.out_mat)

    if layer.has_bias
        bd = layer.b.data

        @inbounds for b in axes(od, 4)
            for c in axes(od, 3)
                bias = bd[c]

                for w in axes(od, 2)
                    for h in axes(od, 1)
                        od[h, w, c, b] += bias
                    end
                end
            end
        end
    end

    return nothing
end

function adjoint!(layer::ConvLayer, x::GraphNode)
    wd = layer.w.data
    wg = layer.w.grad

    kh, kw, cin, cout = size(wd)

    W_flat = reshape(wd, kh * kw * cin, cout)
    Wg_flat = reshape(wg, kh * kw * cin, cout)

    # grad_mat: Cout × N
    grad_to_mat!(layer.grad_mat, layer.out.grad)

    # dW += cols * grad_mat'
    # shapes:
    #   cols       = K × N
    #   grad_mat'  = N × Cout
    #   Wg_flat    = K × Cout
    mul!(Wg_flat, layer.cols, transpose(layer.grad_mat), 1f0, 1f0)

    if layer.has_bias
        bg = layer.b.grad
        gm = layer.grad_mat

        @inbounds for c in 1:cout
            acc = 0f0
            for j in axes(gm, 2)
                acc += gm[c, j]
            end
            bg[c] += acc
        end
    end

    # dcols = W_flat * grad_mat
    # shapes:
    #   W_flat    = K × Cout
    #   grad_mat  = Cout × N
    #   dcols     = K × N
    mul!(layer.dcols, W_flat, layer.grad_mat)

    col2im!(layer.dxpad, layer.dcols, kh, kw)

    crop_unpad_add!(x.grad, layer.dxpad, layer.pad)

    return nothing
end