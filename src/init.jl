function he_init_dense(in_dim::Int, out_dim::Int)
    return randn(Float32, out_dim, in_dim) .* sqrt(2f0 / in_dim)
end

function he_init_conv(kh::Int, kw::Int, cin::Int, cout::Int)
    fan_in = kh * kw * cin
    return randn(Float32, kh, kw, cin, cout) .* sqrt(2f0 / fan_in)
end

function he_uniform!(x; fan_in::Int)
    scale = Float32(sqrt(24.0 / fan_in))
    x .= (rand(Float32, size(x)...) .- 0.5f0) .* scale
    return x
end

function he_normal!(x; fan_in::Int)
    x .= randn(Float32, size(x)...) .* sqrt(2f0 / fan_in)
    return x
end