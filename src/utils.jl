function he_init_dense(in_dim::Int, out_dim::Int)
    randn(Float32, out_dim, in_dim) .* sqrt(2f0 / in_dim)
end

function he_init_conv(kh::Int, kw::Int, cin::Int, cout::Int)
    fan_in = kh * kw * cin
    randn(Float32, kh, kw, cin, cout) .* sqrt(2f0 / fan_in)
end

function softmax(x::Matrix{Float32})
    z = x .- maximum(x, dims=1)
    ex = exp.(z)
    ex ./ sum(ex, dims=1)
end

function cross_entropy(logits::Node{Matrix{Float32}}, y::Vector{Int})
    probs = softmax(logits.value)
    bsz = length(y)

    loss_val = 0f0
    @inbounds for j in 1:bsz
        loss_val -= log(probs[y[j], j] + 1f-12)
    end
    loss_val /= bsz

    out = Node(loss_val)
    out.parents = [logits]

    out.backward_fn = () -> begin
        g = copy(probs)
        @inbounds for j in 1:bsz
            g[y[j], j] -= 1f0
        end
        g ./= bsz
        logits.grad .+= out.grad .* g
    end

    return out
end

function sgd!(params, lr)
    for p in params
        p.value .-= lr .* p.grad .* 20
    end
end
