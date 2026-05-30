struct Dense
    W::Node{Matrix{Float32}}
end

function Dense(input_dim::Int, output_dim::Int)
    W = Node(he_init_dense(input_dim, output_dim))
    return Dense(W)
end

function forward(l::Dense, x::Node{Matrix{Float32}})
    y = l.W.value * x.value
    out = Node(y)
    out.parents = [x, l.W]

    out.backward_fn = () -> begin
        x.grad   .+= l.W.value' * out.grad
        l.W.grad .+= out.grad * x.value'
    end

    return out
end