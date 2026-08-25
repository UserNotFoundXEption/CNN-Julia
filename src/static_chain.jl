struct StaticChain{T <: Tuple}
    layers::T
end

StaticChain(layers...) = StaticChain(layers)

primal_train!(layer, x) = primal!(layer, x)
primal_test!(layer, x) = primal!(layer, x)

@generated function forward_train!(chain::StaticChain{T}, x::GraphNode) where {T}
    N = length(T.parameters)

    exprs = Expr[
        :(curr_x = x)
    ]

    for i in 1:N
        layer_expr = :(getfield(chain.layers, $i))

        push!(exprs, :(primal_train!($layer_expr, curr_x)))
        push!(exprs, :(curr_x = $layer_expr.out))
    end

    push!(exprs, :(return curr_x))

    return Expr(:block, exprs...)
end

@generated function forward_test!(chain::StaticChain{T}, x::GraphNode) where {T}
    N = length(T.parameters)

    exprs = Expr[
        :(curr_x = x)
    ]

    for i in 1:N
        layer_expr = :(getfield(chain.layers, $i))

        push!(exprs, :(primal_test!($layer_expr, curr_x)))
        push!(exprs, :(curr_x = $layer_expr.out))
    end

    push!(exprs, :(return curr_x))

    return Expr(:block, exprs...)
end

@generated function backward!(chain::StaticChain{T}, x::GraphNode) where {T}
    N = length(T.parameters)

    exprs = Expr[]

    for i in N:-1:1
        layer_expr = :(getfield(chain.layers, $i))
        input_expr = i == 1 ? :(x) : :(getfield(chain.layers, $(i - 1)).out)

        push!(exprs, :(adjoint!($layer_expr, $input_expr)))
    end

    push!(exprs, :(return nothing))

    return Expr(:block, exprs...)
end