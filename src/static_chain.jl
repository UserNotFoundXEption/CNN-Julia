struct StaticChain{T <: Tuple}
    layers::T
end

StaticChain(layers...) = StaticChain(layers)

forward_train!(layer, input_node) = forward!(layer, input_node)
forward_test!(layer, input_node) = forward!(layer, input_node)

@generated function forward_train!(chain::StaticChain{T}, input_node::GraphNode) where {T}
    number_of_layers = length(T.parameters)

    expressions = Expr[
        :(current_input = input_node)
    ]

    for layer_index in 1:number_of_layers
        layer_expression = :(getfield(chain.layers, $layer_index))

        push!(expressions, :(forward_train!($layer_expression, current_input)))
        push!(expressions, :(current_input = $layer_expression.output))
    end

    push!(expressions, :(return current_input))

    return Expr(:block, expressions...)
end

@generated function forward_test!(chain::StaticChain{T}, input_node::GraphNode) where {T}
    number_of_layers = length(T.parameters)

    expressions = Expr[
        :(current_input = input_node)
    ]

    for layer_index in 1:number_of_layers
        layer_expression = :(getfield(chain.layers, $layer_index))

        push!(expressions, :(forward_test!($layer_expression, current_input)))
        push!(expressions, :(current_input = $layer_expression.output))
    end

    push!(expressions, :(return current_input))

    return Expr(:block, expressions...)
end

@generated function backward!(chain::StaticChain{T}, input_node::GraphNode) where {T}
    number_of_layers = length(T.parameters)
    expressions = Expr[]

    for layer_index in number_of_layers:-1:1
        layer_expression = :(getfield(chain.layers, $layer_index))
        layer_input_expression = if layer_index == 1
            :(input_node)
        else
            :(getfield(chain.layers, $(layer_index - 1)).output)
        end

        push!(expressions, :(backward!($layer_expression, $layer_input_expression)))
    end

    push!(expressions, :(return nothing))

    return Expr(:block, expressions...)
end
