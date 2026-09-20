

struct DenseLayer{W, B, O} <: Operator
    weights::GraphNode{W}
    bias::GraphNode{B}
    output::GraphNode{O}
    has_bias::Bool
end

function build_layer(blueprint::DenseSpec, memory_pool::MemoryPool, input_shape::Tuple, batch_size::Int)
    declared_input_size = blueprint.dimensions.first
    output_size = blueprint.dimensions.second
    actual_input_size = prod(input_shape)

    _ = actual_input_size

    weights_node = alloc_weight!(memory_pool, output_size, declared_input_size)
    he_normal!(weights_node.data; fan_in=declared_input_size)

    bias_node = if blueprint.bias
        alloc_weight!(memory_pool, output_size)
    else
        alloc_weight!(memory_pool, 0)
    end

    output_node = alloc_act!(memory_pool, output_size, batch_size)

    return DenseLayer(weights_node, bias_node, output_node, blueprint.bias), (output_size,)
end

function forward!(layer::DenseLayer, input_node::GraphNode)
    mul!(layer.output.data, layer.weights.data, input_node.data)

    if layer.has_bias
        output_data = layer.output.data
        bias_data = layer.bias.data

        @inbounds for batch_index in axes(output_data, 2)
            for output_index in axes(output_data, 1)
                output_data[output_index, batch_index] += bias_data[output_index]
            end
        end
    end

    return nothing
end

function backward!(layer::DenseLayer, input_node::GraphNode)
    mul!(
        layer.weights.grad,
        layer.output.grad,
        input_node.data',
        1f0,
        1f0,
    )

    if layer.has_bias
        bias_gradient = layer.bias.grad
        output_gradient = layer.output.grad

        @inbounds for batch_index in axes(output_gradient, 2)
            for output_index in axes(output_gradient, 1)
                bias_gradient[output_index] += output_gradient[output_index, batch_index]
            end
        end
    end

    mul!(
        input_node.grad,
        layer.weights.data',
        layer.output.grad,
        1f0,
        1f0,
    )

    return nothing
end
