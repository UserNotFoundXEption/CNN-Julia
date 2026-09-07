struct ReLULayer{O} <: Operator
    output::GraphNode{O}
end

function build_layer(::ReLUSpec, memory_pool::MemoryPool, input_shape::Tuple, batch_size::Int)
    output_node = alloc_act!(memory_pool, input_shape..., batch_size)
    return ReLULayer(output_node), input_shape
end

function forward!(layer::ReLULayer, input_node::GraphNode)
    input_data = input_node.data
    output_data = layer.output.data

    @inbounds for element_index in eachindex(output_data)
        output_data[element_index] = max(0f0, input_data[element_index])
    end

    return nothing
end

function backward!(layer::ReLULayer, input_node::GraphNode)
    input_data = input_node.data
    output_gradient = layer.output.grad
    input_gradient = input_node.grad

    @inbounds for element_index in eachindex(input_gradient)
        input_gradient[element_index] += ifelse(
            input_data[element_index] > 0f0,
            output_gradient[element_index],
            0f0,
        )
    end

    return nothing
end

struct FlattenLayer{O} <: Operator
    output::GraphNode{O}
    aliased::Bool
end

function build_layer(::FlattenSpec, memory_pool::MemoryPool, input_shape::Tuple, batch_size::Int)
    flattened_size = prod(input_shape)
    output_node = alloc_act!(memory_pool, flattened_size, batch_size)

    return FlattenLayer(output_node, false), (flattened_size,)
end

function build_layer(
    ::FlattenSpec,
    memory_pool::MemoryPool,
    input_shape::Tuple,
    batch_size::Int,
    previous_output::GraphNode,
)
    flattened_size = prod(input_shape)

    output_node = GraphNode(
        reshape(previous_output.data, flattened_size, batch_size),
        reshape(previous_output.grad, flattened_size, batch_size),
    )

    return FlattenLayer(output_node, true), (flattened_size,)
end

function forward!(layer::FlattenLayer, input_node::GraphNode)
    # If Flatten aliases the previous buffer, no data copy is needed.
    if layer.aliased
        return nothing
    end

    copyto!(layer.output.data, input_node.data)

    return nothing
end

function backward!(layer::FlattenLayer, input_node::GraphNode)
    # If data and gradients are aliases, the gradient already reaches the same buffer.
    if layer.aliased
        return nothing
    end

    output_gradient = layer.output.grad
    input_gradient = input_node.grad

    @inbounds for element_index in eachindex(input_gradient)
        input_gradient[element_index] += output_gradient[element_index]
    end

    return nothing
end
