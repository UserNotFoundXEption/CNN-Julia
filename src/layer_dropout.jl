struct DropoutLayer{R, O} <: Operator
    drop_probability::Float32
    mask_buffer::R
    output::GraphNode{O}
end

function build_layer(blueprint::DropoutSpec, memory_pool::MemoryPool, input_shape::Tuple, batch_size::Int)
    mask_buffer = zeros(Float32, input_shape..., batch_size)
    output_node = alloc_act!(memory_pool, input_shape..., batch_size)

    return DropoutLayer(blueprint.drop_probability, mask_buffer, output_node), input_shape
end

function forward_train!(layer::DropoutLayer, input_node::GraphNode)
    input_data = input_node.data
    output_data = layer.output.data
    mask_buffer = layer.mask_buffer
    drop_probability = layer.drop_probability

    if drop_probability <= 0f0
        copyto!(output_data, input_data)
        fill!(mask_buffer, 1f0)
        return nothing
    end

    keep_probability = 1f0 - drop_probability
    inverse_keep_probability = 1f0 / keep_probability

    rand!(mask_buffer)

    @inbounds for element_index in eachindex(output_data)
        mask_value = ifelse(
            mask_buffer[element_index] > drop_probability,
            inverse_keep_probability,
            0f0,
        )
        mask_buffer[element_index] = mask_value
        output_data[element_index] = input_data[element_index] * mask_value
    end

    return nothing
end

function forward_test!(layer::DropoutLayer, input_node::GraphNode)
    copyto!(layer.output.data, input_node.data)
    fill!(layer.mask_buffer, 1f0)

    return nothing
end

function backward!(layer::DropoutLayer, input_node::GraphNode)
    mask_buffer = layer.mask_buffer
    output_gradient = layer.output.grad
    input_gradient = input_node.grad

    @inbounds for element_index in eachindex(input_gradient)
        input_gradient[element_index] += output_gradient[element_index] * mask_buffer[element_index]
    end

    return nothing
end
