struct MaxPoolLayer{O, A} <: Operator
    pool_size::Tuple{Int, Int}
    stride::Int
    max_indices::A
    output::GraphNode{O}
end

function build_layer(blueprint::MaxPoolSpec, memory_pool::MemoryPool, input_shape::Tuple, batch_size::Int)
    input_height, input_width, channels = input_shape
    pool_height, pool_width = blueprint.pool_size
    stride = blueprint.stride

    output_height = fld(input_height - pool_height, stride) + 1
    output_width = fld(input_width - pool_width, stride) + 1

    output_node = alloc_act!(memory_pool, output_height, output_width, channels, batch_size)
    max_indices = Array{Int}(undef, output_height, output_width, channels, batch_size)

    return MaxPoolLayer(blueprint.pool_size, stride, max_indices, output_node), (
        output_height,
        output_width,
        channels,
    )
end

function forward!(layer::MaxPoolLayer, input_node::GraphNode)
    input_data = input_node.data
    output_data = layer.output.data
    max_indices = layer.max_indices

    input_height, input_width, channels, batch_size = size(input_data)
    output_height, output_width, _, _ = size(output_data)

    pool_height, pool_width = layer.pool_size
    stride = layer.stride

    @inbounds for batch_index in 1:batch_size
        batch_offset = (batch_index - 1) * input_height * input_width * channels

        for channel_index in 1:channels
            channel_offset = (channel_index - 1) * input_height * input_width

            for output_width_index in 1:output_width
                input_width_start = (output_width_index - 1) * stride + 1

                for output_height_index in 1:output_height
                    input_height_start = (output_height_index - 1) * stride + 1

                    maximum_value = -Inf32
                    maximum_linear_index = 1

                    for pool_width_offset in 0:pool_width-1
                        input_width_index = input_width_start + pool_width_offset

                        for pool_height_offset in 0:pool_height-1
                            input_height_index = input_height_start + pool_height_offset

                            input_value = input_data[
                                input_height_index,
                                input_width_index,
                                channel_index,
                                batch_index,
                            ]

                            if input_value > maximum_value
                                maximum_value = input_value
                                maximum_linear_index = (
                                    input_height_index +
                                    (input_width_index - 1) * input_height +
                                    channel_offset +
                                    batch_offset
                                )
                            end
                        end
                    end

                    output_data[
                        output_height_index,
                        output_width_index,
                        channel_index,
                        batch_index,
                    ] = maximum_value
                    max_indices[
                        output_height_index,
                        output_width_index,
                        channel_index,
                        batch_index,
                    ] = maximum_linear_index
                end
            end
        end
    end

    return nothing
end

function backward!(layer::MaxPoolLayer, input_node::GraphNode)
    input_gradient = input_node.grad
    output_gradient = layer.output.grad
    max_indices = layer.max_indices

    @inbounds for output_index in eachindex(output_gradient)
        input_gradient[max_indices[output_index]] += output_gradient[output_index]
    end

    return nothing
end
