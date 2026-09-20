struct ConvLayer{W, BIAS, O, C, OM, GM, DC, XP, DXP} <: Operator
    weights::GraphNode{W}
    bias::GraphNode{BIAS}
    output::GraphNode{O}

    padding::Int
    has_bias::Bool

    input_columns::C
    output_matrix::OM
    output_gradient_matrix::GM
    input_columns_gradient::DC

    padded_input::XP
    padded_input_gradient::DXP
end

function build_layer(blueprint::ConvSpec, memory_pool::MemoryPool, input_shape::Tuple, batch_size::Int)
    input_height, input_width, input_channels = input_shape
    kernel_height, kernel_width = blueprint.kernel_size
    output_channels = blueprint.channels.second
    padding = blueprint.padding

    output_height = input_height + 2padding - kernel_height + 1
    output_width = input_width + 2padding - kernel_width + 1

    weights_node = alloc_weight!(
        memory_pool,
        kernel_height,
        kernel_width,
        input_channels,
        output_channels,
    )
    he_normal!(
        weights_node.data;
        fan_in=kernel_height * kernel_width * input_channels,
    )

    bias_node = if blueprint.bias
        alloc_weight!(memory_pool, output_channels)
    else
        alloc_weight!(memory_pool, 0)
    end

    output_node = alloc_act!(
        memory_pool,
        output_height,
        output_width,
        output_channels,
        batch_size,
    )

    kernel_volume = kernel_height * kernel_width * input_channels
    number_of_columns = output_height * output_width * batch_size

    input_columns = zeros(Float32, kernel_volume, number_of_columns)
    output_matrix = zeros(Float32, output_channels, number_of_columns)
    output_gradient_matrix = zeros(Float32, output_channels, number_of_columns)
    input_columns_gradient = zeros(Float32, kernel_volume, number_of_columns)

    padded_input = zeros(
        Float32,
        input_height + 2padding,
        input_width + 2padding,
        input_channels,
        batch_size,
    )
    padded_input_gradient = zeros(
        Float32,
        input_height + 2padding,
        input_width + 2padding,
        input_channels,
        batch_size,
    )

    layer = ConvLayer(
        weights_node,
        bias_node,
        output_node,
        padding,
        blueprint.bias,
        input_columns,
        output_matrix,
        output_gradient_matrix,
        input_columns_gradient,
        padded_input,
        padded_input_gradient,
    )

    return layer, (output_height, output_width, output_channels)
end

function copy_with_padding!(
    padded_input::AbstractArray{Float32,4},
    input_data::AbstractArray{Float32,4},
    padding::Int,
)
    if padding == 0
        copyto!(padded_input, input_data)
    else
        fill!(padded_input, 0f0)

        input_height, input_width, _, _ = size(input_data)

        @views padded_input[
            padding+1:padding+input_height,
            padding+1:padding+input_width,
            :,
            :,
        ] .= input_data
    end

    return nothing
end

function add_unpadded_gradient!(
    input_gradient::AbstractArray{Float32,4},
    padded_input_gradient::AbstractArray{Float32,4},
    padding::Int,
)
    if padding == 0
        input_gradient .+= padded_input_gradient
    else
        input_height, input_width, _, _ = size(input_gradient)

        @views input_gradient .+= padded_input_gradient[
            padding+1:padding+input_height,
            padding+1:padding+input_width,
            :,
            :,
        ]
    end

    return nothing
end

# ========================= im2col / col2im =========================

function im2col!(
    input_columns::Matrix{Float32},
    padded_input::Array{Float32,4},
    kernel_height::Int,
    kernel_width::Int,
)
    padded_height, padded_width, input_channels, batch_size = size(padded_input)

    output_height = padded_height - kernel_height + 1
    output_width = padded_width - kernel_width + 1

    @inbounds for batch_index in 1:batch_size
        batch_column_offset = (batch_index - 1) * output_height * output_width

        for output_width_index in 1:output_width
            for output_height_index in 1:output_height
                column_index = (
                    batch_column_offset +
                    (output_width_index - 1) * output_height +
                    output_height_index
                )
                row_index = 1

                for channel_index in 1:input_channels
                    for kernel_width_index in 1:kernel_width
                        input_width_index = output_width_index + kernel_width_index - 1

                        for kernel_height_index in 1:kernel_height
                            input_height_index = output_height_index + kernel_height_index - 1

                            input_columns[row_index, column_index] = padded_input[
                                input_height_index,
                                input_width_index,
                                channel_index,
                                batch_index,
                            ]
                            row_index += 1
                        end
                    end
                end
            end
        end
    end

    return nothing
end

function col2im!(
    padded_input_gradient::Array{Float32,4},
    input_columns_gradient::Matrix{Float32},
    kernel_height::Int,
    kernel_width::Int,
)
    padded_height, padded_width, input_channels, batch_size = size(padded_input_gradient)

    output_height = padded_height - kernel_height + 1
    output_width = padded_width - kernel_width + 1

    fill!(padded_input_gradient, 0f0)

    @inbounds for batch_index in 1:batch_size
        batch_column_offset = (batch_index - 1) * output_height * output_width

        for output_width_index in 1:output_width
            for output_height_index in 1:output_height
                column_index = (
                    batch_column_offset +
                    (output_width_index - 1) * output_height +
                    output_height_index
                )
                row_index = 1

                for channel_index in 1:input_channels
                    for kernel_width_index in 1:kernel_width
                        input_width_index = output_width_index + kernel_width_index - 1

                        for kernel_height_index in 1:kernel_height
                            input_height_index = output_height_index + kernel_height_index - 1

                            padded_input_gradient[
                                input_height_index,
                                input_width_index,
                                channel_index,
                                batch_index,
                            ] += input_columns_gradient[row_index, column_index]
                            row_index += 1
                        end
                    end
                end
            end
        end
    end

    return nothing
end

# ========================= matrix <-> tensor =========================

function matrix_to_4d!(
    output_data::AbstractArray{Float32,4},
    output_matrix::Matrix{Float32},
)
    output_height, output_width, output_channels, batch_size = size(output_data)

    @inbounds for batch_index in 1:batch_size
        batch_column_offset = (batch_index - 1) * output_height * output_width

        for output_width_index in 1:output_width
            for output_height_index in 1:output_height
                column_index = (
                    batch_column_offset +
                    (output_width_index - 1) * output_height +
                    output_height_index
                )

                for channel_index in 1:output_channels
                    output_data[
                        output_height_index,
                        output_width_index,
                        channel_index,
                        batch_index,
                    ] = output_matrix[channel_index, column_index]
                end
            end
        end
    end

    return nothing
end

function gradient_to_matrix!(
    output_gradient_matrix::Matrix{Float32},
    output_gradient::AbstractArray{Float32,4},
)
    output_height, output_width, output_channels, batch_size = size(output_gradient)

    @inbounds for batch_index in 1:batch_size
        batch_column_offset = (batch_index - 1) * output_height * output_width

        for output_width_index in 1:output_width
            for output_height_index in 1:output_height
                column_index = (
                    batch_column_offset +
                    (output_width_index - 1) * output_height +
                    output_height_index
                )

                for channel_index in 1:output_channels
                    output_gradient_matrix[channel_index, column_index] = output_gradient[
                        output_height_index,
                        output_width_index,
                        channel_index,
                        batch_index,
                    ]
                end
            end
        end
    end

    return nothing
end

# ========================= Conv forward / backward =========================

function forward!(layer::ConvLayer, input_node::GraphNode)
    input_data = input_node.data
    weight_data = layer.weights.data
    output_data = layer.output.data

    kernel_height, kernel_width, input_channels, output_channels = size(weight_data)

    copy_with_padding!(layer.padded_input, input_data, layer.padding)
    im2col!(layer.input_columns, layer.padded_input, kernel_height, kernel_width)

    weight_matrix = reshape(
        weight_data,
        kernel_height * kernel_width * input_channels,
        output_channels,
    )

    mul!(layer.output_matrix, transpose(weight_matrix), layer.input_columns)

    matrix_to_4d!(output_data, layer.output_matrix)

    if layer.has_bias
        bias_data = layer.bias.data

        @inbounds for batch_index in axes(output_data, 4)
            for channel_index in axes(output_data, 3)
                channel_bias = bias_data[channel_index]

                for width_index in axes(output_data, 2)
                    for height_index in axes(output_data, 1)
                        output_data[
                            height_index,
                            width_index,
                            channel_index,
                            batch_index,
                        ] += channel_bias
                    end
                end
            end
        end
    end

    return nothing
end

function backward!(layer::ConvLayer, input_node::GraphNode)
    weight_data = layer.weights.data
    weight_gradient = layer.weights.grad

    kernel_height, kernel_width, input_channels, output_channels = size(weight_data)

    weight_matrix = reshape(
        weight_data,
        kernel_height * kernel_width * input_channels,
        output_channels,
    )
    weight_gradient_matrix = reshape(
        weight_gradient,
        kernel_height * kernel_width * input_channels,
        output_channels,
    )

    gradient_to_matrix!(layer.output_gradient_matrix, layer.output.grad)

    mul!(
        weight_gradient_matrix,
        layer.input_columns,
        transpose(layer.output_gradient_matrix),
        1f0,
        1f0,
    )

    if layer.has_bias
        bias_gradient = layer.bias.grad
        output_gradient_matrix = layer.output_gradient_matrix

        @inbounds for channel_index in 1:output_channels
            channel_gradient_sum = 0f0
            for column_index in axes(output_gradient_matrix, 2)
                channel_gradient_sum += output_gradient_matrix[channel_index, column_index]
            end
            bias_gradient[channel_index] += channel_gradient_sum
        end
    end

    mul!(
        layer.input_columns_gradient,
        weight_matrix,
        layer.output_gradient_matrix,
    )

    col2im!(
        layer.padded_input_gradient,
        layer.input_columns_gradient,
        kernel_height,
        kernel_width,
    )

    add_unpadded_gradient!(
        input_node.grad,
        layer.padded_input_gradient,
        layer.padding,
    )

    return nothing
end