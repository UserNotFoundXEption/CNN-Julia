function he_init_dense(input_dimension::Int, output_dimension::Int)
    return randn(Float32, output_dimension, input_dimension) .* sqrt(2f0 / input_dimension)
end

function he_init_conv(kernel_height::Int, kernel_width::Int, input_channels::Int, output_channels::Int)
    fan_in = kernel_height * kernel_width * input_channels
    return randn(Float32, kernel_height, kernel_width, input_channels, output_channels) .* sqrt(2f0 / fan_in)
end

function he_uniform!(values; fan_in::Int)
    scale = Float32(sqrt(24.0 / fan_in))
    values .= (rand(Float32, size(values)...) .- 0.5f0) .* scale
    return values
end

function he_normal!(values; fan_in::Int)
    values .= randn(Float32, size(values)...) .* sqrt(2f0 / fan_in)
    return values
end
