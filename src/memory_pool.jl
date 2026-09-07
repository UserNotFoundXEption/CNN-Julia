mutable struct MemoryPool
    weights::Vector{Float32}
    weight_gradients::Vector{Float32}
    weight_offset::Int

    activations::Vector{Float32}
    activation_gradients::Vector{Float32}
    activation_offset::Int
end

MemoryPool() = MemoryPool(
    Float32[],
    Float32[],
    1,
    Float32[],
    Float32[],
    1,
)

struct GraphNode{T}
    data::T
    grad::T
end

function alloc_weight!(memory_pool::MemoryPool, dimensions...)
    number_of_values = prod(dimensions)
    start_index = memory_pool.weight_offset
    end_index = start_index + number_of_values - 1

    memory_pool.weight_offset += number_of_values

    append!(memory_pool.weights, zeros(Float32, number_of_values))
    append!(memory_pool.weight_gradients, zeros(Float32, number_of_values))

    data = reshape(view(memory_pool.weights, start_index:end_index), dimensions)
    gradient = reshape(
        view(memory_pool.weight_gradients, start_index:end_index),
        dimensions,
    )

    return GraphNode(data, gradient)
end

function alloc_act!(memory_pool::MemoryPool, dimensions...)
    number_of_values = prod(dimensions)
    start_index = memory_pool.activation_offset
    end_index = start_index + number_of_values - 1

    memory_pool.activation_offset += number_of_values

    append!(memory_pool.activations, zeros(Float32, number_of_values))
    append!(memory_pool.activation_gradients, zeros(Float32, number_of_values))

    data = reshape(view(memory_pool.activations, start_index:end_index), dimensions)
    gradient = reshape(
        view(memory_pool.activation_gradients, start_index:end_index),
        dimensions,
    )

    return GraphNode(data, gradient)
end

function zero_w_grad!(memory_pool::MemoryPool)
    fill!(memory_pool.weight_gradients, 0.0f0)
    return nothing
end

function zero_a_grad!(memory_pool::MemoryPool)
    fill!(memory_pool.activation_gradients, 0.0f0)
    return nothing
end

function zero_grad!(memory_pool::MemoryPool)
    zero_w_grad!(memory_pool)
    zero_a_grad!(memory_pool)
    return nothing
end

function optimize!(memory_pool::MemoryPool, learning_rate::Float32)
    @inbounds for parameter_index in eachindex(memory_pool.weights)
        memory_pool.weights[parameter_index] -= (
            learning_rate * memory_pool.weight_gradients[parameter_index]
        )
    end
    return nothing
end

optimize!(memory_pool::MemoryPool, learning_rate::Real) =
    optimize!(memory_pool, Float32(learning_rate))
