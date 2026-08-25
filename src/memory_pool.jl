mutable struct MemoryPool
    weights::Vector{Float32}
    w_grad::Vector{Float32}
    w_offset::Int

    acts::Vector{Float32}
    a_grad::Vector{Float32}
    a_offset::Int
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

function alloc_weight!(pool::MemoryPool, dims...)
    len = prod(dims)
    start = pool.w_offset
    stop = start + len - 1

    pool.w_offset += len

    append!(pool.weights, zeros(Float32, len))
    append!(pool.w_grad, zeros(Float32, len))

    data = reshape(view(pool.weights, start:stop), dims)
    grad = reshape(view(pool.w_grad, start:stop), dims)

    return GraphNode(data, grad)
end

function alloc_act!(pool::MemoryPool, dims...)
    len = prod(dims)
    start = pool.a_offset
    stop = start + len - 1

    pool.a_offset += len

    append!(pool.acts, zeros(Float32, len))
    append!(pool.a_grad, zeros(Float32, len))

    data = reshape(view(pool.acts, start:stop), dims)
    grad = reshape(view(pool.a_grad, start:stop), dims)

    return GraphNode(data, grad)
end

function zero_w_grad!(pool::MemoryPool)
    fill!(pool.w_grad, 0.0f0)
    return nothing
end

function zero_a_grad!(pool::MemoryPool)
    fill!(pool.a_grad, 0.0f0)
    return nothing
end

function zero_grad!(pool::MemoryPool)
    zero_w_grad!(pool)
    zero_a_grad!(pool)
    return nothing
end

function optimize!(pool::MemoryPool, η::Float32)
    @inbounds for i in eachindex(pool.weights)
        pool.weights[i] -= η * pool.w_grad[i]
    end
    return nothing
end

optimize!(pool::MemoryPool, η::Real) = optimize!(pool, Float32(η))