struct LogitCrossEntropy{P, O}
    probs::GraphNode{P}
    out::GraphNode{O}
end

function LogitCrossEntropy(pool::MemoryPool, num_classes::Int, batch_size::Int)
    probs = alloc_act!(pool, num_classes, batch_size)
    out = alloc_act!(pool, 1)

    return LogitCrossEntropy(probs, out)
end

function onehot!(target::GraphNode, y::AbstractVector{<:Integer})
    td = target.data
    C, B = size(td)

    fill!(td, 0f0)

    @inbounds for b in 1:B
        cls = Int(y[b])
        td[cls, b] = 1f0
    end

    return target
end

function primal!(layer::LogitCrossEntropy, logits::GraphNode, target::GraphNode)
    z = logits.data
    y = target.data
    probs = layer.probs.data

    C, B = size(z)

    total_loss = 0f0

    @inbounds for b in 1:B
        m = -Inf32
        for i in 1:C
            m = max(m, z[i, b])
        end

        sum_exp = 0f0
        for i in 1:C
            e = exp(z[i, b] - m)
            probs[i, b] = e
            sum_exp += e
        end

        log_sum_exp = m + log(sum_exp)
        inv_sum_exp = 1f0 / sum_exp

        for i in 1:C
            probs[i, b] *= inv_sum_exp
            total_loss += y[i, b] * (log_sum_exp - z[i, b])
        end
    end

    layer.out.data[1] = total_loss / Float32(B)

    return layer.out.data[1]
end

function adjoint!(layer::LogitCrossEntropy, logits::GraphNode, target::GraphNode)
    zg = logits.grad
    y = target.data
    probs = layer.probs.data

    C, B = size(zg)

    scale = layer.out.grad[1] / Float32(B)

    @inbounds for b in 1:B
        for i in 1:C
            zg[i, b] += (probs[i, b] - y[i, b]) * scale
        end
    end

    return nothing
end