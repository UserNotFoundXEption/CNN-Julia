struct LogitCrossEntropy{P, O}
    probabilities::GraphNode{P}
    output::GraphNode{O}
end

function LogitCrossEntropy(memory_pool::MemoryPool, number_of_classes::Int, batch_size::Int)
    probabilities_node = alloc_act!(memory_pool, number_of_classes, batch_size)
    loss_output_node = alloc_act!(memory_pool, 1)

    return LogitCrossEntropy(probabilities_node, loss_output_node)
end

function onehot!(target_node::GraphNode, class_labels::AbstractVector{<:Integer})
    target_data = target_node.data
    number_of_classes, batch_size = size(target_data)

    fill!(target_data, 0f0)

    @inbounds for batch_index in 1:batch_size
        class_index = Int(class_labels[batch_index])
        target_data[class_index, batch_index] = 1f0
    end

    return target_node
end

function forward!(
    layer::LogitCrossEntropy,
    logits_node::GraphNode,
    target_node::GraphNode,
)
    logit_values = logits_node.data
    target_values = target_node.data
    probabilities = layer.probabilities.data

    number_of_classes, batch_size = size(logit_values)
    total_loss = 0f0

    @inbounds for batch_index in 1:batch_size
        maximum_logit = -Inf32
        for class_index in 1:number_of_classes
            maximum_logit = max(maximum_logit, logit_values[class_index, batch_index])
        end

        shifted_exponential_sum = 0f0
        for class_index in 1:number_of_classes
            shifted_exponential = exp(
                logit_values[class_index, batch_index] - maximum_logit,
            )
            probabilities[class_index, batch_index] = shifted_exponential
            shifted_exponential_sum += shifted_exponential
        end

        log_sum_exp = maximum_logit + log(shifted_exponential_sum)
        inverse_exponential_sum = 1f0 / shifted_exponential_sum

        for class_index in 1:number_of_classes
            probabilities[class_index, batch_index] *= inverse_exponential_sum
            total_loss += target_values[class_index, batch_index] * (
                log_sum_exp - logit_values[class_index, batch_index]
            )
        end
    end

    layer.output.data[1] = total_loss / Float32(batch_size)

    return layer.output.data[1]
end

function backward!(
    layer::LogitCrossEntropy,
    logits_node::GraphNode,
    target_node::GraphNode,
)
    logits_gradient = logits_node.grad
    target_values = target_node.data
    probabilities = layer.probabilities.data

    number_of_classes, batch_size = size(logits_gradient)
    gradient_scale = layer.output.grad[1] / Float32(batch_size)

    @inbounds for batch_index in 1:batch_size
        for class_index in 1:number_of_classes
            logits_gradient[class_index, batch_index] += (
                probabilities[class_index, batch_index] - target_values[class_index, batch_index]
            ) * gradient_scale
        end
    end

    return nothing
end
