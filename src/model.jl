struct CompiledModel{C, P, I, TG, L, IS, OS}
    chain::C
    pool::P
    input::I
    target::TG
    loss::L
    batch_size::Int
    input_shape::IS
    output_shape::OS
end

build_layer(
    blueprint::Blueprint,
    memory_pool::MemoryPool,
    input_shape::Tuple,
    batch_size::Int,
    previous_output,
) = build_layer(blueprint, memory_pool, input_shape, batch_size)

function build_model(definition::ChainDef, input_shape::Tuple; batch_size::Int=1)
    memory_pool = MemoryPool()

    input_node = alloc_act!(memory_pool, input_shape..., batch_size)

    compiled_layers = Any[]
    current_shape = input_shape
    previous_output = input_node

    for blueprint in definition.blueprints
        layer, current_shape = build_layer(
            blueprint,
            memory_pool,
            current_shape,
            batch_size,
            previous_output,
        )
        push!(compiled_layers, layer)
        previous_output = layer.output
    end

    number_of_classes = current_shape[1]

    target_node = alloc_act!(memory_pool, number_of_classes, batch_size)
    loss_layer = LogitCrossEntropy(memory_pool, number_of_classes, batch_size)

    chain = StaticChain(Tuple(compiled_layers))

    return CompiledModel(
        chain,
        memory_pool,
        input_node,
        target_node,
        loss_layer,
        batch_size,
        input_shape,
        current_shape,
    )
end

function forward_train!(model::CompiledModel)
    return forward_train!(model.chain, model.input)
end

function forward_test!(model::CompiledModel)
    return forward_test!(model.chain, model.input)
end

function model_output(model::CompiledModel)
    return last(model.chain.layers).output
end

function loss!(model::CompiledModel, logits_node::GraphNode)
    return forward!(model.loss, logits_node, model.target)
end

function loss!(model::CompiledModel)
    return loss!(model, model_output(model))
end

function backward!(model::CompiledModel, logits_node::GraphNode)
    model.loss.output.grad[1] = 1f0

    backward!(model.loss, logits_node, model.target)
    backward!(model.chain, model.input)

    return nothing
end

function backward!(model::CompiledModel)
    return backward!(model, model_output(model))
end

function zero_grad!(model::CompiledModel)
    zero_grad!(model.pool)
    return nothing
end

function optimize!(model::CompiledModel, learning_rate::Float32)
    optimize!(model.pool, Float32(learning_rate))
    return nothing
end
