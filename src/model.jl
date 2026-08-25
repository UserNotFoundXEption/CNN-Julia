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

build_layer(bp::Blueprint, pool::MemoryPool, in_shape::Tuple, batch_size::Int, prev_out) =
    build_layer(bp, pool, in_shape, batch_size)

function build_model(def::ChainDef, input_shape::Tuple; batch_size::Int=1)
    pool = MemoryPool()

    input = alloc_act!(pool, input_shape..., batch_size)

    compiled_layers = Any[]
    current_shape = input_shape
    prev_out = input

    for bp in def.blueprints
        layer, current_shape = build_layer(bp, pool, current_shape, batch_size, prev_out)
        push!(compiled_layers, layer)
        prev_out = layer.out
    end

    num_classes = current_shape[1]

    target = alloc_act!(pool, num_classes, batch_size)
    loss = LogitCrossEntropy(pool, num_classes, batch_size)

    chain = StaticChain(Tuple(compiled_layers))

    return CompiledModel(
        chain,
        pool,
        input,
        target,
        loss,
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
    return last(model.chain.layers).out
end

function loss!(model::CompiledModel, logits::GraphNode)
    return primal!(model.loss, logits, model.target)
end

function loss!(model::CompiledModel)
    return loss!(model, model_output(model))
end

function backward!(model::CompiledModel, logits::GraphNode)
    model.loss.out.grad[1] = 1f0

    adjoint!(model.loss, logits, model.target)
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

function optimize!(model::CompiledModel, η::Real)
    optimize!(model.pool, Float32(η))
    return nothing
end