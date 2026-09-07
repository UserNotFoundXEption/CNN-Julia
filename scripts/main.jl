push!(LOAD_PATH, "src")

using AWID
using MLDatasets
using Random
using Statistics
using BenchmarkTools

Random.seed!(42)

train_x, train_y = FashionMNIST(split=:train)[:]
test_x, test_y   = FashionMNIST(split=:test)[:]

settings = (
    lr = 0.01f0,
    epochs = 3,
    batch_size = 1,
)

function preprocess_images(images)
    n = size(images, 3)
    out = Array{Float32,4}(undef, 28, 28, 1, n)

    @inbounds for i in 1:n
        out[:, :, 1, i] .= Float32.(images[:, :, i]) ./ 255f0
    end

    return out
end

train_x_f = preprocess_images(train_x)
test_x_f  = preprocess_images(test_x)

function get_batch(images, labels, idxs::Vector{Int})
    x = images[:, :, :, idxs]

    y = labels[idxs] .+ 1

    return x, y
end

function make_batches(idxs::Vector{Int}, batch_size::Int)
    batches = Vector{Vector{Int}}()

    for start in 1:batch_size:length(idxs)
        stop = min(start + batch_size - 1, length(idxs))

        if stop - start + 1 == batch_size
            push!(batches, idxs[start:stop])
        end
    end

    return batches
end

function make_model(batch_size::Int)
    model_def = AWID.Chain(
        AWID.Conv((3, 3), 1 => 6; padding=1, bias=false),
        AWID.MaxPool((2, 2)),

        AWID.Conv((3, 3), 6 => 16; padding=1, bias=false),
        AWID.MaxPool((2, 2)),

        AWID.Flatten(),

        AWID.Dense(7 * 7 * 16 => 84, AWID.relu),
        AWID.Dropout(0.4f0),

        AWID.Dense(84 => 10),
    )

    return AWID.build_model(model_def, (28, 28, 1); batch_size=batch_size)
end

function train_step!(model, x, y, lr)
    AWID.zero_grad!(model)

    copyto!(model.input.data, x)
    AWID.onehot!(model.target, y)

    logits = AWID.forward_train!(model)
    loss = AWID.loss!(model, logits)

    AWID.backward!(model, logits)
    AWID.optimize!(model, lr)

    return loss
end

function batch_loss!(model, x, y)
    copyto!(model.input.data, x)
    AWID.onehot!(model.target, y)

    logits = AWID.forward_test!(model)
    loss = AWID.loss!(model, logits)

    return loss
end

function accuracy(images, labels, idxs, model; batch_size::Int)
    correct = 0
    total = 0

    batches = make_batches(collect(idxs), batch_size)

    for batch in batches
        x, y = get_batch(images, labels, batch)

        copyto!(model.input.data, x)

        logits = AWID.forward_test!(model)

        @inbounds for j in 1:length(y)
            pred = argmax(view(logits.data, :, j))
            correct += pred == y[j]
        end

        total += length(y)
    end

    return correct / total
end

function mean_loss(images, labels, idxs, model; batch_size::Int)
    total_loss = 0f0
    total_count = 0

    batches = make_batches(collect(idxs), batch_size)

    for batch in batches
        x, y = get_batch(images, labels, batch)

        loss = batch_loss!(model, x, y)

        total_loss += loss * length(y)
        total_count += length(y)
    end

    return total_loss / total_count
end

model = make_model(settings.batch_size)

println("parameters = ", length(model.pool.weights))
println("train size = ", length(train_y))
println("test size  = ", length(test_y))

train_idxs = collect(1:length(train_y))
test_idxs  = collect(1:length(test_y))

train_eval_idxs = train_idxs
test_eval_idxs = test_idxs

@for epoch in 1:settings.epochs
    epoch_start = time()

    shuffle!(train_idxs)
    batches = make_batches(train_idxs, settings.batch_size)
    total_batches = length(batches)

    train_start = time()

    running_loss = 0f0

    for (batch_count, batch) in enumerate(batches)
        x, y = get_batch(train_x_f, train_y, batch)

        loss = train_step!(model, x, y, settings.lr)
        running_loss += loss

        # if batch_count % 1000 == 0 || batch_count == total_batches
        #     avg_loss = running_loss / batch_count

        #     println(
        #         "epoch=$epoch batch=$batch_count/$total_batches " *
        #         "avg_loss=$(round(avg_loss, digits=4))"
        #     )
        # end
    end

    train_time = time() - train_start

    eval_start = time()

    train_acc = accuracy(
        train_x_f,
        train_y,
        train_eval_idxs,
        model;
        batch_size=settings.batch_size,
    )

    test_acc = accuracy(
        test_x_f,
        test_y,
        test_eval_idxs,
        model;
        batch_size=settings.batch_size,
    )

    train_loss = mean_loss(
        train_x_f,
        train_y,
        train_eval_idxs,
        model;
        batch_size=settings.batch_size,
    )

    test_loss = mean_loss(
        test_x_f,
        test_y,
        test_eval_idxs,
        model;
        batch_size=settings.batch_size,
    )

    eval_time = time() - eval_start
    epoch_time = time() - epoch_start

    println(
        "epoch=$epoch " *
        "train_loss=$(round(train_loss, digits=4)) " *
        "test_loss=$(round(test_loss, digits=4)) " *
        "train_acc=$(round(train_acc, digits=3)) " *
        "test_acc=$(round(test_acc, digits=3)) " *
        "| train_time=$(round(train_time, digits=2))s " *
        "eval_time=$(round(eval_time, digits=2))s " *
        "total=$(round(epoch_time, digits=2))s"
    )
end