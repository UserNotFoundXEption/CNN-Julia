push!(LOAD_PATH, "src")
using AWID
using MLDatasets
using Random
using Statistics
using LinearAlgebra
using Dates
using BenchmarkTools

Random.seed!(42)

train_x, train_y = FashionMNIST(split=:train)[:]
test_x, test_y   = FashionMNIST(split=:test)[:]

settings = (
    lr = 0.01f0,
    epochs = 3,
    batch_size = 10,
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

function make_batches(idxs, batch_size)
    batches = Vector{Vector{Int}}()
    for start in 1:batch_size:length(idxs)
        stop = min(start + batch_size - 1, length(idxs))
        push!(batches, idxs[start:stop])
    end
    return batches
end

function make_model()
    conv1 = AWID.Conv(3, 3, 1, 6; pad=1)  
    pool1 = AWID.MaxPool(2, 2)               

    conv2 = AWID.Conv(3, 3, 6, 16; pad=1)   
    pool2 = AWID.MaxPool(2, 2)              

    fc1 = AWID.Dense(7 * 7 * 16, 84)   
    drop1 = AWID.Dropout(0.4f0)     
    fc2 = AWID.Dense(84, 10)

    params = [
        conv1.W,
        conv2.W,
        fc1.W,
        fc2.W
    ]

    return (
        conv1=conv1,
        pool1=pool1,
        conv2=conv2,
        pool2=pool2,
        fc1=fc1,
        drop1=drop1,
        fc2=fc2,
        params=params
    )
end

function model_forward(m, xnode; training=true)
    h = AWID.forward(m.conv1, xnode)
    h = AWID.relu(h)
    h = AWID.forward(m.pool1, h)

    h = AWID.forward(m.conv2, h)
    h = AWID.relu(h)
    h = AWID.forward(m.pool2, h)

    h = AWID.flatten(h)
    h = AWID.forward(m.fc1, h)
    h = AWID.relu(h)
    h = AWID.forward(m.drop1, h; training=training)
    h = AWID.forward(m.fc2, h)

    return h
end

function benchmark_model_parts()
    m = make_model()

    batch_idxs = 1:10
    x, y = get_batch(train_x_f, train_y, collect(batch_idxs))

    xnode = AWID.Node(x)
    training = true

    println("\n--- model_forward total ---")
    @btime model_forward($m, $xnode; training=$training)

    println("\n--- conv1 ---")
    @btime AWID.forward($(m.conv1), $xnode)

    h1 = AWID.forward(m.conv1, xnode)

    println("\n--- relu1 ---")
    @btime AWID.relu($h1)

    r1 = AWID.relu(h1)

    println("\n--- pool1 ---")
    @btime AWID.forward($(m.pool1), $r1)

    p1 = AWID.forward(m.pool1, r1)

    println("\n--- conv2 ---")
    @btime AWID.forward($(m.conv2), $p1)

    h2 = AWID.forward(m.conv2, p1)

    println("\n--- relu2 ---")
    @btime AWID.relu($h2)

    r2 = AWID.relu(h2)

    println("\n--- pool2 ---")
    @btime AWID.forward($(m.pool2), $r2)

    p2 = AWID.forward(m.pool2, r2)

    println("\n--- flatten ---")
    @btime AWID.flatten($p2)

    f = AWID.flatten(p2)

    println("\n--- fc1 ---")
    @btime AWID.forward($(m.fc1), $f)

    fc1 = AWID.forward(m.fc1, f)

    println("\n--- relu3 ---")
    @btime AWID.relu($fc1)

    r3 = AWID.relu(fc1)

    println("\n--- dropout ---")
    @btime AWID.forward($(m.drop1), $r3; training=$training)

    d1 = AWID.forward(m.drop1, r3; training=training)

    println("\n--- fc2 ---")
    @btime AWID.forward($(m.fc2), $d1)

    return nothing
end

function benchmark_train_step_parts()
    m = make_model()
    batch_idxs = collect(1:10)

    x, y = get_batch(train_x_f, train_y, batch_idxs)
    xnode = AWID.Node(x)

    println("\n--- forward only ---")
    @btime model_forward($m, $xnode; training=true)

    logits = model_forward(m, xnode; training=true)
    loss = AWID.cross_entropy(logits, y)

    println("\n--- backward only ---")
    @btime AWID.backward!($loss)

    println("\n--- full train step ---")
    @btime begin
        AWID.zero_grad!($(m.params))
        logits = model_forward($m, AWID.Node($x); training=true)
        loss = AWID.cross_entropy(logits, $y)
        AWID.backward!(loss)
        AWID.sgd!($(m.params), 0.01f0)
    end

    return nothing
end

function benchmark_backward()
    m = make_model()
    batch_idxs = collect(1:10)

    x, y = get_batch(train_x_f, train_y, batch_idxs)

    println("\n--- backward full model ---")
    @btime begin
        xnode = AWID.Node($x)
        logits = model_forward($m, xnode; training=true)
        loss = AWID.cross_entropy(logits, $y)
        AWID.backward!(loss)
    end

    xnode = AWID.Node(x)

    h1 = AWID.forward(m.conv1, xnode)
    r1 = AWID.relu(h1)
    p1 = AWID.forward(m.pool1, r1)

    h2 = AWID.forward(m.conv2, p1)
    r2 = AWID.relu(h2)
    p2 = AWID.forward(m.pool2, r2)

    f = AWID.flatten(p2)
    fc1 = AWID.forward(m.fc1, f)
    r3 = AWID.relu(fc1)
    d1 = AWID.forward(m.drop1, r3; training=true)
    fc2 = AWID.forward(m.fc2, d1)

    println("\n--- fc2 backward_fn ---")
    fill!(fc2.grad, 1f0)
    @btime $(fc2.backward_fn)()

    println("\n--- dropout backward_fn ---")
    fill!(d1.grad, 1f0)
    @btime $(d1.backward_fn)()

    println("\n--- relu3 backward_fn ---")
    fill!(r3.grad, 1f0)
    @btime $(r3.backward_fn)()

    println("\n--- fc1 backward_fn ---")
    fill!(fc1.grad, 1f0)
    @btime $(fc1.backward_fn)()

    println("\n--- flatten backward_fn ---")
    fill!(f.grad, 1f0)
    @btime $(f.backward_fn)()

    println("\n--- pool2 backward_fn ---")
    fill!(p2.grad, 1f0)
    @btime $(p2.backward_fn)()

    println("\n--- relu2 backward_fn ---")
    fill!(r2.grad, 1f0)
    @btime $(r2.backward_fn)()

    println("\n--- conv2 backward_fn ---")
    fill!(h2.grad, 1f0)
    @btime $(h2.backward_fn)()

    println("\n--- pool1 backward_fn ---")
    fill!(p1.grad, 1f0)
    @btime $(p1.backward_fn)()

    println("\n--- relu1 backward_fn ---")
    fill!(r1.grad, 1f0)
    @btime $(r1.backward_fn)()

    println("\n--- conv1 backward_fn ---")
    fill!(h1.grad, 1f0)
    @btime $(h1.backward_fn)()

    return nothing
end

function batch_step!(batch_idxs, train_x, train_y, model, lr)
    AWID.zero_grad!(model.params)

    x, y = get_batch(train_x, train_y, batch_idxs)
    logits = model_forward(model, AWID.Node(x); training=true)
    loss = AWID.cross_entropy(logits, y)

    AWID.backward!(loss)
    AWID.sgd!(model.params, lr)

    return loss.value
end

function accuracy(images, labels, idxs, model; batch_size=128)
    correct = 0
    total = 0

    batches = make_batches(collect(idxs), batch_size)

    for batch in batches
        x, y = get_batch(images, labels, batch)
        logits = model_forward(model, AWID.Node(x); training=false)

        for j in 1:length(y)
            pred = argmax(view(logits.value, :, j))
            correct += (pred == y[j])
        end
        total += length(y)
    end

    return correct / total
end

function mean_loss(images, labels, idxs, model; batch_size=128)
    total = 0f0
    total_count = 0

    batches = make_batches(collect(idxs), batch_size)

    for batch in batches
        x, y = get_batch(images, labels, batch)
        logits = model_forward(model, AWID.Node(x); training=false)
        loss = AWID.cross_entropy(logits, y)

        total += loss.value * length(y)
        total_count += length(y)
    end

    return total / total_count
end

model = make_model()

train_idxs = collect(1:length(train_y))
test_idxs  = collect(1:length(test_y))

train_eval_idxs = collect(1:1000)
test_eval_idxs  = collect(1:1000)

println("train size = ", length(train_idxs))
println("test size  = ", length(test_idxs))

for epoch in 1:settings.epochs
    epoch_start = time()

    shuffle!(train_idxs)
    batches = make_batches(train_idxs, settings.batch_size)
    total_batches = length(batches)

    train_start = time()

    for (batch_count, batch) in enumerate(batches)
        batch_step!(batch, train_x_f, train_y, model, settings.lr)

        if batch_count % 1000 == 0 || batch_count == total_batches
            println("epoch=$epoch batch=$batch_count/$total_batches")
        end
    end

    train_time = time() - train_start

    eval_start = time()

    train_acc  = accuracy(train_x_f, train_y, train_eval_idxs, model)
    test_acc   = accuracy(test_x_f, test_y, test_eval_idxs, model)

    train_loss = mean_loss(train_x_f, train_y, train_eval_idxs, model)
    test_loss  = mean_loss(test_x_f, test_y, test_eval_idxs, model)

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