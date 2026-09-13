push!(LOAD_PATH, "src")

using AWID
using MLDatasets
using Random

Random.seed!(42)

train_data = MLDatasets.FashionMNIST(split=:train)
test_data  = MLDatasets.FashionMNIST(split=:test)

settings = (;
    eta = 0.01f0,
    epochs = 3,
    batchsize = 10,
)

function loader(data; batchsize::Int=1, shuffle::Bool=true)
    indices = collect(1:length(data.targets))

    if shuffle
        shuffle!(indices)
    end

    usable = length(indices) - length(indices) % batchsize
    indices = indices[1:usable]

    return (
        begin
            batch = collect(batch_indices)

            x = reshape(
                Float32.(data.features[:, :, batch]) ./ 255f0,
                28,
                28,
                1,
                batchsize,
            )

            y = Int.(data.targets[batch]) .+ 1

            (x, y)
        end
        for batch_indices in Iterators.partition(indices, batchsize)
    )
end


net_def = AWID.Chain(
    AWID.Conv((3, 3), 1 => 6; padding=1, bias=false),
    AWID.MaxPool((2, 2)),

    AWID.Conv((3, 3), 6 => 16; padding=1, bias=false),
    AWID.MaxPool((2, 2)),

    AWID.Flatten(),

    AWID.Dense(784 => 84, AWID.relu),
    AWID.Dropout(0.4f0),

    AWID.Dense(84 => 10),
)

net = AWID.build_model(
    net_def,
    (28, 28, 1);
    batch_size=settings.batchsize,
)


function predict_classes(logits)
    batchsize = size(logits.data, 2)

    return [
        argmax(view(logits.data, :, sample)) - 1
        for sample in 1:batchsize
    ]
end


function train_step!(model, x, y)
    AWID.zero_grad!(model)

    copyto!(model.input.data, x)
    AWID.onehot!(model.target, y)

    logits = AWID.forward_train!(model)
    loss = AWID.loss!(model, logits)

    AWID.backward!(model, logits)
    AWID.optimize!(model, settings.eta)

    return loss
end


function loss_and_accuracy(model, data)
    total_loss = 0f0
    correct = 0
    total = 0

    for (x, y) in loader(
        data;
        batchsize=settings.batchsize,
        shuffle=false,
    )
        copyto!(model.input.data, x)
        AWID.onehot!(model.target, y)

        y_hat = AWID.forward_test!(model)
        loss = AWID.loss!(model, y_hat)

        predictions = predict_classes(y_hat)
        labels = y .- 1

        total_loss += loss * length(y)
        correct += sum(predictions .== labels)
        total += length(y)
    end

    loss = total_loss / total
    acc = round(100 * correct / total; digits=2)

    return (; loss, acc, split=data.split)
end


println("parameters = ", length(net.pool.weights))


x1, y1 = first(
    loader(
        train_data;
        batchsize=settings.batchsize,
        shuffle=false,
    )
)

copyto!(net.input.data, x1)
y1hat = AWID.forward_test!(net)

@show hcat(
    predict_classes(y1hat),
    y1 .- 1,
)

@show loss_and_accuracy(net, test_data)


train_log = []
accuracy = zeros(settings.epochs, 2)


for epoch in 1:settings.epochs
    @time for (x, y) in loader(
        train_data;
        batchsize=settings.batchsize,
        shuffle=true,
    )
        train_step!(net, x, y)
    end

    loss, acc, _ = loss_and_accuracy(
        net,
        train_data,
    )

    test_loss, test_acc, _ = loss_and_accuracy(
        net,
        test_data,
    )

    @info epoch acc test_acc

    nt = (;
        epoch,
        loss,
        acc,
        test_loss,
        test_acc,
    )

    push!(train_log, nt)

    accuracy[epoch, 1] = acc
    accuracy[epoch, 2] = test_acc
end


x1, y1 = first(
    loader(
        test_data;
        batchsize=settings.batchsize,
        shuffle=false,
    )
)

copyto!(net.input.data, x1)
y1hat = AWID.forward_test!(net)

@show hcat(
    predict_classes(y1hat),
    y1 .- 1,
)

@show loss_and_accuracy(net, test_data)
