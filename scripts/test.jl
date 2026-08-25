push!(LOAD_PATH, "src")
using AWID
using Random

Random.seed!(42)

model_def = AWID.Chain(
    AWID.Conv((3, 3), 1 => 6; pad=1, bias=false),
    AWID.MaxPool((2, 2)),
    AWID.Conv((3, 3), 6 => 16; pad=1, bias=false),
    AWID.MaxPool((2, 2)),
    AWID.Flatten(),
    AWID.Dense(7 * 7 * 16 => 84, AWID.relu),
    AWID.Dropout(0.4f0),
    AWID.Dense(84 => 10),
)

model = AWID.build_model(model_def, (28, 28, 1); batch_size=10)

x = rand(Float32, 28, 28, 1, 10)
y = rand(1:10, 10)

AWID.zero_grad!(model)

copyto!(model.input.data, x)
AWID.onehot!(model.target, y)

logits = AWID.forward_train!(model)
loss = AWID.loss!(model, logits)

@show loss
@show size(logits.data)
@show model.output_shape
@show length(model.pool.weights)

AWID.backward!(model, logits)

@show sum(abs, model.pool.w_grad)
@show sum(abs, model.input.grad)

AWID.optimize!(model, 0.01f0)