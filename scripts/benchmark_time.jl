push!(LOAD_PATH, "src")

using AWID
using BenchmarkTools
using LinearAlgebra
using Random
using Statistics
using Printf

Random.seed!(42)
BLAS.set_num_threads(1)

const BATCH_SIZE = 10
const BENCHMARK_SAMPLES = 500
const BENCHMARK_SECONDS = 20.0
const PRINT_FULL_TRIALS = false


function format_time_nanoseconds(time_ns::Real)
    if time_ns < 1_000
        return @sprintf("%.2f ns", time_ns)
    elseif time_ns < 1_000_000
        return @sprintf("%.2f μs", time_ns / 1_000)
    elseif time_ns < 1_000_000_000
        return @sprintf("%.2f ms", time_ns / 1_000_000)
    else
        return @sprintf("%.3f s", time_ns / 1_000_000_000)
    end
end

function format_bytes(number_of_bytes::Integer)
    if number_of_bytes < 1024
        return "$(number_of_bytes) B"
    elseif number_of_bytes < 1024^2
        return @sprintf("%.2f KiB", number_of_bytes / 1024)
    elseif number_of_bytes < 1024^3
        return @sprintf("%.2f MiB", number_of_bytes / 1024^2)
    else
        return @sprintf("%.2f GiB", number_of_bytes / 1024^3)
    end
end

function build_test_layer(
    blueprint,
    input_shape::Tuple,
    batch_size::Int,
)
    memory_pool = AWID.MemoryPool()

    input_node = AWID.alloc_act!(
        memory_pool,
        input_shape...,
        batch_size,
    )

    randn!(input_node.data)

    layer, output_shape = AWID.build_layer(
        blueprint,
        memory_pool,
        input_shape,
        batch_size,
        input_node,
    )

    return memory_pool, input_node, layer, output_shape
end

function prepare_layer_backward!(
    memory_pool,
    layer,
    input_node,
)
    AWID.zero_grad!(memory_pool)

    AWID.forward_train!(
        layer,
        input_node,
    )

    fill!(layer.output.grad, 1f0)

    return nothing
end

function benchmark_layer!(
    results,
    layer_name::String,
    blueprint,
    input_shape::Tuple;
    batch_size::Int=BATCH_SIZE,
    samples::Int=BENCHMARK_SAMPLES,
    seconds::Real=BENCHMARK_SECONDS,
)
    memory_pool, input_node, layer, output_shape = build_test_layer(
        blueprint,
        input_shape,
        batch_size,
    )

    AWID.forward_train!(layer, input_node)

    prepare_layer_backward!(
        memory_pool,
        layer,
        input_node,
    )
    AWID.backward!(layer, input_node)

    GC.gc()

    forward_benchmark = @benchmarkable AWID.forward_train!(
        $layer,
        $input_node,
    ) evals=1

    forward_trial = BenchmarkTools.run(
        forward_benchmark;
        samples=samples,
        seconds=Float64(seconds),
    )

    backward_benchmark = @benchmarkable AWID.backward!(
        $layer,
        $input_node,
    ) setup=(
        prepare_layer_backward!(
            $memory_pool,
            $layer,
            $input_node,
        )
    ) evals=1

    backward_trial = BenchmarkTools.run(
        backward_benchmark;
        samples=samples,
        seconds=Float64(seconds),
    )

    forward_median = median(forward_trial)
    forward_minimum = minimum(forward_trial)

    backward_median = median(backward_trial)
    backward_minimum = minimum(backward_trial)

    push!(
        results,
        (
            layer=layer_name,
            operation="forward",
            input_shape=string((input_shape..., batch_size)),
            output_shape=string((output_shape..., batch_size)),
            minimum_time_ns=forward_minimum.time,
            median_time_ns=forward_median.time,
        ),
    )

    push!(
        results,
        (
            layer=layer_name,
            operation="backward",
            input_shape=string((input_shape..., batch_size)),
            output_shape=string((output_shape..., batch_size)),
            minimum_time_ns=backward_minimum.time,
            median_time_ns=backward_median.time,
        ),
    )

    if PRINT_FULL_TRIALS
        println("\n============================================================")
        println(layer_name, " | input=", (input_shape..., batch_size),
                " | output=", (output_shape..., batch_size))

        println("\nFORWARD:")
        display(forward_trial)

        println("\nBACKWARD:")
        display(backward_trial)
    end

    return nothing
end

function prepare_loss_backward!(
    memory_pool,
    loss_layer,
    logits_node,
    target_node,
)
    AWID.zero_grad!(memory_pool)

    AWID.forward!(
        loss_layer,
        logits_node,
        target_node,
    )

    loss_layer.output.grad[1] = 1f0

    return nothing
end

function benchmark_loss!(
    results;
    number_of_classes::Int=10,
    batch_size::Int=BATCH_SIZE,
    samples::Int=BENCHMARK_SAMPLES,
    seconds::Real=BENCHMARK_SECONDS,
)
    memory_pool = AWID.MemoryPool()

    logits_node = AWID.alloc_act!(
        memory_pool,
        number_of_classes,
        batch_size,
    )

    target_node = AWID.alloc_act!(
        memory_pool,
        number_of_classes,
        batch_size,
    )

    loss_layer = AWID.LogitCrossEntropy(
        memory_pool,
        number_of_classes,
        batch_size,
    )

    randn!(logits_node.data)

    labels = [
        ((sample_index - 1) % number_of_classes) + 1
        for sample_index in 1:batch_size
    ]

    AWID.onehot!(target_node, labels)

    AWID.forward!(
        loss_layer,
        logits_node,
        target_node,
    )

    prepare_loss_backward!(
        memory_pool,
        loss_layer,
        logits_node,
        target_node,
    )

    AWID.backward!(
        loss_layer,
        logits_node,
        target_node,
    )

    GC.gc()

    forward_benchmark = @benchmarkable AWID.forward!(
        $loss_layer,
        $logits_node,
        $target_node,
    ) evals=1

    forward_trial = BenchmarkTools.run(
        forward_benchmark;
        samples=samples,
        seconds=Float64(seconds),
    )

    backward_benchmark = @benchmarkable AWID.backward!(
        $loss_layer,
        $logits_node,
        $target_node,
    ) setup=(
        prepare_loss_backward!(
            $memory_pool,
            $loss_layer,
            $logits_node,
            $target_node,
        )
    ) evals=1

    backward_trial = BenchmarkTools.run(
        backward_benchmark;
        samples=samples,
        seconds=Float64(seconds),
    )

    forward_median = median(forward_trial)
    forward_minimum = minimum(forward_trial)

    backward_median = median(backward_trial)
    backward_minimum = minimum(backward_trial)

    shape_string = string((number_of_classes, batch_size))

    push!(
        results,
        (
            layer="LogitCrossEntropy",
            operation="forward",
            input_shape=shape_string,
            output_shape="(1,)",
            minimum_time_ns=forward_minimum.time,
            median_time_ns=forward_median.time,
        ),
    )

    push!(
        results,
        (
            layer="LogitCrossEntropy",
            operation="backward",
            input_shape=shape_string,
            output_shape="(1,)",
            minimum_time_ns=backward_minimum.time,
            median_time_ns=backward_median.time,
        ),
    )

    if PRINT_FULL_TRIALS
        println("\n============================================================")
        println("LogitCrossEntropy")

        println("\nFORWARD:")
        display(forward_trial)

        println("\nBACKWARD:")
        display(backward_trial)
    end

    return nothing
end

results = []

println("Time benchmark")
println("batch_size      = ", BATCH_SIZE)
println("BLAS threads    = ", BLAS.get_num_threads())
println("samples (max)   = ", BENCHMARK_SAMPLES)
println("seconds (limit) = ", BENCHMARK_SECONDS)
println()

benchmark_layer!(
    results,
    "Conv1 3x3 1→6",
    AWID.Conv((3, 3), 1 => 6; padding=1, bias=false),
    (28, 28, 1),
)

benchmark_layer!(
    results,
    "MaxPool1 2x2",
    AWID.MaxPool((2, 2)),
    (28, 28, 6),
)

benchmark_layer!(
    results,
    "Conv2 3x3 6→16",
    AWID.Conv((3, 3), 6 => 16; padding=1, bias=false),
    (14, 14, 6),
)

benchmark_layer!(
    results,
    "MaxPool2 2x2",
    AWID.MaxPool((2, 2)),
    (14, 14, 16),
)

benchmark_layer!(
    results,
    "Flatten 7x7x16",
    AWID.Flatten(),
    (7, 7, 16),
)

benchmark_layer!(
    results,
    "Dense 784→84",
    AWID.Dense(7 * 7 * 16 => 84),
    (7 * 7 * 16,),
)

benchmark_layer!(
    results,
    "ReLU 84",
    AWID.relu,
    (84,),
)

benchmark_layer!(
    results,
    "Dropout 84 p=0.4",
    AWID.Dropout(0.4f0),
    (84,),
)

benchmark_layer!(
    results,
    "Dense 84→10",
    AWID.Dense(84 => 10),
    (84,),
)

benchmark_loss!(
    results;
    number_of_classes=10,
)

println()
println("=================================================================================================================")
println("RESULTS")
println("=================================================================================================================")

@printf(
    "%-22s %-10s %-18s %-18s %12s %12s \n",
    "layer",
    "operation",
    "input",
    "output",
    "minimum",
    "mean",
)

println(repeat("-", 118))

for result in results
    @printf(
        "%-22s %-10s %-18s %-18s %12s %12s \n",
        result.layer,
        result.operation,
        result.input_shape,
        result.output_shape,
        format_time_nanoseconds(result.minimum_time_ns),
        format_time_nanoseconds(result.median_time_ns),
    )
end