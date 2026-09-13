push!(LOAD_PATH, "src")

using AWID
using BenchmarkTools
using Printf

const BATCH_SIZE = 10
const BENCHMARK_SAMPLES = 100


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


function benchmark_layer_allocation(
    layer_name::String,
    blueprint,
    input_shape::Tuple;
    batch_size::Int=BATCH_SIZE,
)
    benchmark = @benchmarkable AWID.build_layer(
        $blueprint,
        memory_pool,
        $input_shape,
        $batch_size,
        input_node,
    ) setup=(
        memory_pool = AWID.MemoryPool();

        input_node = AWID.alloc_act!(
            memory_pool,
            $input_shape...,
            $batch_size,
        )
    ) evals=1

    trial = BenchmarkTools.run(
        benchmark;
        samples=BENCHMARK_SAMPLES,
    )

    result = median(trial)

    return (
        layer=layer_name,
        memory=result.memory,
        allocations=result.allocs,
    )
end


results = [
    benchmark_layer_allocation(
        "Conv1 3x3 1→6",
        AWID.Conv((3, 3), 1 => 6; padding=1, bias=false),
        (28, 28, 1),
    ),

    benchmark_layer_allocation(
        "MaxPool1 2x2",
        AWID.MaxPool((2, 2)),
        (28, 28, 6),
    ),

    benchmark_layer_allocation(
        "Conv2 3x3 6→16",
        AWID.Conv((3, 3), 6 => 16; padding=1, bias=false),
        (14, 14, 6),
    ),

    benchmark_layer_allocation(
        "MaxPool2 2x2",
        AWID.MaxPool((2, 2)),
        (14, 14, 16),
    ),

    benchmark_layer_allocation(
        "Flatten 7x7x16",
        AWID.Flatten(),
        (7, 7, 16),
    ),

    benchmark_layer_allocation(
        "Dense 784→84",
        AWID.Dense(7 * 7 * 16 => 84),
        (7 * 7 * 16,),
    ),

    benchmark_layer_allocation(
        "ReLU 84",
        AWID.relu,
        (84,),
    ),

    benchmark_layer_allocation(
        "Dropout 84 p=0.4",
        AWID.Dropout(0.4f0),
        (84,),
    ),

    benchmark_layer_allocation(
        "Dense 84→10",
        AWID.Dense(84 => 10),
        (84,),
    ),
]


println()
println("==============================================================")
println("ALOKACJE PODCZAS BUDOWANIA WARSTW")
println("==============================================================")

@printf(
    "%-24s %16s %14s\n",
    "warstwa",
    "pamiec",
    "alokacje",
)

println(repeat("-", 58))

for result in results
    @printf(
        "%-24s %16s %14d\n",
        result.layer,
        format_bytes(result.memory),
        result.allocations,
    )
end
