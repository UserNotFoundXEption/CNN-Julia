module AWID

using LinearAlgebra
using Random

include("memory_pool.jl")
include("init.jl")
include("blueprints.jl")
include("static_chain.jl")

include("layer_dense.jl")
include("layer_relu_flatten.jl")
include("layer_dropout.jl")
include("layer_maxpool.jl")
include("layer_conv.jl")

include("loss.jl")
include("model.jl")

export MemoryPool,
       GraphNode,
       alloc_weight!,
       alloc_act!,
       zero_w_grad!,
       zero_a_grad!,
       zero_grad!,
       optimize!,

       Operator,
       Blueprint,
       Chain,
       ChainDef,
       StaticChain,
       CompiledModel,
       build_model,

       Dense,
       Conv,
       MaxPool,
       Dropout,
       Flatten,
       relu,
       flatten,

       LogitCrossEntropy,
       onehot!,
       loss!,
       model_output,

       forward!,
       forward_train!,
       forward_test!,
       backward!

end
