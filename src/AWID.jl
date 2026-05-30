module AWID

include("node.jl")
include("layer_conv.jl")
include("layer_dense.jl")
include("layer_dropout.jl")
include("layer_maxpool.jl")
include("utils.jl")

export Node, backward!, zero_grad!, relu, flatten, forward, pad_input, Dense, Conv, MaxPool, Dropout, sgd!, GradientDescent, accumulate!, optimize!, softmax, cross_entropy

end