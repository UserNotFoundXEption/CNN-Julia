import Base: +, *, -

mutable struct Node{T}
    value::T
    grad::T
    parents::Vector{Node}
    backward_fn::Union{Nothing, Function}
end

Node(val::T) where {T} = Node{T}(val, makegrad(val), Node[], nothing)

makegrad(val::Number) = zero(val)
makegrad(val::AbstractArray{T}) where {T} = zeros(T, size(val))

function backward!(node::Node)
    if node.grad isa Number
        node.grad = one(node.grad)
    else
        fill!(node.grad, one(eltype(node.grad)))
    end

    visited = Set{Node}()
    order = Node[]

    function topo(n::Node)
        if !(n in visited)
            push!(visited, n)
            for p in n.parents
                topo(p)
            end
            push!(order, n)
        end
    end

    topo(node)

    for n in reverse(order)
        if n.backward_fn !== nothing
            n.backward_fn()
        end
    end
end

function zero_grad!(params::Vector{<:Node})
    for p in params
        fill!(p.grad, 0f0)
    end
end

function +(a::Node, b::Node)
    out = Node(a.value .+ b.value)
    out.parents = [a, b]

    out.backward_fn = () -> begin
        a.grad .+= out.grad
        b.grad .+= out.grad
    end

    return out
end

function *(a::Node, b::Node)
    out_val = a.value * b.value
    out = Node(out_val)
    out.parents = [a, b]

    out.backward_fn = () -> begin
        if ndims(a.value) == 2 && ndims(b.value) == 1
            a.grad .+= out.grad * b.value'
            b.grad .+= a.value' * out.grad
        elseif ndims(a.value) == 2 && ndims(b.value) == 2
            a.grad .+= out.grad * b.value'
            b.grad .+= a.value' * out.grad
        else
            error("Unsupported shapes in Node * Node backward: $(size(a.value)) * $(size(b.value))")
        end
    end

    return out
end

function relu(x::Node{<:AbstractArray{Float32}})
    y = max.(0f0, x.value)
    out = Node(y)
    out.parents = [x]

    out.backward_fn = () -> begin
        @inbounds for i in eachindex(x.value)
            x.grad[i] += x.value[i] > 0f0 ? out.grad[i] : 0f0
        end
    end

    return out
end

function flatten(x::Node{Array{Float32,4}})
    h, w, c, b = size(x.value)
    out = Node(reshape(x.value, h * w * c, b))
    out.parents = [x]
    out.backward_fn = () -> begin
        x.grad .+= reshape(out.grad, h, w, c, b)
    end

    return out
end