mutable struct Dropout
    p::Float32
end

Dropout(p::Real) = Dropout(Float32(p))

function forward(l::Dropout, x::Node; training::Bool=true)
    if !training || l.p <= 0f0
        out = Node(copy(x.value))
        out.parents = [x]
        out.backward_fn = () -> begin
            x.grad .+= out.grad
        end
        return out
    end

    keep_prob = 1f0 - l.p
    mask = Float32.(rand(Float32, size(x.value)...) .< keep_prob) ./ keep_prob

    out = Node(x.value .* mask)
    out.parents = [x]

    out.backward_fn = () -> begin
        x.grad .+= out.grad .* mask
    end

    return out
end