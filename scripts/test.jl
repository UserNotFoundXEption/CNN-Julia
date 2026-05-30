push!(LOAD_PATH, "src")

if !isdefined(Main, :Cafe)
    using Cafe
end

pool = Cafe.MaxPool2D(2, 2)

x = rand(Float32, 28, 28, 8, 4)
xn = Cafe.Node(x)

y = Cafe.forward(pool, xn)

println(size(y.value))    

fill!(y.grad, 1f0)
y.backward_fn()

println(size(xn.grad)) 
println(sum(xn.grad))   