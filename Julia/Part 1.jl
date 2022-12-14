#=
Creates a basic neuron with 3 inputs.
=#

inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3.0

output = inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
println(output)
