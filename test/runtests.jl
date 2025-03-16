using OptimizationSteadyStateGradient
using Test

using OptimizationSteadyStateGradient
using Test
using DifferentialEquations

@testset "OptimizationSteadyStateGradient.jl" begin
    # Test with Rosenbrock function
    function rosenbrock(u, p)
        (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
    end
    
    u0 = zeros(2)
    p = [1.0, 100.0]
    optf = OptimizationFunction(rosenbrock, AutoForwardDiff())
    prob = OptimizationProblem(optf, u0, p)
    
    # Solve with SteadyStateGradientOptimizer
    sol = solve(prob, SteadyStateGradientOptimizer())
    
    # Check that solution is close to the expected minimum [1.0, 1.0]
    @test isapprox(sol[1], 1.0, atol=1e-3)
    @test isapprox(sol[2], 1.0, atol=1e-3)
end
