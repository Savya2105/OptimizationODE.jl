using OptimizationODE
using Test

@testset "OptimizationODE.jl" begin
    function rosenbrock(u, p)
        (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2
    end
    
    u0 = zeros(2)
    p = [1.0, 100.0]
    optf = OptimizationFunction(rosenbrock, AutoForwardDiff())
    prob = OptimizationProblem(optf, u0, p)
    
    sol = solve(prob, ODESteadyStateOptimizer())
    
    @test isapprox(sol[1], 1.0, atol=1e-3)
    @test isapprox(sol[2], 1.0, atol=1e-3)
end
