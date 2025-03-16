struct SteadyStateGradientOptimizer <: Optimization.AbstractOptimizer
    ss_solver # Steady state solver
    maxiters::Int
end

# Default constructor
function SteadyStateGradientOptimizer(;ss_solver = DynamicSS(Rodas5(autodiff=false)), 
                                      maxiters::Int = Int(1e6))
    return SteadyStateGradientOptimizer(ss_solver, maxiters)
end

function Optimization.solve(prob::OptimizationProblem, 
                            opt::SteadyStateGradientOptimizer; 
                            kwargs...)
    u0, p = prob.u0, prob.p
    opt_cache = Optimization.OptimizationCache(prob, Optimization.LBFGS())
    optf = opt_cache.f
    
    function gradient_flow!(du, u, p, t)
        optf.grad(du, u, p)
        du .= -du
    end
    
    ode_prob = SteadyStateProblem(gradient_flow!, u0, p)
    sol = solve(ode_prob, opt.ss_solver; maxiters=opt.maxiters, kwargs...)
    
    # Calculate final function value for the solution
    f_val = prob.f(sol.u, p)
    
    return Optimization.build_solution(prob, 
                                       opt, 
                                       sol.u,
                                       f_val; 
                                       original=sol)
end
