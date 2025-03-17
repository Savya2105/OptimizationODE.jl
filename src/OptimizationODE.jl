module OptimizationODE

using Reexport
@reexport using Optimization
using Optimization.SciMLBase

export ODESteadyStateOptimizer

struct ODESteadyStateOptimizer <: Optimization.AbstractOptimizer
    ss_solver
    maxiters::Int  
end

ODESteadyStateOptimizer(; 
    ss_solver = DynamicSS(Rodas5(autodiff=false)), 
    maxiters = Int(1e6)
) = ODESteadyStateOptimizer(ss_solver, maxiters)

function SciMLBase.solve(prob::Optimization.OptimizationProblem, opt::ODESteadyStateOptimizer; 
                         callback = nothing, kwargs...)
    u0, p = prob.u0, prob.p
    
    opt_cache = Optimization.OptimizationCache(prob)
    optf = opt_cache.f
    
    function gradient_flow!(du, u, p, t)
        optf.grad(du, u, p)
        du .= -du
    end
    
    ss_prob = SteadyStateProblem(gradient_flow!, u0, p)
    
    sol = solve(ss_prob, opt.ss_solver; 
                maxiters = opt.maxiters, 
                kwargs...)
    
    u_final = sol.u
    
    obj_final = optf.f(u_final, p)
    
    return SciMLBase.OptimizationSolution(
        u_final,            
        opt_cache,            
        opt,                
        obj_final,  
        sol.retcode,          
        sol,
        nothing                
    )
end
end
