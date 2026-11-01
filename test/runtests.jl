using PDFP
using CUDA
using LinearAlgebra
using Random
using SparseArrays
using Test

const EXAMPLES = joinpath(@__DIR__, "..", "examples")
example(name) = read_mps(joinpath(EXAMPLES, name * ".mps.gz"))

# min Σ_{j∈I} |x_j - x̃_j| s.t. A x (≤, =) b, l ≤ x ≤ u, solved by PDFP.
function project(milp::MILP, x_tilde::Vector{Float64}; use_gpu = false, tol = 1e-8)
    n = length(milp.c); m = length(milp.b)
    is_eq = milp.row_sense .=== :eq
    is_int = falses(n); is_int[milp.int_idx] .= true
    con_lb = [is_eq[i] ? milp.b[i] : -Inf for i = 1:m]
    sp = PDFP.scale_problem(milp.A, zeros(n), copy(milp.l), copy(milp.u), con_lb, copy(milp.b);
                            bound_objective_rescaling = false)
    w = ones(n); w[milp.int_idx] .= sp.D_c[milp.int_idx]
    s = PDFP.PDFPState(sp.A, sp.con_ub, Vector{Bool}(is_eq), Vector{Bool}(is_int),
                       sp.var_lb, sp.var_ub, w; use_gpu)
    PDFP.set_target!(s, x_tilde ./ sp.D_c)
    p = PDFP.SolverParameters(; eps_optimal_relative = tol, eps_feasible_relative = tol,
                              iteration_limit = 1_000_000)
    status = PDFP.pdfp_solve!(s, p, 1.0)
    return sp.D_c .* Array(s.pdhg_x), status
end

@testset "PDFP.jl" begin
    @testset "MPS reader" begin
        m = example("p0201")
        @test length(m.c) == 201
        @test length(m.int_idx) == 201
        @test all(m.l[m.int_idx] .== 0) && all(m.u[m.int_idx] .== 1)
        @test all(s -> s in (:le, :eq), m.row_sense)
        m = example("pk1")
        @test PDFP.num_continuous(m) == 31
    end

    @testset "projection" begin
        # min |x1 - 1| + |x2 - 1| s.t. x1 + x2 ≤ 1, 0 ≤ x ≤ 1: distance 1
        m = MILP([0.0, 0.0], 0.0, sparse([1.0 1.0]), [1.0], [:le], [0.0, 0.0], [1.0, 1.0],
                 [1, 2], ["x1", "x2"], "toy")
        x, status = project(m, [1.0, 1.0])
        @test status === :OPTIMAL
        @test sum(abs, x .- 1.0) ≈ 1.0 atol = 1e-5
        @test sum(x) <= 1.0 + 1e-6

        # a general integer: min |x1 - 3| s.t. x1 + x2 = 2, 0 ≤ x1 ≤ 5, x2 ≥ 0 continuous
        m = MILP([0.0, 0.0], 0.0, sparse([1.0 1.0]), [2.0], [:eq], [0.0, 0.0], [5.0, Inf],
                 [1], ["x1", "x2"], "toy2")
        x, status = project(m, [3.0, 0.0])
        @test status === :OPTIMAL
        @test x[1] ≈ 2.0 atol = 1e-5
    end

    @testset "rounding" begin
        m = example("p0201")
        rng = MersenneTwister(1)
        # an LP-integral feasible point rounds to itself
        r = feasibility_pump(m; use_gpu = false, time_limit = 60.0)
        rounder = PDFP.Rounder(m)
        @test PDFP.round_guarded(rounder, r.x, rng, 0) == r.x
        # with no LP-integral coordinate the batch is empty and the guarded
        # rounder is sequential CPR
        x = fill(0.5, length(m.c)) .+ 0.01 .* rand(MersenneTwister(2), length(m.c))
        a = PDFP.round_guarded(PDFP.Rounder(m; policy = nothing), x, MersenneTwister(3), 20)
        b = PDFP.round_sequential(PDFP.Rounder(m; policy = nothing), x, MersenneTwister(3), 20)
        @test a == b
        @test all(v -> v == round(v), a)
    end

    @testset "pump on $name" for name in ["p0201", "gen-ip054", "pk1", "markshare2"]
        m = example(name)
        r = feasibility_pump(m; use_gpu = false, time_limit = 120.0)
        @test r.found
        @test is_feasible(m, r.x)
        @test r.objective ≈ dot(m.c, r.x) + m.c0
    end

    if CUDA.functional()
        @testset "GPU" begin
            m = example("pk1")
            x_tilde = round.(rand(MersenneTwister(4), length(m.c)))
            x_cpu, _ = project(m, x_tilde; use_gpu = false)
            x_gpu, _ = project(m, x_tilde; use_gpu = true)
            I = m.int_idx
            @test sum(abs, x_gpu[I] .- x_tilde[I]) ≈ sum(abs, x_cpu[I] .- x_tilde[I]) rtol = 1e-4
            r = feasibility_pump(m; use_gpu = true, time_limit = 120.0)
            @test r.found && is_feasible(m, r.x)
        end
    else
        @info "CUDA not functional: GPU tests skipped"
    end
end
