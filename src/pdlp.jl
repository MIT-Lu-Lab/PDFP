# Restarted Halpern PDHG with reflection for linear programs, following
# cuPDLPx (Lu, Peng, Yang 2025). It solves the pump's root LP relaxation
#
#   min c'x   s.t.  con_lb ≤ A x ≤ con_ub,   var_lb ≤ x ≤ var_ub.
#
# Every routine is written with broadcasts and `mul!`, so the same code runs on
# `Vector` (CPU) and `CuVector` (GPU); the fused GPU kernels in
# pdlp_kernels.jl replace the two step functions on the GPU.

Base.@kwdef struct RestartParameters
    artificial_restart_threshold::Float64 = 0.36
    sufficient_reduction_for_restart::Float64 = 0.2
    necessary_reduction_for_restart::Float64 = 0.5
    k_p::Float64 = 0.99
    k_i::Float64 = 0.01
    k_d::Float64 = 0.0
    i_smooth::Float64 = 0.3
end

"Parameters shared by the PDLP and PDFP solvers; the defaults are cuPDLPx's."
Base.@kwdef struct SolverParameters
    eps_optimal_relative::Float64 = 1e-4
    eps_feasible_relative::Float64 = 1e-4
    iteration_limit::Int = 100_000
    time_limit::Float64 = 3600.0
    termination_evaluation_frequency::Int = 200
    reflection_coefficient::Float64 = 1.0
    sv_max_iterations::Int = 5000
    sv_tolerance::Float64 = 1e-4
    restart::RestartParameters = RestartParameters()
end

Base.@kwdef mutable struct PDLPState{V<:AbstractVector{Float64},M}
    A::M; A_t::M
    c::V
    var_lb::V; var_ub::V
    con_lb::V; con_ub::V
    n::Int; m::Int
    initial_primal::V; current_primal::V; pdhg_primal::V; reflected_primal::V
    initial_dual::V;   current_dual::V;   pdhg_dual::V;   reflected_dual::V
    primal_product::V; dual_product::V           # A x and A'y
    tmp_n::V; tmp_m::V
    dual_slack::V                                # bound multiplier from the primal projection
    delta_primal::V; delta_dual::V
    # bound/objective rescaling factors of the scaled problem
    bound_scale::Float64 = 1.0
    objective_scale::Float64 = 1.0
    # cuPDLPx termination semantics: residuals in original units and
    # infeasibility detection (see `cupdlpx_termination!`)
    cupdlpx_termination::Bool = false
    row_unscale::Any = nothing                   # 1 ./ D_r
    col_unscale::Any = nothing                   # 1 ./ D_c
    eps_infeasible::Float64 = 1e-14
    constraint_bound_norm::Float64 = 0.0
    objective_vector_norm::Float64 = 0.0
    step_size::Float64 = 0.0
    primal_weight::Float64 = 1.0
    best_primal_weight::Float64 = 1.0
    best_primal_dual_residual_gap::Float64 = Inf
    primal_weight_error_sum::Float64 = 0.0
    primal_weight_last_error::Float64 = 0.0
    fixed_point_error::Float64 = Inf
    initial_fixed_point_error::Float64 = Inf
    last_trial_fixed_point_error::Float64 = Inf
    inner_count::Int = 0                         # iterations since the last restart
    total_count::Int = 0
    relative_primal_residual::Float64 = Inf
    relative_dual_residual::Float64 = Inf
    relative_optimality_gap::Float64 = Inf
    objective_value::Float64 = NaN
    max_primal_ray_infeasibility::Float64 = NaN
    primal_ray_linear_objective::Float64 = NaN
    max_dual_ray_infeasibility::Float64 = NaN
    dual_ray_objective::Float64 = NaN
    termination_reason::Symbol = :UNSPECIFIED
end

_device_vector(v::Vector{Float64}, use_gpu::Bool) = use_gpu ? CuVector{Float64}(v) : copy(v)
_device_matrix(A::SparseMatrixCSC{Float64,Int}, use_gpu::Bool) =
    use_gpu ? CuSparseMatrixCSR(SparseMatrixCSC{Float64,Int32}(A)) : A

function PDLPState(sp::ScaledProblem; use_gpu::Bool)
    m, n = size(sp.A)
    dev(v) = _device_vector(v, use_gpu)
    zn() = dev(zeros(n)); zm() = dev(zeros(m))
    A = _device_matrix(sp.A, use_gpu)
    A_t = _device_matrix(sparse(transpose(sp.A)), use_gpu)
    return PDLPState(; A, A_t, c = dev(sp.c),
        var_lb = dev(sp.var_lb), var_ub = dev(sp.var_ub),
        con_lb = dev(sp.con_lb), con_ub = dev(sp.con_ub), n, m,
        initial_primal = zn(), current_primal = zn(), pdhg_primal = zn(), reflected_primal = zn(),
        initial_dual = zm(), current_dual = zm(), pdhg_dual = zm(), reflected_dual = zm(),
        primal_product = zm(), dual_product = zn(), tmp_n = zn(), tmp_m = zm(),
        dual_slack = zn(), delta_primal = zn(), delta_dual = zm(),
        bound_scale = sp.bound_scale, objective_scale = sp.objective_scale)
end

"""
    cupdlpx_termination!(s, D_r, D_c)

Measure the residuals as cuPDLPx does, in original units against the norms of
the original vectors, and detect primal and dual infeasibility from the
fixed-point residual direction.
"""
function cupdlpx_termination!(s::PDLPState, D_r::Vector{Float64}, D_c::Vector{Float64})
    s.row_unscale = copyto!(similar(s.tmp_m), 1.0 ./ D_r)
    s.col_unscale = copyto!(similar(s.tmp_n), 1.0 ./ D_c)
    s.cupdlpx_termination = true
    return s
end

# --- PDHG step ----------------------------------------------------------------

# x⁺ = Π_X(x - τ (c - A'y)); the dual slack is recorded on major steps only.
function primal_step!(s::PDLPState, k_offset::Int, ω::Float64, is_major::Bool)
    step = s.step_size / s.primal_weight
    k = s.inner_count + k_offset
    weight = k / (k + 1.0); omw = 1.0 - ω; omwt = 1.0 - weight
    if is_major
        @. s.dual_slack = s.current_primal - step * (s.c - s.dual_product)      # unprojected
        @. s.tmp_n = clamp(s.dual_slack, s.var_lb, s.var_ub)
        copyto!(s.pdhg_primal, s.tmp_n)
        @. s.dual_slack = (s.tmp_n - s.dual_slack) / step
    else
        @. s.tmp_n = clamp(s.current_primal - step * (s.c - s.dual_product), s.var_lb, s.var_ub)
        copyto!(s.pdhg_primal, s.tmp_n)
    end
    @. s.reflected_primal = 2 * s.tmp_n - s.current_primal
    @. s.current_primal = weight * (ω * s.reflected_primal + omw * s.current_primal) +
                          omwt * s.initial_primal
    return nothing
end

function dual_step!(s::PDLPState, k_offset::Int, ω::Float64, is_major::Bool)
    step = s.step_size * s.primal_weight
    k = s.inner_count + k_offset
    weight = k / (k + 1.0); omw = 1.0 - ω; omwt = 1.0 - weight
    @. s.tmp_m = ((s.current_dual / step - s.primal_product) -
                  clamp(s.current_dual / step - s.primal_product, -s.con_ub, -s.con_lb)) * step
    is_major && copyto!(s.pdhg_dual, s.tmp_m)
    @. s.reflected_dual = 2 * s.tmp_m - s.current_dual
    @. s.current_dual = weight * (ω * s.reflected_dual + omw * s.current_dual) +
                        omwt * s.initial_dual
    return nothing
end

function pdhg_step!(s::PDLPState, k_offset::Int, ω::Float64, is_major::Bool)
    mul!(s.dual_product, s.A_t, s.current_dual)
    primal_step!(s, k_offset, ω, is_major)
    mul!(s.primal_product, s.A, s.reflected_primal)
    dual_step!(s, k_offset, ω, is_major)
    return nothing
end

# --- restarts -------------------------------------------------------------------

function compute_fixed_point_error!(s::PDLPState)
    @. s.delta_primal = s.pdhg_primal - s.reflected_primal
    @. s.delta_dual = s.pdhg_dual - s.reflected_dual
    movement = sum(abs2, s.delta_primal) * s.primal_weight + sum(abs2, s.delta_dual) / s.primal_weight
    mul!(s.dual_product, s.A_t, s.delta_dual)
    interaction = 2 * s.step_size * dot(s.dual_product, s.delta_primal)
    s.fixed_point_error = sqrt(max(movement + interaction, 0.0))
    return s.fixed_point_error
end

function should_restart(s, p::SolverParameters)
    rp = p.restart
    restart = false
    if s.total_count == p.termination_evaluation_frequency
        restart = true
    elseif s.total_count > p.termination_evaluation_frequency
        s.fixed_point_error <= rp.sufficient_reduction_for_restart * s.initial_fixed_point_error &&
            (restart = true)
        s.fixed_point_error <= rp.necessary_reduction_for_restart * s.initial_fixed_point_error &&
            s.fixed_point_error > s.last_trial_fixed_point_error && (restart = true)
        s.inner_count >= rp.artificial_restart_threshold * s.total_count && (restart = true)
    end
    s.last_trial_fixed_point_error = s.fixed_point_error
    return restart
end

# PID update of the primal weight, shared by both solvers. `primal_dist` and
# `dual_dist` are the distances moved since the last restart.
function update_primal_weight!(s, rp::RestartParameters, primal_dist::Float64, dual_dist::Float64)
    ratio = s.relative_dual_residual / max(s.relative_primal_residual, eps())
    if 1e-16 < primal_dist < 1e12 && 1e-16 < dual_dist < 1e12 && 1e-8 < ratio < 1e8
        err = log(dual_dist) - log(primal_dist) - log(s.primal_weight)
        s.primal_weight_error_sum *= rp.i_smooth
        s.primal_weight_error_sum += err
        delta_err = err - s.primal_weight_last_error
        s.primal_weight *= exp(rp.k_p * err + rp.k_i * s.primal_weight_error_sum + rp.k_d * delta_err)
        s.primal_weight_last_error = err
    else
        s.primal_weight = s.best_primal_weight
        s.primal_weight_error_sum = 0.0
        s.primal_weight_last_error = 0.0
    end
    gap = abs(log10(s.relative_dual_residual / max(s.relative_primal_residual, eps())))
    if gap < s.best_primal_dual_residual_gap
        s.best_primal_dual_residual_gap = gap
        s.best_primal_weight = s.primal_weight
    end
    return nothing
end

function perform_restart!(s::PDLPState, p::SolverParameters)
    @. s.delta_primal = s.initial_primal - s.pdhg_primal
    @. s.delta_dual = s.initial_dual - s.pdhg_dual
    update_primal_weight!(s, p.restart, norm(s.delta_primal), norm(s.delta_dual))
    copyto!(s.initial_primal, s.pdhg_primal); copyto!(s.current_primal, s.pdhg_primal)
    copyto!(s.initial_dual, s.pdhg_dual);     copyto!(s.current_dual, s.pdhg_dual)
    s.inner_count = 0
    s.last_trial_fixed_point_error = Inf
    return nothing
end

# --- termination ------------------------------------------------------------------

function compute_residual!(s::PDLPState)
    mul!(s.primal_product, s.A, s.pdhg_primal)
    @. s.tmp_m = s.primal_product - clamp(s.primal_product, s.con_lb, s.con_ub)
    s.cupdlpx_termination && (s.tmp_m .*= s.row_unscale)
    primal_residual = norm(s.tmp_m) / s.bound_scale

    mul!(s.dual_product, s.A_t, s.pdhg_dual)
    @. s.tmp_n = s.c - s.dual_product - s.dual_slack
    s.cupdlpx_termination && (s.tmp_n .*= s.col_unscale)
    dual_residual = norm(s.tmp_n) / s.objective_scale

    s.relative_primal_residual = primal_residual / (1.0 + s.constraint_bound_norm)
    s.relative_dual_residual = dual_residual / (1.0 + s.objective_vector_norm)

    scale = s.bound_scale * s.objective_scale
    s.objective_value = dot(s.c, s.pdhg_primal) / scale
    @. s.tmp_m = max(s.pdhg_dual, 0.0) * ifelse(abs(s.con_lb) < 1e19, s.con_lb, 0.0) +
                 min(s.pdhg_dual, 0.0) * ifelse(abs(s.con_ub) < 1e19, s.con_ub, 0.0)
    dual_objective = (dot(s.dual_slack, s.pdhg_primal) + sum(s.tmp_m)) / scale
    s.relative_optimality_gap = abs(s.objective_value - dual_objective) /
                                (1.0 + abs(s.objective_value) + abs(dual_objective))
    return nothing
end

# Approximate primal and dual rays from the fixed-point residual direction
# (cuPDLPx utils.cu). Runs right after compute_fixed_point_error!, whose deltas
# hold pdhg - reflected; cuPDLPx's ray is reflected - pdhg.
function compute_infeasibility_information!(s::PDLPState)
    d = s.delta_primal; y = s.delta_dual
    @. d = -d; @. y = -y
    @. d = ifelse(isfinite(s.var_lb), max(d, 0.0), d)
    @. d = ifelse(isfinite(s.var_ub), min(d, 0.0), d)
    @. y = ifelse(isfinite(s.con_lb), y, min(y, 0.0))
    @. y = ifelse(isfinite(s.con_ub), y, max(y, 0.0))
    primal_ray_norm = s.n == 0 ? 0.0 : maximum(abs, d)
    primal_ray_norm > 0.0 && (d ./= primal_ray_norm)
    dual_ray_norm = s.m == 0 ? 0.0 : maximum(abs, y)
    mul!(s.primal_product, s.A, d)
    mul!(s.dual_product, s.A_t, y)
    scale = s.bound_scale * s.objective_scale
    s.primal_ray_linear_objective = dot(s.c, d) / scale
    @. s.tmp_m = max(y, 0.0) * ifelse(isfinite(s.con_lb), s.con_lb, 0.0) +
                 min(y, 0.0) * ifelse(isfinite(s.con_ub), s.con_ub, 0.0)
    primal_slack = sum(s.tmp_m)
    @. s.tmp_n = max(-s.dual_product, 0.0) * ifelse(isfinite(s.var_lb), s.var_lb, 0.0) +
                 min(-s.dual_product, 0.0) * ifelse(isfinite(s.var_ub), s.var_ub, 0.0)
    dual_slack = sum(s.tmp_n)
    s.dual_ray_objective = (primal_slack + dual_slack) / scale
    @. s.tmp_m = max(0.0, -s.primal_product) * isfinite(s.con_lb) +
                 max(0.0, s.primal_product) * isfinite(s.con_ub)
    s.tmp_m .*= s.row_unscale
    @. s.tmp_n = max(0.0, -s.dual_product) * !isfinite(s.var_lb) -
                 min(0.0, -s.dual_product) * !isfinite(s.var_ub)
    s.tmp_n .*= s.col_unscale
    s.max_primal_ray_infeasibility = s.m == 0 ? 0.0 : maximum(abs, s.tmp_m)
    dual_scratch_norm = s.n == 0 ? 0.0 : maximum(abs, s.tmp_n)
    scaling = max(dual_ray_norm, dual_scratch_norm)
    if scaling > 0.0
        s.max_dual_ray_infeasibility = dual_scratch_norm / scaling
        s.dual_ray_objective /= scaling
    else
        s.max_dual_ray_infeasibility = 0.0
        s.dual_ray_objective = 0.0
    end
    return nothing
end

_primal_infeasible(s::PDLPState) = s.dual_ray_objective > 0.0 &&
    s.max_dual_ray_infeasibility / s.dual_ray_objective <= s.eps_infeasible
_dual_infeasible(s::PDLPState) = s.primal_ray_linear_objective < 0.0 &&
    s.max_primal_ray_infeasibility / (-s.primal_ray_linear_objective) <= s.eps_infeasible

function initialize_step_size_and_primal_weight!(s::PDLPState, p::SolverParameters)
    if nnz(s.A) == 0
        s.step_size = 1.0
    else
        sv = estimate_max_singular_value(s.A, s.A_t, s.m, s.tmp_m, s.tmp_n;
                                         max_iterations = p.sv_max_iterations,
                                         tolerance = p.sv_tolerance)
        s.step_size = isfinite(sv) && sv > 0 ? 0.998 / sv : 0.998
    end
    if s.cupdlpx_termination
        # norms of the original (unscaled) vectors
        s.objective_vector_norm = norm(s.c .* s.col_unscale) / s.objective_scale
        lo = s.con_lb .* s.row_unscale ./ s.bound_scale
        hi = s.con_ub .* s.row_unscale ./ s.bound_scale
    else
        s.objective_vector_norm = norm(s.c)
        lo = s.con_lb; hi = s.con_ub
    end
    lo_sq = sum(abs2, ifelse.(isfinite.(lo) .& (s.con_lb .!= s.con_ub), lo, 0.0))
    hi_sq = sum(abs2, ifelse.(isfinite.(hi), hi, 0.0))
    s.constraint_bound_norm = sqrt(lo_sq + hi_sq)
    s.primal_weight = 1.0
    s.best_primal_weight = 1.0
    return nothing
end

"""
    pdlp_solve!(s, p) -> Symbol

Run restarted Halpern PDHG from the state's current iterate until the
optimality test of cuPDLPx holds, infeasibility is detected (under cuPDLPx
termination), or a limit is reached. Returns the termination reason.
"""
function pdlp_solve!(s::PDLPState, p::SolverParameters)
    t_start = time()
    initialize_step_size_and_primal_weight!(s, p)
    ω = p.reflection_coefficient
    freq = p.termination_evaluation_frequency
    do_restart = false
    while s.total_count < p.iteration_limit
        pdhg_step!(s, 1, ω, true)
        if do_restart
            compute_fixed_point_error!(s)
            s.initial_fixed_point_error = s.fixed_point_error
            do_restart = false
        end
        for i = 2:freq-1
            pdhg_step!(s, i, ω, false)
        end
        pdhg_step!(s, freq, ω, true)

        compute_fixed_point_error!(s)
        compute_residual!(s)
        s.cupdlpx_termination && compute_infeasibility_information!(s)
        s.inner_count += freq
        s.total_count += freq

        if s.relative_primal_residual <= p.eps_feasible_relative &&
           s.relative_dual_residual <= p.eps_feasible_relative &&
           s.relative_optimality_gap <= p.eps_optimal_relative
            s.termination_reason = :OPTIMAL
            break
        end
        if s.cupdlpx_termination && _primal_infeasible(s)
            s.termination_reason = :PRIMAL_INFEASIBLE
            break
        end
        if s.cupdlpx_termination && _dual_infeasible(s)
            s.termination_reason = :DUAL_INFEASIBLE
            break
        end
        if time() - t_start > p.time_limit
            s.termination_reason = :TIME_LIMIT
            break
        end
        do_restart = should_restart(s, p)
        do_restart && perform_restart!(s, p)
    end
    s.termination_reason === :UNSPECIFIED && (s.termination_reason = :ITERATION_LIMIT)
    return s.termination_reason
end

"""
    solve_lp_relaxation(milp; tolerance, time_limit, use_gpu, cupdlpx_termination)

Solve the LP relaxation of `milp` with PDLP. Returns the primal solution in
original units, the bound multipliers `dual_slack` (in scaled units; only their
signs are used), the termination reason and the iteration count.
"""
function solve_lp_relaxation(milp::MILP; tolerance::Float64, time_limit::Float64,
                             use_gpu::Bool, cupdlpx_termination::Bool)
    m = num_constraints(milp)
    con_lb = [milp.row_sense[i] === :le ? -Inf : milp.b[i] for i = 1:m]
    sp = scale_problem(milp.A, copy(milp.c), copy(milp.l), copy(milp.u), con_lb, copy(milp.b))
    s = PDLPState(sp; use_gpu)
    cupdlpx_termination && cupdlpx_termination!(s, sp.D_r, sp.D_c)
    p = SolverParameters(; iteration_limit = 5_000_000, eps_optimal_relative = tolerance,
                         eps_feasible_relative = tolerance, time_limit)
    status = pdlp_solve!(s, p)
    x = (sp.D_c .* Array(s.pdhg_primal)) ./ sp.bound_scale
    return x, Array(s.dual_slack), status, s.total_count
end
