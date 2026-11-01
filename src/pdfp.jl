# PDFP: restarted Halpern PDHG with reflection on the pump's projection problem
#
#   min_x  Σ_{j∈I} w_j |x_j - x̃_j|   s.t.  A x (≤, =) b,   ℓ ≤ x ≤ u,
#
# solved directly through its saddle-point form
#
#   min_{ℓ≤x≤u} max_{y,z}  y'(x̃ - x) + z'(A x - b),
#   |y_j| ≤ w_j (j ∈ I),  y_j = 0 (j ∉ I),  z_i ≥ 0 on ≤ rows,
#
# so no auxiliary variables or constraints are needed, also for general
# integers. The data are preconditioned (x = D_c .* x̂); the weights w_j = D_c[j]
# keep the distance in original units. The coupling operator is K = [-I_I; A],
# hence the step size 0.998 / √(1 + ‖A‖²).

Base.@kwdef mutable struct PDFPState{V<:AbstractVector{Float64},M,B<:AbstractVector{Bool}}
    A::M; A_t::M
    b::V
    is_eq::B
    is_int::B
    var_lb::V; var_ub::V
    x_tilde::V                       # target, in scaled units
    w::V                             # half-width of the y box
    n::Int; m::Int
    initial_x::V; current_x::V; pdhg_x::V; reflected_x::V
    initial_y::V; current_y::V; pdhg_y::V; reflected_y::V
    initial_z::V; current_z::V; pdhg_z::V; reflected_z::V
    Ax::V; Atz::V; tmp_n::V; tmp_m::V
    dual_slack::V
    delta_x::V; delta_y::V; delta_z::V
    # primal residual in original units (see `original_primal_residual!`)
    original_primal_residual::Bool = false
    row_unscale::Any = nothing
    b_norm_original::Float64 = NaN
    max_singular_value::Float64 = NaN      # ‖A‖₂, computed once
    step_size::Float64 = 0.0
    primal_weight::Float64 = 1.0
    best_primal_weight::Float64 = 1.0
    best_primal_dual_residual_gap::Float64 = Inf
    primal_weight_error_sum::Float64 = 0.0
    primal_weight_last_error::Float64 = 0.0
    fixed_point_error::Float64 = Inf
    initial_fixed_point_error::Float64 = Inf
    last_trial_fixed_point_error::Float64 = Inf
    inner_count::Int = 0
    total_count::Int = 0
    relative_primal_residual::Float64 = Inf
    relative_dual_residual::Float64 = Inf
    relative_l1_gap::Float64 = Inf
    dual_residual_denominator::Float64 = 1.0   # 1 + ‖w_I‖₂
    l1_gap_denominator::Float64 = 1.0          # 1 + ‖x̃_I‖₁ in original units
    termination_reason::Symbol = :UNSPECIFIED
end

"""
    PDFPState(A, b, is_eq, is_int, var_lb, var_ub, w; use_gpu)

Projector state for the scaled data. It is built once per pump run; every
projection only replaces the target and warm-starts from the last iterate.
"""
function PDFPState(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64}, is_eq::Vector{Bool},
                   is_int::Vector{Bool}, var_lb::Vector{Float64}, var_ub::Vector{Float64},
                   w::Vector{Float64}; use_gpu::Bool)
    m, n = size(A)
    dev(v) = _device_vector(v, use_gpu)
    devb(v) = use_gpu ? CuVector{Bool}(v) : copy(v)
    zn() = dev(zeros(n)); zm() = dev(zeros(m))
    return PDFPState(; A = _device_matrix(A, use_gpu),
        A_t = _device_matrix(sparse(transpose(A)), use_gpu),
        b = dev(b), is_eq = devb(is_eq), is_int = devb(is_int),
        var_lb = dev(var_lb), var_ub = dev(var_ub), x_tilde = zn(), w = dev(w), n, m,
        initial_x = zn(), current_x = zn(), pdhg_x = zn(), reflected_x = zn(),
        initial_y = zn(), current_y = zn(), pdhg_y = zn(), reflected_y = zn(),
        initial_z = zm(), current_z = zm(), pdhg_z = zm(), reflected_z = zm(),
        Ax = zm(), Atz = zn(), tmp_n = zn(), tmp_m = zm(), dual_slack = zn(),
        delta_x = zn(), delta_y = zn(), delta_z = zm())
end

"""
    original_primal_residual!(s, D_r)

Measure the primal residual in original units, relative to the original ‖b‖,
instead of on the scaled rows. Used on models with continuous variables,
whose feasibility depends on the accuracy of the projection itself.
"""
function original_primal_residual!(s::PDFPState, D_r::Vector{Float64})
    s.row_unscale = copyto!(similar(s.tmp_m), 1.0 ./ D_r)
    s.b_norm_original = norm(Array(s.b) ./ D_r)
    s.original_primal_residual = true
    return s
end

"""
    set_target!(s, x_tilde_scaled)

Start a new projection onto `x_tilde_scaled`, warm-started from the current
iterate.
"""
function set_target!(s::PDFPState, x_tilde_scaled::Vector{Float64})
    copyto!(s.x_tilde, x_tilde_scaled)
    return warm_start!(s)
end

"Anchor the Halpern iteration at the current iterate and reset the counters."
function warm_start!(s::PDFPState)
    copyto!(s.initial_x, s.current_x)
    copyto!(s.initial_y, s.current_y)
    copyto!(s.initial_z, s.current_z)
    s.inner_count = 0
    s.total_count = 0
    s.fixed_point_error = Inf
    s.last_trial_fixed_point_error = Inf
    s.termination_reason = :UNSPECIFIED
    return s
end

# --- PDHG step ----------------------------------------------------------------

# x⁺ = Π_[ℓ,u](x - τ (A'z - y))
function primal_step!(s::PDFPState, k_offset::Int, ω::Float64)
    τ = s.step_size / s.primal_weight
    k = s.inner_count + k_offset
    weight = k / (k + 1.0); omw = 1.0 - ω; omwt = 1.0 - weight
    mul!(s.Atz, s.A_t, s.current_z)
    @. s.dual_slack = s.current_x - τ * (s.Atz - s.current_y)        # unprojected
    @. s.tmp_n = clamp(s.dual_slack, s.var_lb, s.var_ub)
    copyto!(s.pdhg_x, s.tmp_n)
    @. s.dual_slack = (s.tmp_n - s.dual_slack) / τ
    @. s.reflected_x = 2 * s.tmp_n - s.current_x
    @. s.current_x = weight * (ω * s.reflected_x + omw * s.current_x) + omwt * s.initial_x
    return nothing
end

# y⁺ = Π_[-w,w](y + σ (x̃ - x̄)) on I, 0 elsewhere
function dual_y_step!(s::PDFPState, k_offset::Int, ω::Float64)
    σ = s.step_size * s.primal_weight
    k = s.inner_count + k_offset
    weight = k / (k + 1.0); omw = 1.0 - ω; omwt = 1.0 - weight
    @. s.tmp_n = ifelse(s.is_int, clamp(s.current_y + σ * (s.x_tilde - s.reflected_x), -s.w, s.w), 0.0)
    copyto!(s.pdhg_y, s.tmp_n)
    @. s.reflected_y = 2 * s.tmp_n - s.current_y
    @. s.current_y = weight * (ω * s.reflected_y + omw * s.current_y) + omwt * s.initial_y
    return nothing
end

# z⁺ = z + σ (A x̄ - b), projected onto z ≥ 0 on ≤ rows
function dual_z_step!(s::PDFPState, k_offset::Int, ω::Float64)
    σ = s.step_size * s.primal_weight
    k = s.inner_count + k_offset
    weight = k / (k + 1.0); omw = 1.0 - ω; omwt = 1.0 - weight
    mul!(s.Ax, s.A, s.reflected_x)
    @. s.tmp_m = ifelse(s.is_eq, s.current_z + σ * (s.Ax - s.b), max(s.current_z + σ * (s.Ax - s.b), 0.0))
    copyto!(s.pdhg_z, s.tmp_m)
    @. s.reflected_z = 2 * s.tmp_m - s.current_z
    @. s.current_z = weight * (ω * s.reflected_z + omw * s.current_z) + omwt * s.initial_z
    return nothing
end

function pdhg_step!(s::PDFPState, k_offset::Int, ω::Float64)
    primal_step!(s, k_offset, ω)
    dual_y_step!(s, k_offset, ω)
    dual_z_step!(s, k_offset, ω)
    return nothing
end

# --- restarts ---------------------------------------------------------------------

function compute_fixed_point_error!(s::PDFPState)
    @. s.delta_x = s.pdhg_x - s.reflected_x
    @. s.delta_y = s.pdhg_y - s.reflected_y
    @. s.delta_z = s.pdhg_z - s.reflected_z
    movement = sum(abs2, s.delta_x) * s.primal_weight +
               (sum(abs2, s.delta_y) + sum(abs2, s.delta_z)) / s.primal_weight
    mul!(s.Atz, s.A_t, s.delta_z)
    @. s.tmp_n = s.Atz - s.delta_y
    interaction = 2 * s.step_size * dot(s.tmp_n, s.delta_x)
    s.fixed_point_error = sqrt(max(movement + interaction, 0.0))
    return s.fixed_point_error
end

function perform_restart!(s::PDFPState, p::SolverParameters)
    @. s.delta_x = s.initial_x - s.pdhg_x
    @. s.delta_y = s.initial_y - s.pdhg_y
    @. s.delta_z = s.initial_z - s.pdhg_z
    update_primal_weight!(s, p.restart, norm(s.delta_x),
                          sqrt(sum(abs2, s.delta_y) + sum(abs2, s.delta_z)))
    copyto!(s.initial_x, s.pdhg_x); copyto!(s.current_x, s.pdhg_x)
    copyto!(s.initial_y, s.pdhg_y); copyto!(s.current_y, s.pdhg_y)
    copyto!(s.initial_z, s.pdhg_z); copyto!(s.current_z, s.pdhg_z)
    s.inner_count = 0
    s.last_trial_fixed_point_error = Inf
    return nothing
end

# --- termination ------------------------------------------------------------------

# Relative primal residual (constraint violation), relative dual residual
# (stationarity in x, normalized by 1 + ‖w_I‖₂, the largest norm of a
# subgradient of the weighted ℓ₁ term) and relative gap between the saddle
# value and the ℓ₁ distance.
function compute_residual!(s::PDFPState)
    mul!(s.Ax, s.A, s.pdhg_x)
    @. s.tmp_m = ifelse(s.is_eq, abs(s.Ax - s.b), max(s.Ax - s.b, 0.0))
    s.original_primal_residual && (s.tmp_m .*= s.row_unscale)
    b_norm = s.original_primal_residual ? s.b_norm_original : norm(s.b)
    s.relative_primal_residual = norm(s.tmp_m) / (1.0 + b_norm)

    mul!(s.Atz, s.A_t, s.pdhg_z)
    @. s.tmp_n = s.Atz - s.pdhg_y - s.dual_slack
    s.relative_dual_residual = norm(s.tmp_n) / s.dual_residual_denominator

    @. s.tmp_n = ifelse(s.is_int, s.w * abs(s.pdhg_x - s.x_tilde), 0.0)
    l1 = sum(s.tmp_n)
    @. s.tmp_n = ifelse(s.is_int, s.pdhg_y * (s.x_tilde - s.pdhg_x), 0.0)
    y_term = sum(s.tmp_n)
    z_term = dot(s.pdhg_z, s.Ax .- s.b)
    s.relative_l1_gap = abs(y_term + z_term - l1) / s.l1_gap_denominator
    return nothing
end

function initialize_step_size_and_primal_weight!(s::PDFPState, p::SolverParameters,
                                                  primal_weight::Float64)
    if nnz(s.A) == 0
        s.step_size = 1.0
    else
        if isnan(s.max_singular_value)
            s.max_singular_value = estimate_max_singular_value(s.A, s.A_t, s.m, s.tmp_m, s.tmp_n;
                                        max_iterations = p.sv_max_iterations,
                                        tolerance = p.sv_tolerance)
        end
        s.step_size = 0.998 / sqrt(1.0 + s.max_singular_value^2)
    end
    s.primal_weight = primal_weight
    s.best_primal_weight = primal_weight
    s.primal_weight_error_sum = 0.0
    s.primal_weight_last_error = 0.0
    @. s.tmp_n = ifelse(s.is_int, s.w, 0.0)
    s.dual_residual_denominator = 1.0 + norm(s.tmp_n)
    @. s.tmp_n = ifelse(s.is_int, s.w * abs(s.x_tilde), 0.0)
    s.l1_gap_denominator = 1.0 + sum(s.tmp_n)
    return nothing
end

"""
    pdfp_solve!(s, p, primal_weight) -> Symbol

Project the target held in `s` (see `set_target!`), warm-started from the
state's iterate. Stops when the relative primal residual is at most
`p.eps_feasible_relative` and both the relative dual residual and the relative
ℓ₁ gap are at most `p.eps_optimal_relative`, or at a limit. Returns the
termination reason; the projection is `s.pdhg_x` in scaled units.
"""
function pdfp_solve!(s::PDFPState, p::SolverParameters, primal_weight::Float64)
    t_start = time()
    initialize_step_size_and_primal_weight!(s, p, primal_weight)
    ω = p.reflection_coefficient
    freq = p.termination_evaluation_frequency
    do_restart = false
    while s.total_count < p.iteration_limit
        pdhg_step!(s, 1, ω)
        if do_restart
            compute_fixed_point_error!(s)
            s.initial_fixed_point_error = s.fixed_point_error
            do_restart = false
        end
        for i = 2:freq
            pdhg_step!(s, i, ω)
        end
        compute_fixed_point_error!(s)
        compute_residual!(s)
        s.inner_count += freq
        s.total_count += freq

        if s.relative_primal_residual <= p.eps_feasible_relative &&
           s.relative_dual_residual <= p.eps_optimal_relative &&
           s.relative_l1_gap <= p.eps_optimal_relative
            s.termination_reason = :OPTIMAL
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
