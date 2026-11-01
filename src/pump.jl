# The feasibility pump loop (Fischetti, Glover and Lodi 2005; FP 2.0 of
# Fischetti and Salvagnin 2009) with PDFP as the projector and guarded
# two-phase CPR as the rounder.

"""
    PumpResult

Outcome of `feasibility_pump`.

- `found`: whether `x` is a feasible solution.
- `x`: the feasible solution, or the last rounded point if none was found.
- `objective`: `c'x + c0`.
- `max_violation`: largest constraint violation at `x`.
- `iterations`: number of projections (pump iterations).
- `perturbations`: number of restarts after a cycle or every `restart_period`
  iterations.
- `pdhg_iterations`: PDHG iterations over all projections.
- `root_lp_status`, `root_lp_time`: termination reason and time of the root LP.
- `projection_time`, `rounding_time`: time spent in the two phases.
- `time`: total time.
"""
struct PumpResult
    found::Bool
    x::Vector{Float64}
    objective::Float64
    max_violation::Float64
    iterations::Int
    perturbations::Int
    pdhg_iterations::Int
    root_lp_status::Symbol
    root_lp_time::Float64
    projection_time::Float64
    rounding_time::Float64
    time::Float64
end

# Counters of a run. Kept in one mutable object rather than in local
# variables: a closure capturing a reassigned local makes Julia box it for the
# whole function, and loops over boxed vectors are slow.
Base.@kwdef mutable struct PumpCounters
    iterations::Int = 0
    perturbations::Int = 0
    pdhg_iterations::Int = 0
    root_lp_status::Symbol = :UNSPECIFIED
    root_lp_time::Float64 = 0.0
    projection_time::Float64 = 0.0
    rounding_time::Float64 = 0.0
end

_differs(a::Vector{Float64}, b::Vector{Float64}, idx::Vector{Int}) = any(j -> a[j] != b[j], idx)

# Nearest integer on I, kept inside the bounds.
function _round_to_nearest(x::Vector{Float64}, milp::MILP)
    x̃ = copy(x)
    @inbounds for j in milp.int_idx
        v = round(x[j])
        isfinite(milp.l[j]) && v < milp.l[j] && (v = ceil(milp.l[j]))
        isfinite(milp.u[j]) && v > milp.u[j] && (v = floor(milp.u[j]))
        x̃[j] = v
    end
    return x̃
end

# Hash of the integer coordinates, for cycle detection. -0.0 and 0.0 hash
# differently in Julia, so -0.0 is mapped to 0.0 first.
function _hash_integers(x::Vector{Float64}, int_idx::Vector{Int})
    h = 0x9e3779b97f4a7c15
    @inbounds for j in int_idx
        v = x[j]
        v == 0.0 && (v = 0.0)
        h = hash(v, h)
    end
    return h
end

# The batch schedule from the root LP. A batch fixes LP-integral variables at
# their LP values; where the LP is indifferent to them (few of them are
# pushed against their bound by the objective) those values can be jointly
# infeasible, and :regroup lets propagation from the displaced fractional
# variables reach them first.
function _batch_schedule(milp::MILP, x::Vector{Float64}, dual_slack::Vector{Float64},
                         push_threshold::Float64)
    n_pushed = 0; n_integral = 0
    @inbounds for j in milp.int_idx
        abs(x[j] - round(x[j])) == 0.0 || continue
        n_integral += 1
        at_lower = isfinite(milp.l[j]) && abs(x[j] - milp.l[j]) < 1e-9
        at_upper = isfinite(milp.u[j]) && abs(x[j] - milp.u[j]) < 1e-9
        ((at_lower && dual_slack[j] > 0) || (at_upper && dual_slack[j] < 0)) && (n_pushed += 1)
    end
    fraction = n_integral > 0 ? n_pushed / n_integral : 1.0
    return fraction < push_threshold ? :regroup : :all
end

# Total violation of the rows without continuous variables. On a mixed model
# the first rounding keeps the continuous variables at their root-LP values,
# so rows holding them can look violated where the integer part is feasible.
function _total_integer_row_violation(milp::MILP, x::Vector{Float64})
    m = num_constraints(milp)
    rows = rowvals(milp.A); vals = nonzeros(milp.A)
    activity = zeros(m)
    @inbounds for j in eachindex(x), p in nzrange(milp.A, j)
        activity[rows[p]] += vals[p] * x[j]
    end
    skip = falses(m)
    is_int = falses(length(x)); is_int[milp.int_idx] .= true
    @inbounds for j in eachindex(x)
        is_int[j] && continue
        for p in nzrange(milp.A, j)
            skip[rows[p]] = true
        end
    end
    total = 0.0
    @inbounds for i = 1:m
        skip[i] || (total += _row_violation(milp, i, activity[i]))
    end
    return total
end

# FP's stalling move: flip the TT integer variables farthest from x.
function _flip_farthest!(x̃::Vector{Float64}, x::Vector{Float64}, milp::MILP, TT::Int)
    l = milp.l; u = milp.u
    distances = [(abs(x[j] - x̃[j]), j) for j in milp.int_idx]
    sort!(distances; rev = true, by = first)
    @inbounds for k = 1:min(TT, length(distances))
        d, j = distances[k]
        d == 0.0 && continue
        cur = x̃[j]
        direction = x[j] >= cur ? 1.0 : -1.0
        v = cur + direction
        (v < l[j] || v > u[j]) && (v = cur - direction)
        (v < l[j] || v > u[j]) && continue
        x̃[j] = v
    end
end

# Restart strength for general integers (FeasPumpCollection): decays by 0.85
# per iteration, grows on every restart, capped at max(|G|/10, 10).
mutable struct RestartLadder
    flips::Float64
    last_restart::Int
    average_flips::Int
    max_flips::Int
end

# FP's restart. A binary flips toward x when |x_j - x̃_j| + max(ρ_j, 0) > 0.5,
# with ρ_j uniform on [-0.3, 0.7]; the other integer variables are redrawn
# uniformly over their domain, as many as the ladder says.
function _perturb!(x̃::Vector{Float64}, x::Vector{Float64}, milp::MILP, rng::AbstractRNG,
                   general::Vector{Int}, is_general::BitVector, ladder::RestartLadder,
                   iteration::Int)
    l = milp.l; u = milp.u
    if !isempty(general)
        while ladder.last_restart < iteration
            ladder.last_restart += 1
            ladder.flips *= 0.85
        end
        ladder.flips = min(ladder.flips + 2 * ladder.average_flips + 1, ladder.max_flips)
        @inbounds for _ = 1:max(1, floor(Int, ladder.flips))
            j = general[rand(rng, 1:length(general))]
            lo = l[j]; hi = u[j]; r = rand(rng)
            v = if isfinite(hi - lo) && hi - lo < 1e15
                floor(lo + (1.0 + hi - lo) * r)
            elseif isfinite(lo) && x̃[j] - lo < 1e9
                lo + (2e9 - 1.0) * r
            elseif isfinite(hi) && hi - x̃[j] < 1e9
                hi - (2e9 - 1.0) * r
            else
                x̃[j] + (2e9 - 1.0) * r - 1e9
            end
            v = round(clamp(v, lo, hi))
            abs(v - x̃[j]) > 1e-6 && (x̃[j] = v)
        end
    end
    @inbounds for j in milp.int_idx
        is_general[j] && continue
        ρ = -0.3 + rand(rng)
        if abs(x[j] - x̃[j]) + max(ρ, 0.0) > 0.5
            cur = x̃[j]
            direction = x[j] >= cur ? 1.0 : -1.0
            v = cur + direction
            (v < l[j] || v > u[j]) && (v = cur - direction)
            (v < l[j] || v > u[j]) && continue
            x̃[j] = v
        end
    end
end

"""
    feasibility_pump(milp; kwargs...) -> PumpResult

Search for a feasible solution of `milp` with the feasibility pump: alternate
rounding an LP-feasible point `x` to an integral point `x̃` and projecting `x̃`
back onto the LP relaxation in ℓ₁ distance, until the two meet.

The projection is computed by PDFP on the GPU (or the CPU), warm-started from
the previous projection, with the gap tolerance `gap_start · gap_decay^k`
floored at `gap_floor` and the residual tolerance `residual_ratio` times the
gap tolerance. The rounding is guarded two-phase CPR with violation-guided
rounding.

# Keyword arguments
- `time_limit = 1800.0`: in seconds, root LP included.
- `seed = 1`: seed of the random numbers (rounding swaps, flips, restarts).
- `use_gpu = CUDA.functional()`.
- `verbose = false`.
- `flip_parameter = 20`: FP's T; a stalling iteration flips a random number
  of integer variables between T/2 and 3T/2.
- `restart_period = 100`: FP's R; a restart is forced after this many
  iterations without one.
- `swap_iteration = 10`: from this iteration on, CPR swaps |I|/10 random
  pairs of its ranking.
- `gap_start = 1e-2`, `gap_decay = 0.98`, `gap_floor = 1e-4`: tolerance
  schedule of the projections (Scylla, Mexi et al. 2023).
- `residual_ratio = 0.1`: residual tolerance over gap tolerance.
- `root_lp_tolerance = 1e-2`: tolerance of the root LP relaxation.
- `projection_iteration_limit = 500_000`: PDHG iterations per projection.
- `batch_policy = :cost`: `:cost` learns per run whether trying the batch
  pays; `:always` tries it on every call.
- `push_threshold = 0.01`: below this fraction of LP-integral variables pushed
  against a bound by the root LP's objective, the rounder uses the `:regroup`
  schedule.
- `violation_threshold = 5000.0`: the violation-guided rule is turned on when
  the first rounding violates the rows without continuous variables by at
  least this much in total.
- `propagation_budget = 100_000`: rows propagated per propagation call, on
  pure-integer models (unlimited on mixed ones).
- `feasibility_tolerance = 1e-6`: absolute tolerance on integrality and on
  every constraint.
"""
function feasibility_pump(milp::MILP;
                          time_limit::Real = 1800.0,
                          seed::Integer = 1,
                          use_gpu::Bool = CUDA.functional(),
                          verbose::Bool = false,
                          flip_parameter::Int = 20,
                          restart_period::Int = 100,
                          swap_iteration::Int = 10,
                          gap_start::Float64 = 1e-2,
                          gap_decay::Float64 = 0.98,
                          gap_floor::Float64 = 1e-4,
                          residual_ratio::Float64 = 0.1,
                          root_lp_tolerance::Float64 = 1e-2,
                          projection_iteration_limit::Int = 500_000,
                          batch_policy::Symbol = :cost,
                          push_threshold::Float64 = 0.01,
                          violation_threshold::Float64 = 5000.0,
                          propagation_budget::Int = 100_000,
                          feasibility_tolerance::Float64 = 1e-6)
    batch_policy in (:cost, :always) || throw(ArgumentError("batch_policy must be :cost or :always"))
    t0 = time()
    elapsed() = time() - t0
    remaining() = max(time_limit - elapsed(), 1.0)
    n = num_variables(milp); m = num_constraints(milp)
    int_idx = milp.int_idx
    tol = feasibility_tolerance
    # On a model with continuous variables success depends on the continuous
    # part of the projection, so the solvers measure residuals in original
    # units, and the propagation budget is lifted (it would hide from the
    # batch's guard the contradictions that run through continuous rows).
    mixed = num_continuous(milp) > 0
    rng = MersenneTwister(seed)
    rounder = Rounder(milp; max_pops = mixed ? typemax(Int) : propagation_budget, swap_iteration,
                      policy = batch_policy === :cost ? BatchPolicy() : nothing)
    stats = PumpCounters()
    verbose && _print_header(milp, use_gpu)

    # Root LP relaxation.
    x, dual_slack, stats.root_lp_status, root_iterations = solve_lp_relaxation(milp;
        tolerance = root_lp_tolerance, time_limit = remaining(), use_gpu,
        cupdlpx_termination = mixed)
    stats.root_lp_time = elapsed()
    verbose && _log(@sprintf("root LP     %s after %d iterations, %.2f s\n",
                       stats.root_lp_status, root_iterations, stats.root_lp_time))
    if _is_integral(x, int_idx, tol)
        x̃ = _round_to_nearest(x, milp)
        max_violation(milp, x̃) <= tol && return _finish(milp, true, x̃, stats, elapsed(), verbose)
    end

    # First rounding, then the decision on the violation-guided rule.
    rounder.schedule = _batch_schedule(milp, x, dual_slack, push_threshold)
    t = time()
    x̃ = round_guarded(rounder, x, rng, 0)
    stats.rounding_time += time() - t
    total_violation = _total_integer_row_violation(milp, x̃)
    if total_violation >= violation_threshold
        rounder.direction = falses(n)
        rounder.direction[int_idx] .= true
    end
    verbose && _log(@sprintf("rounder     %s schedule, violation-guided rounding %s (first rounding violates by %.4g)\n",
                       rounder.schedule, rounder.direction === nothing ? "off" : "on", total_violation))

    # The projector, built once: scaled data on the device, warm-started at
    # the root LP solution.
    is_eq = milp.row_sense .=== :eq
    is_int = falses(n); is_int[int_idx] .= true
    con_lb = [is_eq[i] ? milp.b[i] : -Inf for i = 1:m]
    sp = scale_problem(milp.A, zeros(n), copy(milp.l), copy(milp.u), con_lb, copy(milp.b);
                       bound_objective_rescaling = false)
    D_c = sp.D_c
    weights = ones(n); weights[int_idx] .= D_c[int_idx]
    projector = PDFPState(sp.A, sp.con_ub, Vector{Bool}(is_eq), Vector{Bool}(is_int),
                          sp.var_lb, sp.var_ub, weights; use_gpu)
    mixed && original_primal_residual!(projector, sp.D_r)
    copyto!(projector.current_x, x ./ D_c)
    copyto!(projector.initial_x, x ./ D_c)
    primal_weight = sqrt(1.0 + (m > 0 ? length(int_idx) / m : 0.0))
    gap = gap_start
    tightening_budget = gap_start

    general = [j for j in int_idx if !(milp.l[j] == 0.0 && milp.u[j] == 1.0)]
    is_general = falses(n); is_general[general] .= true
    ladder = RestartLadder(0.0, 0, flip_parameter, max(div(length(general), 10), 10))
    seen = Set{UInt64}()
    h_current = _hash_integers(x̃, int_idx)
    since_restart = 0

    verbose && _log(@sprintf("%6s %12s %10s %10s %9s %8s %9s\n",
                       "iter", "distance", "violation", "pdhg", "gap tol", "action", "time"))
    while elapsed() < time_limit
        violation = max_violation(milp, x̃)
        if violation <= tol && _is_integral(x̃, int_idx, tol)
            return _finish(milp, true, copy(x̃), stats, elapsed(), verbose)
        end
        stats.iterations += 1
        iteration = stats.iterations
        since_restart += 1

        # Project x̃ onto the LP relaxation.
        t = time()
        set_target!(projector, x̃ ./ D_c)
        gap_used = gap
        params = SolverParameters(; iteration_limit = projection_iteration_limit,
                                  eps_optimal_relative = gap,
                                  eps_feasible_relative = gap * residual_ratio,
                                  time_limit = remaining())
        pdfp_solve!(projector, params, primal_weight)
        x = D_c .* Array(projector.pdhg_x)
        inner = projector.total_count
        # Mixed models: an integral projection whose rounding is infeasible only
        # through its continuous part is re-solved with a tenfold tighter
        # tolerance, five times per run at most (as in FeasPumpCollection).
        while mixed && tightening_budget > 1e-6 && elapsed() < time_limit &&
              _is_integral(x, int_idx, tol) &&
              max_violation(milp, _round_to_nearest(x, milp)) > tol
            tightening_budget *= 0.1
            gap = max(gap * 0.1, 1e-8)
            params = SolverParameters(; iteration_limit = projection_iteration_limit,
                                      eps_optimal_relative = gap,
                                      eps_feasible_relative = gap * residual_ratio,
                                      time_limit = remaining())
            warm_start!(projector)
            pdfp_solve!(projector, params, primal_weight)
            x = D_c .* Array(projector.pdhg_x)
            inner += projector.total_count
        end
        stats.pdhg_iterations += inner
        gap > gap_floor && (gap = max(gap * gap_decay, gap_floor))
        stats.projection_time += time() - t
        distance = 0.0
        @inbounds for j in int_idx
            distance += abs(x[j] - x̃[j])
        end

        if _is_integral(x, int_idx, tol)
            candidate = _round_to_nearest(x, milp)
            if max_violation(milp, candidate) <= tol
                verbose && _print_iteration(iteration, distance, violation, inner, gap_used, "integral", elapsed())
                return _finish(milp, true, candidate, stats, elapsed(), verbose)
            end
        end

        # Round x; if the rounding did not change, flip (FP's stalling move).
        t = time()
        x_round = round_guarded(rounder, x, rng, iteration)
        stats.rounding_time += time() - t
        candidate = copy(x̃)
        action = "round"
        if _differs(x_round, x̃, int_idx)
            candidate .= x_round
        else
            flips = rand(rng, (flip_parameter ÷ 2):(3 * flip_parameter ÷ 2))
            _flip_farthest!(candidate, x, milp, flips)
            action = "flip"
        end

        # Restart on a cycle (a point seen before) and periodically.
        h_candidate = _hash_integers(candidate, int_idx)
        cycled = h_candidate in seen
        periodic = since_restart >= restart_period
        if cycled || periodic
            _perturb!(candidate, x, milp, rng, general, is_general, ladder, iteration)
            stats.perturbations += 1
            since_restart = 0
            for _ = 1:10            # until the point is new
                h = _hash_integers(candidate, int_idx)
                (h in seen || h == h_current) || break
                _perturb!(candidate, x, milp, rng, general, is_general, ladder, iteration)
            end
            action = cycled ? "cycle" : "restart"
        end
        push!(seen, h_current)
        h_current = (cycled || periodic) ? _hash_integers(candidate, int_idx) : h_candidate
        x̃ = candidate
        verbose && (iteration <= 10 || iteration % 10 == 0) &&
            _print_iteration(iteration, distance, violation, inner, gap_used, action, elapsed())
    end
    found = max_violation(milp, x̃) <= tol && _is_integral(x̃, int_idx, tol)
    return _finish(milp, found, x̃, stats, elapsed(), verbose)
end

function _finish(milp::MILP, found::Bool, x̃::Vector{Float64}, stats::PumpCounters,
                 time::Float64, verbose::Bool)
    violation = max_violation(milp, x̃)
    objective = dot(milp.c, x̃) + milp.c0
    if verbose
        if found
            _log(@sprintf("FEASIBLE    objective %.10g after %d iterations, %.2f s\n",
                    objective, stats.iterations, time))
        else
            _log(@sprintf("NOT FOUND   after %d iterations, %.2f s (max violation %.3g)\n",
                    stats.iterations, time, violation))
        end
        _log(@sprintf("            root LP %.2f s, projections %.2f s, rounding %.2f s\n",
                stats.root_lp_time, stats.projection_time, stats.rounding_time))
    end
    return PumpResult(found, x̃, objective, violation, stats.iterations, stats.perturbations,
                      stats.pdhg_iterations, stats.root_lp_status, stats.root_lp_time,
                      stats.projection_time, stats.rounding_time, time)
end

# Formatted with @sprintf and written with print, so that the compiled code does
# not depend on the type of stdout (a warm-up that prints to devnull then also
# compiles the log for a terminal or a file).
_log(s::String) = print(stdout, s)

function _print_header(milp::MILP, use_gpu::Bool)
    _log(@sprintf("PDFP feasibility pump: %s\n", milp.name))
    _log(@sprintf("model       %d rows, %d columns (%d integer, %d continuous), %d nonzeros\n",
            num_constraints(milp), num_variables(milp), length(milp.int_idx),
            num_continuous(milp), nnz(milp.A)))
    _log(@sprintf("device      %s\n", use_gpu ? CUDA.name(CUDA.device()) : "CPU"))
end

_print_iteration(iteration, distance, violation, inner, gap, action, time) =
    _log(@sprintf("%6d %12.6g %10.3g %10d %9.2e %8s %8.2fs\n",
            iteration, distance, violation, inner, gap, action, time))
