# Rounding: FP 2.0's constraint-propagation rounding (CPR) and our guarded
# two-phase CPR.
#
# CPR ranks the integer variables by fractionality (binaries first), then fixes
# them one at a time to their rounded LP value, propagating each fixing. The
# guarded rounder first fixes, in a single propagation, every variable whose LP
# value is already integral (the batch), then rounds the rest sequentially. If
# the batch makes a row infeasible, its work is discarded and the call falls
# back to sequential CPR.

"""
    BatchPolicy

Online rule that decides, per call, whether to try the batch. Before each call
it tries the batch iff `p (w + s) + (1 - p) h < s`, where `s` is the cost of a
sequential call, `h` the cost of a call whose batch held, `w` the work wasted
by a batch that failed, and `p` the failure rate of tried batches. Costs are
exponential moving averages with weight `ξ` on the newest observation; the
counts behind `p` decay by `ν` per tried call. The batch is tried until a
failure has been seen, and again after `reprobe` consecutive skips.
"""
mutable struct BatchPolicy
    ξ::Float64
    ν::Float64
    reprobe::Int
    n_failed::Float64
    n_held::Float64
    s::Float64
    h::Float64
    w::Float64
    skipped::Int          # consecutive skips
end

BatchPolicy(; ξ = 0.3, ν = 0.9, reprobe = 20) = BatchPolicy(ξ, ν, reprobe, 0.0, 0.0, NaN, NaN, NaN, 0)

_ema(P::BatchPolicy, old::Float64, x::Float64) = isnan(old) ? x : (1.0 - P.ξ) * old + P.ξ * x

function wants_batch(P::BatchPolicy)
    P.skipped >= P.reprobe && return true
    (isnan(P.s) || P.n_failed == 0.0) && return true
    h = isnan(P.h) ? 0.0 : P.h
    p = P.n_failed / (P.n_failed + P.n_held)
    return p * (P.w + P.s) + (1.0 - p) * h < P.s
end

function observe_batch!(P::BatchPolicy, failed::Bool, cost::Float64, wasted::Float64)
    P.n_failed *= P.ν; P.n_held *= P.ν; P.skipped = 0
    if failed
        P.n_failed += 1.0
        P.w = _ema(P, P.w, wasted)
        P.s = _ema(P, P.s, max(cost - wasted, 0.0))
    else
        P.n_held += 1.0
        P.h = _ema(P, P.h, cost)
    end
    return nothing
end

function observe_skip!(P::BatchPolicy, cost::Float64)
    P.s = _ema(P, P.s, cost)
    P.skipped += 1
    return nothing
end

"""
    Rounder

Rounding state that persists across the calls of one pump run.

- `schedule`: `:all` batches every LP-integral variable; `:regroup` first
  rounds, sequentially, the fractional variables that the random swaps moved
  ahead of the last LP-integral one in the ranking.
- `direction`: integer columns on which the violation-guided rule is on, or
  `nothing`.
- `max_pops`: propagation budget per propagation call.
- `policy`: the batch policy, or `nothing` to try the batch on every call.
"""
mutable struct Rounder
    milp::MILP
    data::Union{Nothing,PropagationData}     # built on the first call
    schedule::Symbol
    direction::Union{Nothing,BitVector}
    max_pops::Int
    swap_iteration::Int                      # random ranking swaps from this iteration on
    policy::Union{Nothing,BatchPolicy}
    n_calls::Int
    n_batched::Int                           # calls whose batch held
    n_fallbacks::Int                         # calls whose batch failed
end

Rounder(milp::MILP; schedule::Symbol = :all, max_pops::Int = typemax(Int),
        swap_iteration::Int = 10, policy::Union{Nothing,BatchPolicy} = BatchPolicy()) =
    Rounder(milp, nothing, schedule, nothing, max_pops, swap_iteration, policy, 0, 0, 0)

function _new_state(r::Rounder)
    r.data === nothing && (r.data = PropagationData(r.milp))
    return PropagationState(r.data, r.max_pops)
end

_rank_key(milp::MILP, j::Int, x::AbstractVector) =
    abs(x[j] - round(x[j])) - ((milp.l[j] == 0.0 && milp.u[j] == 1.0) ? 10.0 : 0.0)

# Ascending fractionality, binaries first; from `swap_iteration` on, |I|/10
# random transpositions (FP 2.0's anti-cycling rule).
function _ranking(r::Rounder, x_star::AbstractVector, rng::AbstractRNG, iteration::Int)
    ranked = collect(r.milp.int_idx)
    ranked = ranked[sortperm([_rank_key(r.milp, j, x_star) for j in ranked])]
    if iteration >= r.swap_iteration && length(ranked) >= 10
        for _ = 1:max(1, div(length(ranked), 10))
            a = rand(rng, 1:length(ranked)); b = rand(rng, 1:length(ranked))
            ranked[a], ranked[b] = ranked[b], ranked[a]
        end
    end
    return ranked
end

_row_violation(milp::MILP, i::Int, activity::Float64) =
    milp.row_sense[i] === :le ? max(0.0, activity - milp.b[i]) : abs(activity - milp.b[i])

# Violation-guided rounding: of the two ends of j's domain, the one that adds
# less violation to j's rows, given the activity of the variables fixed so far;
# `target` on a tie.
function _guided_value(milp::MILP, s::PropagationState, j::Int, target::Float64,
                       activity::Vector{Float64})
    s.u[j] - s.l[j] > BOUND_TOLERANCE || return target
    lo = ceil(s.l[j] - BOUND_TOLERANCE); hi = floor(s.u[j] + BOUND_TOLERANCE)
    hi > lo || return target
    rows = rowvals(milp.A); vals = nonzeros(milp.A)
    score(v) = begin
        t = 0.0
        @inbounds for p in nzrange(milp.A, j)
            i = rows[p]
            t += _row_violation(milp, i, activity[i] + vals[p] * v) - _row_violation(milp, i, activity[i])
        end
        t
    end
    score_lo = score(lo); score_hi = score(hi)
    score_lo < score_hi - 1e-12 && return lo
    score_hi < score_lo - 1e-12 && return hi
    return target
end

function _add_activity!(milp::MILP, activity::Vector{Float64}, j::Int, v::Float64)
    rows = rowvals(milp.A); vals = nonzeros(milp.A)
    @inbounds for p in nzrange(milp.A, j)
        activity[rows[p]] += vals[p] * v
    end
end

# Sequential CPR over `idx`: fix each variable to its rounded LP value (or the
# guided value) and propagate.
function _round_sequentially!(r::Rounder, s::PropagationState, x̃::Vector{Float64},
                              idx::AbstractVector{Int}, x_star::AbstractVector,
                              activity::Union{Nothing,Vector{Float64}})
    for j in idx
        if is_fixed(s, j)
            x̃[j] = s.l[j]
            continue
        end
        target = round(clamp(x_star[j], s.l[j], s.u[j]))
        if target < s.l[j] - BOUND_TOLERANCE || target > s.u[j] + BOUND_TOLERANCE
            other = round(clamp(x_star[j] + (target < s.l[j] ? 1.0 : -1.0), s.l[j], s.u[j]))
            target = clamp(other, s.l[j], s.u[j])
        end
        if activity !== nothing && r.direction[j]
            target = _guided_value(r.milp, s, j, target, activity)
        end
        fix!(s, j, target)
        propagate!(s)
        activity !== nothing && _add_activity!(r.milp, activity, j, target)
        x̃[j] = target
    end
    return nothing
end

function _read_integers!(r::Rounder, s::PropagationState, x̃::Vector{Float64}, x_star::AbstractVector)
    @inbounds for j in r.milp.int_idx
        x̃[j] = is_fixed(s, j) ? s.l[j] : round(clamp(x_star[j], s.l[j], s.u[j]))
    end
    return x̃
end

_activity(r::Rounder) = r.direction === nothing ? nothing : zeros(num_constraints(r.milp))

"""
    round_sequential(r, x_star, rng, iteration) -> x̃

FP 2.0's constraint-propagation rounding of `x_star`. Continuous coordinates
keep their values.
"""
function round_sequential(r::Rounder, x_star::AbstractVector, rng::AbstractRNG, iteration::Int)
    s = _new_state(r)
    x̃ = collect(Float64, x_star)
    ranked = _ranking(r, x_star, rng, iteration)
    _round_sequentially!(r, s, x̃, ranked, x_star, _activity(r))
    return _read_integers!(r, s, x̃, x_star)
end

"""
    round_guarded(r, x_star, rng, iteration) -> x̃

Guarded two-phase CPR of `x_star`: batch the LP-integral variables, fall back
to `round_sequential` if the batch makes a row infeasible.
"""
function round_guarded(r::Rounder, x_star::AbstractVector, rng::AbstractRNG, iteration::Int)
    r.n_calls += 1
    P = r.policy
    t0 = P === nothing ? UInt64(0) : time_ns()
    if P !== nothing && !wants_batch(P)
        x̃ = round_sequential(r, x_star, rng, iteration)
        observe_skip!(P, (time_ns() - t0) / 1e9)
        return x̃
    end
    s = _new_state(r)
    x̃ = collect(Float64, x_star)
    activity = _activity(r)
    ranked = _ranking(r, x_star, rng, iteration)

    integral(j) = abs(x_star[j] - round(x_star[j])) == 0.0
    if r.schedule === :regroup
        last = something(findlast(integral, ranked), 0)
        before = [ranked[k] for k = 1:last if !integral(ranked[k])]
        batch = [ranked[k] for k = 1:last if integral(ranked[k])]
        after = ranked[last+1:end]
    else
        before = Int[]
        batch = [j for j in ranked if integral(j)]
        after = [j for j in ranked if !integral(j)]
    end

    _round_sequentially!(r, s, x̃, before, x_star, activity)
    @inbounds for j in batch
        if is_fixed(s, j)
            x̃[j] = s.l[j]
            continue
        end
        target = round(clamp(x_star[j], s.l[j], s.u[j]))
        fix!(s, j, target)
        activity !== nothing && _add_activity!(r.milp, activity, j, target)
        x̃[j] = target
    end
    propagate!(s)

    if s.infeasible           # the guard: discard the batch
        r.n_fallbacks += 1
        wasted = P === nothing ? 0.0 : (time_ns() - t0) / 1e9
        x̃ = round_sequential(r, x_star, rng, iteration)
        P === nothing || observe_batch!(P, true, (time_ns() - t0) / 1e9, wasted)
        return x̃
    end
    r.n_batched += 1
    _round_sequentially!(r, s, x̃, after, x_star, activity)
    _read_integers!(r, s, x̃, x_star)
    P === nothing || observe_batch!(P, false, (time_ns() - t0) / 1e9, 0.0)
    return x̃
end
