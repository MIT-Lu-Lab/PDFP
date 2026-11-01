# Bound propagation over the linear constraints (activity-based bound
# strengthening), the engine behind constraint-propagation rounding (CPR) of
# FP 2.0 (Fischetti and Salvagnin 2009). Each row keeps its minimum and maximum
# activity under the current domains; a domain change updates them
# incrementally and queues the rows that may now tighten further bounds.

const BOUND_TOLERANCE = 1e-6
const INFEASIBILITY_TOLERANCE = 1e-6
const MAX_TIGHTENINGS = 10          # wake-up cap per variable and direction (FP 2.0 §5.3)
const DOMAIN_INFINITY = 1e12        # stands in for infinite bounds in activity arithmetic

# Row classes with a specialised propagator; both give the same tightenings as
# the general one, faster.
const ROW_GENERAL = 0x00
const ROW_CARDINALITY = 0x01        # all coefficients 1 over binaries
const ROW_KNAPSACK = 0x02           # ≤ row, all coefficients positive, over binaries

"Everything about the propagation that depends on the model only."
struct PropagationData
    A::SparseMatrixCSC{Float64,Int}
    A_t::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
    row_sense::Vector{Symbol}
    l::Vector{Float64}
    u::Vector{Float64}
    is_int::BitVector
    L_min::Vector{Float64}
    L_max::Vector{Float64}
    row_kind::Vector{UInt8}
    row_maxcoef::Vector{Float64}
    # max_j |a_ij| (u_j - l_j) over the original bounds; a row whose slack
    # exceeds it cannot tighten anything
    row_maxroom::Vector{Float64}
end

function PropagationData(milp::MILP)
    n = num_variables(milp); m = num_constraints(milp)
    l = [isfinite(v) ? v : -DOMAIN_INFINITY for v in milp.l]
    u = [isfinite(v) ? v : DOMAIN_INFINITY for v in milp.u]
    is_int = falses(n)
    is_int[milp.int_idx] .= true
    A_t = sparse(milp.A')
    rows = rowvals(A_t); vals = nonzeros(A_t)
    L_min = zeros(m); L_max = zeros(m)
    row_kind = fill(ROW_GENERAL, m); row_maxcoef = zeros(m); row_maxroom = zeros(m)
    @inbounds for i = 1:m
        lmin = 0.0; lmax = 0.0
        for p in nzrange(A_t, i)
            j = rows[p]; a = vals[p]
            if a > 0
                lmin += a * l[j]; lmax += a * u[j]
            else
                lmin += a * u[j]; lmax += a * l[j]
            end
        end
        L_min[i] = lmin; L_max[i] = lmax

        all_one = true; all_pos = true; all_bin = true; maxcoef = 0.0
        for p in nzrange(A_t, i)
            j = rows[p]; a = vals[p]
            abs(a) > maxcoef && (maxcoef = abs(a))
            a == 1.0 || (all_one = false)
            a > 0.0 || (all_pos = false)
            (is_int[j] && milp.l[j] == 0.0 && milp.u[j] == 1.0) || (all_bin = false)
            (all_bin && (all_one || all_pos)) || break
        end
        row_maxcoef[i] = maxcoef
        if all_bin && all_one
            row_kind[i] = ROW_CARDINALITY
        elseif all_bin && all_pos && milp.row_sense[i] === :le
            row_kind[i] = ROW_KNAPSACK
        end

        room = 0.0
        for p in nzrange(A_t, i)
            width = milp.u[rows[p]] - milp.l[rows[p]]
            r = isfinite(width) ? abs(vals[p]) * width : Inf
            r > room && (room = r)
        end
        row_maxroom[i] = room
    end
    return PropagationData(milp.A, A_t, milp.b, milp.row_sense, l, u, is_int,
                           L_min, L_max, row_kind, row_maxcoef, row_maxroom)
end

"The domains of one rounding pass and the propagation queue."
mutable struct PropagationState
    data::PropagationData
    l::Vector{Float64}
    u::Vector{Float64}
    L_min::Vector{Float64}
    L_max::Vector{Float64}
    in_queue::BitVector
    queue::Vector{Int}
    tighten_lo::Vector{Int}
    tighten_hi::Vector{Int}
    row_maxroom::Vector{Float64}    # live upper bound on a row's max room
    max_pops::Int                   # propagation budget per call to `propagate!`
    infeasible::Bool                # some row was found infeasible
end

function PropagationState(d::PropagationData, max_pops::Int)
    m = length(d.b); n = length(d.l)
    return PropagationState(d, copy(d.l), copy(d.u), copy(d.L_min), copy(d.L_max),
                            falses(m), Int[], zeros(Int, n), zeros(Int, n),
                            copy(d.row_maxroom), max_pops, false)
end

@inline function _enqueue!(s::PropagationState, i::Int)
    if !s.in_queue[i]
        push!(s.queue, i); s.in_queue[i] = true
    end
end

# Lower u[j] to new_u (rounded down for integers). A bound that would cross the
# lower bound fixes the variable there. Returns whether u[j] changed.
function _tighten_upper!(s::PropagationState, j::Int, new_u::Float64)
    u_old = s.u[j]
    new_u < u_old - BOUND_TOLERANCE || return false
    nu = s.data.is_int[j] ? floor(new_u + BOUND_TOLERANCE) : new_u
    nu < s.l[j] - BOUND_TOLERANCE && (nu = s.l[j])
    nu = max(nu, s.l[j])
    nu >= u_old - BOUND_TOLERANCE && return false
    s.u[j] = nu
    s.tighten_hi[j] += 1
    # past the cap the bound is still committed, but its rows are not woken
    wake = s.tighten_hi[j] <= MAX_TIGHTENINGS || abs(nu - s.l[j]) < BOUND_TOLERANCE
    A = s.data.A; rows = rowvals(A); vals = nonzeros(A)
    du = nu - u_old
    @inbounds for p in nzrange(A, j)
        i = rows[p]; a = vals[p]
        a > 0 ? (s.L_max[i] += a * du) : (s.L_min[i] += a * du)
        wake && _enqueue!(s, i)
    end
    return true
end

function _tighten_lower!(s::PropagationState, j::Int, new_l::Float64)
    l_old = s.l[j]
    new_l > l_old + BOUND_TOLERANCE || return false
    nl = s.data.is_int[j] ? ceil(new_l - BOUND_TOLERANCE) : new_l
    nl > s.u[j] + BOUND_TOLERANCE && (nl = s.u[j])
    nl = min(nl, s.u[j])
    nl <= l_old + BOUND_TOLERANCE && return false
    s.l[j] = nl
    s.tighten_lo[j] += 1
    wake = s.tighten_lo[j] <= MAX_TIGHTENINGS || abs(s.u[j] - nl) < BOUND_TOLERANCE
    A = s.data.A; rows = rowvals(A); vals = nonzeros(A)
    dl = nl - l_old
    @inbounds for p in nzrange(A, j)
        i = rows[p]; a = vals[p]
        a > 0 ? (s.L_min[i] += a * dl) : (s.L_max[i] += a * dl)
        wake && _enqueue!(s, i)
    end
    return true
end

# Σ_{j∈S} x_j (≤ or =) b over binaries: no tightening is possible unless a
# slack is below 1, and no division is needed.
function _propagate_cardinality_row!(s::PropagationState, i::Int)
    sense = s.data.row_sense[i]; b = s.data.b[i]
    slack_up = b - s.L_min[i]
    slack_dn = s.L_max[i] - b
    if slack_up < -INFEASIBILITY_TOLERANCE || (sense === :eq && slack_dn < -INFEASIBILITY_TOLERANCE)
        s.infeasible = true
        return
    end
    up = slack_up < 1.0 - BOUND_TOLERANCE
    dn = sense === :eq && slack_dn < 1.0 - BOUND_TOLERANCE
    (up || dn) || return
    A_t = s.data.A_t; rows = rowvals(A_t)
    @inbounds for p in nzrange(A_t, i)
        j = rows[p]
        up && _tighten_upper!(s, j, s.l[j] + slack_up)
        dn && _tighten_lower!(s, j, s.u[j] - slack_dn)
    end
end

# Σ a_ij x_j ≤ b with a_ij > 0 over binaries: x_j is fixed to 0 iff a_ij > slack.
function _propagate_knapsack_row!(s::PropagationState, i::Int)
    slack_up = s.data.b[i] - s.L_min[i]
    if slack_up < -INFEASIBILITY_TOLERANCE
        s.infeasible = true
        return
    end
    s.data.row_maxcoef[i] <= slack_up + BOUND_TOLERANCE && return
    A_t = s.data.A_t; rows = rowvals(A_t); vals = nonzeros(A_t)
    @inbounds for p in nzrange(A_t, i)
        j = rows[p]
        vals[p] > slack_up + BOUND_TOLERANCE && _tighten_upper!(s, j, s.l[j])
    end
end

function _propagate_row!(s::PropagationState, i::Int)
    kind = s.data.row_kind[i]
    kind == ROW_CARDINALITY && return _propagate_cardinality_row!(s, i)
    kind == ROW_KNAPSACK && return _propagate_knapsack_row!(s, i)

    sense = s.data.row_sense[i]; b = s.data.b[i]
    slack_up = b - s.L_min[i]
    slack_dn = s.L_max[i] - b
    if slack_up < -INFEASIBILITY_TOLERANCE || (sense === :eq && slack_dn < -INFEASIBILITY_TOLERANCE)
        s.infeasible = true
        return
    end
    # A variable can tighten only if its room |a_ij| (u_j - l_j) exceeds the slack.
    threshold = (sense === :le ? slack_up : min(slack_up, slack_dn)) + BOUND_TOLERANCE
    s.row_maxroom[i] <= threshold && return
    A_t = s.data.A_t; rows = rowvals(A_t); vals = nonzeros(A_t)
    max_room = 0.0
    @inbounds for p in nzrange(A_t, i)
        room = abs(vals[p]) * (s.u[rows[p]] - s.l[rows[p]])
        room > max_room && (max_room = room)
    end
    s.row_maxroom[i] = max_room
    sense === :le && max_room <= slack_up + BOUND_TOLERANCE && return
    @inbounds for p in nzrange(A_t, i)
        j = rows[p]; a = vals[p]
        if a > 0
            _tighten_upper!(s, j, s.l[j] + slack_up / a)
        else
            _tighten_lower!(s, j, s.u[j] + slack_up / a)
        end
        if sense === :eq
            if a > 0
                _tighten_lower!(s, j, s.u[j] - slack_dn / a)
            else
                _tighten_upper!(s, j, s.l[j] - slack_dn / a)
            end
        end
    end
end

"Propagate the queued rows until the queue is empty or the budget is spent."
function propagate!(s::PropagationState)
    pops = 0
    while !isempty(s.queue) && pops < s.max_pops
        i = popfirst!(s.queue)
        s.in_queue[i] = false
        _propagate_row!(s, i)
        pops += 1
    end
    return pops
end

"""
    fix!(s, j, value)

Fix x_j = value and queue its rows (the caller propagates). A value outside
the current domain marks the state infeasible and changes nothing.
"""
function fix!(s::PropagationState, j::Int, value::Float64)
    if value < s.l[j] - BOUND_TOLERANCE || value > s.u[j] + BOUND_TOLERANCE
        s.infeasible = true
        return false
    end
    v = clamp(value, s.l[j], s.u[j])
    l_old = s.l[j]; u_old = s.u[j]
    abs(v - l_old) < BOUND_TOLERANCE && abs(v - u_old) < BOUND_TOLERANCE && return true
    s.l[j] = v; s.u[j] = v
    s.tighten_lo[j] += 1; s.tighten_hi[j] += 1
    A = s.data.A; rows = rowvals(A); vals = nonzeros(A)
    dl = v - l_old; du = v - u_old
    @inbounds for p in nzrange(A, j)
        i = rows[p]; a = vals[p]
        if a > 0
            s.L_min[i] += a * dl; s.L_max[i] += a * du
        else
            s.L_min[i] += a * du; s.L_max[i] += a * dl
        end
        _enqueue!(s, i)
    end
    return true
end

is_fixed(s::PropagationState, j::Int) = abs(s.u[j] - s.l[j]) < BOUND_TOLERANCE
