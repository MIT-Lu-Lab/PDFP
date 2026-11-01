"""
    MILP

A mixed-integer linear program in the form

    min  c'x + c0
    s.t. A x ≤ b    (rows with `row_sense[i] == :le`)
         A x = b    (rows with `row_sense[i] == :eq`)
         l ≤ x ≤ u
         x_j ∈ ℤ    for j ∈ `int_idx`

Ranged rows are split into two `:le` rows and `≥` rows are negated. Infinite
bounds are stored as `±Inf`.
"""
struct MILP
    c::Vector{Float64}
    c0::Float64
    A::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
    row_sense::Vector{Symbol}
    l::Vector{Float64}
    u::Vector{Float64}
    int_idx::Vector{Int}
    var_names::Vector{String}
    name::String
end

num_variables(milp::MILP) = length(milp.c)
num_constraints(milp::MILP) = length(milp.b)
num_continuous(milp::MILP) = length(milp.c) - length(milp.int_idx)

const MPS_INFINITY = 1.0e20

_is_infinite(v::Real) = !isfinite(v) || abs(v) >= MPS_INFINITY

"""
    read_mps(path) -> MILP

Read a free-format MPS file (optionally gzipped, `.mps.gz`). A maximization
objective is negated, so the returned model is always a minimization.
"""
function read_mps(path::AbstractString)
    qp = if endswith(path, ".gz")
        open(io -> readqps(GzipDecompressorStream(io); mpsformat = :free), path)
    else
        readqps(path; mpsformat = :free)
    end
    n = qp.nvar
    sign = qp.objsense == :max ? -1.0 : 1.0
    c = sign .* qp.c
    c0 = sign * qp.c0

    l = [_is_infinite(v) ? (v < 0 ? -Inf : Inf) : v for v in qp.lvar]
    u = [_is_infinite(v) ? (v < 0 ? -Inf : Inf) : v for v in qp.uvar]
    int_idx = Int[]
    for j = 1:n
        if qp.vartypes[j] == QPSReader.VTYPE_Binary
            push!(int_idx, j)
            l[j] = max(l[j], 0.0)
            u[j] = min(u[j], 1.0)
        elseif qp.vartypes[j] == QPSReader.VTYPE_Integer
            push!(int_idx, j)
        end
    end

    # Each original row maps to one or two canonical rows, each with a sign.
    b = Float64[]
    row_sense = Symbol[]
    canonical = [Tuple{Int,Float64}[] for _ = 1:qp.ncon]
    for i = 1:qp.ncon
        qp.contypes[i] == QPSReader.RTYPE_Objective && continue
        lo, hi = qp.lcon[i], qp.ucon[i]
        if !_is_infinite(lo) && !_is_infinite(hi) && lo == hi
            push!(b, hi); push!(row_sense, :eq); push!(canonical[i], (length(b), 1.0))
            continue
        end
        if !_is_infinite(hi)
            push!(b, hi); push!(row_sense, :le); push!(canonical[i], (length(b), 1.0))
        end
        if !_is_infinite(lo)
            push!(b, -lo); push!(row_sense, :le); push!(canonical[i], (length(b), -1.0))
        end
    end

    I = Int[]; J = Int[]; V = Float64[]
    for k in eachindex(qp.avals)
        for (row, s) in canonical[qp.arows[k]]
            push!(I, row); push!(J, qp.acols[k]); push!(V, s * qp.avals[k])
        end
    end
    A = sparse(I, J, V, length(b), n)
    name = qp.name === nothing ? replace(basename(path), r"\.mps(\.gz)?$" => "") : qp.name
    return MILP(c, c0, A, b, row_sense, l, u, int_idx, qp.varnames, name)
end

"""
    max_violation(milp, x) -> Float64

Largest violation of a linear constraint `A x ≤ b` / `A x = b` at `x`.
Variable bounds and integrality are not checked.
"""
function max_violation(milp::MILP, x::AbstractVector)
    Ax = milp.A * x
    v = 0.0
    @inbounds for i in eachindex(milp.b)
        d = Ax[i] - milp.b[i]
        if milp.row_sense[i] === :le
            d > v && (v = d)
        else
            abs(d) > v && (v = abs(d))
        end
    end
    return v
end

"""
    is_feasible(milp, x; tol = 1e-6) -> Bool

Whether `x` satisfies integrality, the variable bounds and every constraint of
`milp`, each to the absolute tolerance `tol`.
"""
function is_feasible(milp::MILP, x::AbstractVector; tol::Float64 = 1e-6)
    length(x) == num_variables(milp) || return false
    _is_integral(x, milp.int_idx, tol) || return false
    all(j -> milp.l[j] - tol <= x[j] <= milp.u[j] + tol, eachindex(x)) || return false
    return max_violation(milp, x) <= tol
end

_is_integral(x::AbstractVector, idx, tol::Float64 = 1e-6) =
    all(j -> abs(x[j] - round(x[j])) <= tol, idx)

"""
    write_solution(path, milp, x)

Write `x` as one `name<TAB>value` line per variable. MPS names may contain
commas, so the separator is a tab.
"""
function write_solution(path::AbstractString, milp::MILP, x::AbstractVector)
    open(path, "w") do io
        for j in eachindex(x)
            println(io, milp.var_names[j], '\t', repr(x[j]))
        end
    end
    return path
end
