# Diagonal preconditioning shared by both PDHG solvers, following cuPDLPx:
# 10 rounds of Ruiz equilibration, one Pock-Chambolle pass with α = 1, and
# optionally the bound/objective rescaling. With D_r, D_c the row and column
# scalings, the scaled problem is
#
#   min (D_c c)' x̂   s.t.  D_r con_lb ≤ (D_r A D_c) x̂ ≤ D_r con_ub,
#                          var_lb ./ D_c ≤ x̂ ≤ var_ub ./ D_c,
#
# and x = D_c .* x̂ recovers the original variables.

const SCALING_EPSILON = 1e-12

struct ScaledProblem
    A::SparseMatrixCSC{Float64,Int}
    c::Vector{Float64}
    var_lb::Vector{Float64}
    var_ub::Vector{Float64}
    con_lb::Vector{Float64}
    con_ub::Vector{Float64}
    D_r::Vector{Float64}
    D_c::Vector{Float64}
    bound_scale::Float64      # constraint_scale of the bound/objective rescaling
    objective_scale::Float64
end

function scale_problem(A::SparseMatrixCSC{Float64,Int}, c::Vector{Float64},
                       var_lb::Vector{Float64}, var_ub::Vector{Float64},
                       con_lb::Vector{Float64}, con_ub::Vector{Float64};
                       ruiz_iterations::Int = 10, pock_chambolle_alpha::Float64 = 1.0,
                       bound_objective_rescaling::Bool = true)
    A = copy(A)
    m, n = size(A)
    D_r = ones(m); D_c = ones(n)
    rows = rowvals(A); vals = nonzeros(A)

    for _ = 1:ruiz_iterations
        row_max = zeros(m); col_max = zeros(n)
        @inbounds for j = 1:n, p in nzrange(A, j)
            v = abs(vals[p]); i = rows[p]
            v > row_max[i] && (row_max[i] = v)
            v > col_max[j] && (col_max[j] = v)
        end
        rscale = [row_max[i] < SCALING_EPSILON ? 1.0 : 1 / sqrt(row_max[i]) for i = 1:m]
        cscale = [col_max[j] < SCALING_EPSILON ? 1.0 : 1 / sqrt(col_max[j]) for j = 1:n]
        @inbounds for j = 1:n, p in nzrange(A, j)
            vals[p] *= rscale[rows[p]] * cscale[j]
        end
        D_r .*= rscale; D_c .*= cscale
    end

    # Pock-Chambolle: α on the rows, 2 - α on the columns.
    α = pock_chambolle_alpha
    row_sum = zeros(m); col_sum = zeros(n)
    @inbounds for j = 1:n, p in nzrange(A, j)
        v = abs(vals[p]); i = rows[p]
        row_sum[i] += v^α
        col_sum[j] += v^(2 - α)
    end
    rscale = [row_sum[i] < SCALING_EPSILON ? 1.0 : 1 / sqrt(row_sum[i]) for i = 1:m]
    cscale = [col_sum[j] < SCALING_EPSILON ? 1.0 : 1 / sqrt(col_sum[j]) for j = 1:n]
    @inbounds for j = 1:n, p in nzrange(A, j)
        vals[p] *= rscale[rows[p]] * cscale[j]
    end
    D_r .*= rscale; D_c .*= cscale

    c_s = c .* D_c
    var_lb_s = [isfinite(var_lb[j]) ? var_lb[j] / D_c[j] : var_lb[j] for j = 1:n]
    var_ub_s = [isfinite(var_ub[j]) ? var_ub[j] / D_c[j] : var_ub[j] for j = 1:n]
    con_lb_s = [isfinite(con_lb[i]) ? con_lb[i] * D_r[i] : con_lb[i] for i = 1:m]
    con_ub_s = [isfinite(con_ub[i]) ? con_ub[i] * D_r[i] : con_ub[i] for i = 1:m]

    # Bound/objective rescaling: bring ‖bounds‖ and ‖c‖ to O(1), which makes
    # an initial primal weight of 1 appropriate.
    bound_scale = 1.0; objective_scale = 1.0
    if bound_objective_rescaling
        bound_sq = 0.0
        @inbounds for i = 1:m
            finite_lo = isfinite(con_lb[i]); finite_hi = isfinite(con_ub[i])
            lo = con_lb_s[i]; hi = con_ub_s[i]
            if finite_lo && (!finite_hi || abs(lo - hi) > SCALING_EPSILON)
                bound_sq += lo * lo
            end
            finite_hi && (bound_sq += hi * hi)
        end
        bound_scale = 1.0 / (sqrt(bound_sq) + 1.0)
        objective_scale = 1.0 / (sqrt(sum(abs2, c_s)) + 1.0)
        @inbounds for i = 1:m
            isfinite(con_lb[i]) && (con_lb_s[i] *= bound_scale)
            isfinite(con_ub[i]) && (con_ub_s[i] *= bound_scale)
        end
        @inbounds for j = 1:n
            isfinite(var_lb[j]) && (var_lb_s[j] *= bound_scale)
            isfinite(var_ub[j]) && (var_ub_s[j] *= bound_scale)
            c_s[j] *= objective_scale
        end
    end
    return ScaledProblem(A, c_s, var_lb_s, var_ub_s, con_lb_s, con_ub_s, D_r, D_c,
                         bound_scale, objective_scale)
end

"""
    estimate_max_singular_value(A, A_t, m; max_iterations, tolerance) -> Float64

‖A‖₂ by power iteration on A A' (in the space of the shorter dimension of a
wide matrix), from a fixed random start so that runs are reproducible.
`A` and `A_t` may live on the CPU or the GPU.
"""
function estimate_max_singular_value(A, A_t, m::Int, scratch_m, scratch_n;
                                     max_iterations::Int = 5000, tolerance::Float64 = 1e-4)
    eig = similar(scratch_m); nxt = similar(scratch_m); dual = similar(scratch_n)
    copyto!(eig, randn(MersenneTwister(20240517), Float64, m))
    sigma_sq = 1.0
    for _ = 1:max_iterations
        copyto!(nxt, eig)
        nrm = norm(nxt)
        nrm == 0 && break
        nxt ./= nrm
        mul!(dual, A_t, nxt)
        mul!(eig, A, dual)
        sigma_sq = dot(nxt, eig)
        @. nxt = eig - sigma_sq * nxt
        norm(nxt) < tolerance && break
    end
    return sqrt(max(sigma_sq, 0.0))
end
