# Fused CUDA kernels for the PDLP step: one launch per primal or dual update
# instead of several broadcasts. Selected by dispatch on CuVector states.

function _pdlp_primal_kernel!(pdhg_primal, dual_slack, reflected_primal, current_primal,
                              initial_primal, c, dual_product, var_lb, var_ub,
                              step, weight, ω, omw, omwt, is_major, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            cx = current_primal[i]
            temp = cx - step * (c[i] - dual_product[i])
            pdhg = clamp(temp, var_lb[i], var_ub[i])
            pdhg_primal[i] = pdhg
            is_major && (dual_slack[i] = (pdhg - temp) / step)
            refl = 2.0 * pdhg - cx
            reflected_primal[i] = refl
            current_primal[i] = weight * (ω * refl + omw * cx) + omwt * initial_primal[i]
        end
    end
    return nothing
end

function primal_step!(s::PDLPState{V}, k_offset::Int, ω::Float64,
                      is_major::Bool) where {V<:CuVector{Float64}}
    step = s.step_size / s.primal_weight
    k = s.inner_count + k_offset
    weight = k / (k + 1.0); omw = 1.0 - ω; omwt = 1.0 - weight
    @cuda threads=256 blocks=cld(s.n, 256) _pdlp_primal_kernel!(
        s.pdhg_primal, s.dual_slack, s.reflected_primal, s.current_primal,
        s.initial_primal, s.c, s.dual_product, s.var_lb, s.var_ub,
        step, weight, ω, omw, omwt, is_major, s.n)
    return nothing
end

function _pdlp_dual_kernel!(pdhg_dual, reflected_dual, current_dual, initial_dual,
                            primal_product, con_lb, con_ub,
                            step, weight, ω, omw, omwt, is_major, m)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= m
        @inbounds begin
            cy = current_dual[i]
            temp = cy / step - primal_product[i]
            pdhg = (temp - clamp(temp, -con_ub[i], -con_lb[i])) * step
            is_major && (pdhg_dual[i] = pdhg)
            refl = 2.0 * pdhg - cy
            reflected_dual[i] = refl
            current_dual[i] = weight * (ω * refl + omw * cy) + omwt * initial_dual[i]
        end
    end
    return nothing
end

function dual_step!(s::PDLPState{V}, k_offset::Int, ω::Float64,
                    is_major::Bool) where {V<:CuVector{Float64}}
    step = s.step_size * s.primal_weight
    k = s.inner_count + k_offset
    weight = k / (k + 1.0); omw = 1.0 - ω; omwt = 1.0 - weight
    @cuda threads=256 blocks=cld(s.m, 256) _pdlp_dual_kernel!(
        s.pdhg_dual, s.reflected_dual, s.current_dual, s.initial_dual,
        s.primal_product, s.con_lb, s.con_ub,
        step, weight, ω, omw, omwt, is_major, s.m)
    return nothing
end
