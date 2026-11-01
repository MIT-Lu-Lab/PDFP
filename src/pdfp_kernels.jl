# Fused CUDA kernels for the PDFP step: one launch per update of x, y and z
# instead of several broadcasts. Selected by dispatch on CuVector states.

function _pdfp_primal_kernel!(pdhg_x, dual_slack, reflected_x, current_x, initial_x,
                              current_y, Atz, var_lb, var_ub, τ, weight, ω, omw, omwt, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            cx = current_x[i]
            temp = cx - τ * (Atz[i] - current_y[i])
            pdhg = clamp(temp, var_lb[i], var_ub[i])
            pdhg_x[i] = pdhg
            dual_slack[i] = (pdhg - temp) / τ
            refl = 2.0 * pdhg - cx
            reflected_x[i] = refl
            current_x[i] = weight * (ω * refl + omw * cx) + omwt * initial_x[i]
        end
    end
    return nothing
end

function primal_step!(s::PDFPState{V}, k_offset::Int, ω::Float64) where {V<:CuVector{Float64}}
    τ = s.step_size / s.primal_weight
    k = s.inner_count + k_offset
    weight = k / (k + 1.0); omw = 1.0 - ω; omwt = 1.0 - weight
    mul!(s.Atz, s.A_t, s.current_z)
    @cuda threads=256 blocks=cld(s.n, 256) _pdfp_primal_kernel!(
        s.pdhg_x, s.dual_slack, s.reflected_x, s.current_x, s.initial_x,
        s.current_y, s.Atz, s.var_lb, s.var_ub, τ, weight, ω, omw, omwt, s.n)
    return nothing
end

function _pdfp_dual_y_kernel!(pdhg_y, current_y, reflected_y, initial_y, x_tilde, reflected_x,
                              is_int, w, σ, weight, ω, omw, omwt, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            cy = current_y[i]
            wi = w[i]
            y = is_int[i] ? clamp(cy + σ * (x_tilde[i] - reflected_x[i]), -wi, wi) : 0.0
            pdhg_y[i] = y
            refl = 2.0 * y - cy
            reflected_y[i] = refl
            current_y[i] = weight * (ω * refl + omw * cy) + omwt * initial_y[i]
        end
    end
    return nothing
end

function dual_y_step!(s::PDFPState{V}, k_offset::Int, ω::Float64) where {V<:CuVector{Float64}}
    σ = s.step_size * s.primal_weight
    k = s.inner_count + k_offset
    weight = k / (k + 1.0); omw = 1.0 - ω; omwt = 1.0 - weight
    @cuda threads=256 blocks=cld(s.n, 256) _pdfp_dual_y_kernel!(
        s.pdhg_y, s.current_y, s.reflected_y, s.initial_y, s.x_tilde, s.reflected_x,
        s.is_int, s.w, σ, weight, ω, omw, omwt, s.n)
    return nothing
end

function _pdfp_dual_z_kernel!(pdhg_z, current_z, reflected_z, initial_z, Ax, b, is_eq,
                              σ, weight, ω, omw, omwt, m)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= m
        @inbounds begin
            cz = current_z[i]
            raw = cz + σ * (Ax[i] - b[i])
            z = is_eq[i] ? raw : max(raw, 0.0)
            pdhg_z[i] = z
            refl = 2.0 * z - cz
            reflected_z[i] = refl
            current_z[i] = weight * (ω * refl + omw * cz) + omwt * initial_z[i]
        end
    end
    return nothing
end

function dual_z_step!(s::PDFPState{V}, k_offset::Int, ω::Float64) where {V<:CuVector{Float64}}
    σ = s.step_size * s.primal_weight
    k = s.inner_count + k_offset
    weight = k / (k + 1.0); omw = 1.0 - ω; omwt = 1.0 - weight
    mul!(s.Ax, s.A, s.reflected_x)
    @cuda threads=256 blocks=cld(s.m, 256) _pdfp_dual_z_kernel!(
        s.pdhg_z, s.current_z, s.reflected_z, s.initial_z, s.Ax, s.b, s.is_eq,
        σ, weight, ω, omw, omwt, s.m)
    return nothing
end
