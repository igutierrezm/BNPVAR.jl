struct Model <: AbstractGSBPs.AbstractGSBP
    # Data
    p::Int
    N::Int
    T::Int
    y::Vector{Float64}
    X::Matrix{Float64}
    yvec::Vector{Vector{Float64}}
    Xvec::Vector{Matrix{Float64}}
    # Hyperparameters
    Ω0::Matrix{Float64}
    S0::Matrix{Float64}
    β0::Vector{Float64}
    v0::Int
    q0::Float64
    c0::Float64
    # Parameters
    β::Vector{Vector{Float64}}
    Σ::Vector{Matrix{Float64}}
    g::Vector{Bool}
    ψ::Vector{Float64}
    b::Vector{Float64}
    gdict::Dict{Int, Vector{Int}}
    inv_gvec::Vector{Int}
    # Skeleton
    skl::AbstractGSBPs.GSBPSkeleton{Vector{Float64}, Matrix{Float64}}
    function Model(; p, N, T, Z,
        S0::Matrix{Float64} = 1.0 * I(N) |> collect,
        β0::Vector{Float64} = zeros(N * (1 + N * p)),
        v0::Int = N + 1,
        q0::Float64 = 0.01,
        c0::Float64 = 10.0,
        g::Vector{Bool} = ones(Bool, N * (N - 1)),
        gdict::Dict{Int, Vector{Int}} = init_gdict(N, p)
    )
        k = N * (1 + N * p)
        yvec = [Z[t, :] for t in (1 + p):T]
        lags = ones(T - p, 1)
        for j in 1:p
            lags = [lags Z[(1 + p - j):(T - j), :]]
        end
        Xvec = [kron(I(N), lags[t, :]') for t in 1:(T - p)]
        y = vcat(yvec...)
        X = vcat(Xvec...)
        Σ = [deepcopy(S0)]
        β = [zeros(k)]
        ψ = ones(k)
        b = ones(k)
        Ω0 = Matrix{Float64}(I(k))
        for r in 1:k
            Ω0[r, r] = ψ[r]
        end
        for u in eachindex(g)
            for r in gdict[u]
                Ω0[r, r] = g[u] ? ψ[r] : q0 * ψ[r]
            end
        end
        inv_gvec = init_inv_gvec(N, p, gdict)
        skl = AbstractGSBPs.GSBPSkeleton(; y = yvec, x = Xvec)
        new(p, N, T - p, y, X, yvec, Xvec, Ω0, S0, β0, v0, q0, c0, β, Σ, g, ψ, b, gdict, inv_gvec, skl)
    end
end

function get_ij_pair(idx, N)
    i = 1 + (idx ÷ N)
    j = 1 + (idx - 1) % (N - 1)
    j += j >= i
    (i, j)
end

function init_gdict(N, p)
    out = Dict{Int, Vector{Int}}()
    for idx in 1:(N * (N - 1))
        i = 1 + (idx ÷ N)
        j = 1 + (idx - 1) % (N - 1)
        j += j >= i
        tmp = [(p * N + 1) * (j - 1) + N * (m - 1) + i + 1 for m in 1:p]
        out[idx] = deepcopy(tmp)
    end
    return out
end

function init_inv_gvec(N, p, gdict)
    k = N * (1 + N * p)
    out = zeros(Int, k)
    for (key, value) in gdict
        out[value] .= key
    end
    return out
end

function AbstractGSBPs.get_skeleton(model::Model)
    model.skl
end

function AbstractGSBPs.loglikcontrib(model::Model, y0, x0, d0::Int)
    (; β, Σ) = model
    return Distributions.logpdf(Distributions.MvNormal(x0 * β[d0], Σ[d0]), y0)
end

function AbstractGSBPs.step_atoms!(model::Model, K::Int)
    (; y, X, yvec, Xvec, N, T, p, Ω0, S0, β0, v0, β, Σ) = model
    d = AbstractGSBPs.get_cluster_labels(model)
    k = N * (1 + N * p)
    while length(β) < K
        push!(β, zeros(k))
        push!(Σ, Matrix(1.0 * I(N)))
    end
    submodel = BayesVAR.Model(; N, p, Ω0, S0, β0, v0)
    idx = zeros(Bool, N * T)
    for k in 1:K
        row = 1
        for t = 1:T
            for j = 1:N
                idx[row] = d[t] == k
                row += 1
            end
        end
        yk = y[idx, :]
        Xk = X[idx, :]
        yveck = yvec[d .== k]
        Xveck = Xvec[d .== k]
        BayesVAR.update_β!(submodel, yk, Xk, β[k], Σ[k])
        BayesVAR.update_Σ!(submodel, yveck, Xveck, β[k], Σ[k])
    end
    update_b!(model)
    update_ψ!(model)
    update_g!(model, K)
    return nothing
end

function update_g!(model, K)
    (; Ω0, q0, ψ, g, gdict) = model
    idx_star = rand(1:length(g))
    g0 = deepcopy(g)
    g1 = deepcopy(g)
    g1[idx_star] = !g0[idx_star]
    log_acceptance_rate = log_B(model, g0, g1, K)
    if log(rand()) < log_acceptance_rate
        g[idx_star] = !g[idx_star]
        for subidx in gdict[idx_star]
            Ω0[subidx, subidx] = g[idx_star] ? ψ[subidx] : q0 * ψ[subidx]
        end
    end
    return nothing
end

function log_B(model, g0, g1, K)
    (; β0, q0, ψ, β, gdict) = model
    log_num = 0.0
    log_den = 0.0
    for idx in eachindex(g0)
        g0[idx] == g1[idx] && continue
        for subidx in gdict[idx]
            ok0 = g0[idx] ? √(ψ[subidx]) : √(q0 * ψ[subidx])
            ok1 = g1[idx] ? √(ψ[subidx]) : √(q0 * ψ[subidx])
            for cluster in 1:K
                log_num += Distributions.logpdf(
                    Distributions.Normal(β0[subidx], ok1),
                    β[cluster][subidx]
                )
                log_den += Distributions.logpdf(
                    Distributions.Normal(β0[subidx], ok0),
                    β[cluster][subidx]
                )
            end
        end
    end
    return log_num - log_den
end

function update_b!(model)
    (; c0, b, ψ) = model
    for idx in eachindex(b)
        rate = 1.0 / c0 + 1 / ψ[idx]
        dist = Gamma(1, 1 / rate)
        b[idx] = 1 / rand(dist)
    end
    return nothing
end

function update_ψ!(model)
    (; q0, b, ψ, g, inv_gvec) = model
    for idx in eachindex(ψ)
        g0 = inv_gvec[idx]
        rate = b[idx] + 0.5 * b[idx] / (g0 + q0 * (1 - g0))
        dist = Gamma(1, 1 / rate)
        ψ[idx] = 1 / rand(dist)
    end
    return nothing
end



# # All what follows is related to Rosenthal et al. (2022)

# function propose_addition(model, K)
#     (; g) = model
#     k = length(g)
#     g0 = deepcopy(g)
#     g1 = deepcopy(g)
#     lb_a = 1 / k
#     ub_a = k
#     idx_star = 0
#     log_w_star = -Inf
#     for idx in 1:k
#         g1[idx] && continue
#         g1[idx] = true
#         log_w = log_B(model, g0, g1, K)
#         log_w = min(log_w, log(ub_a))
#         log_w = max(log_w, log(lb_a))
#         if log_w > log_w_star
#             log_w_star = log_w
#             idx_star = idx
#         end
#         g1[idx] = false
#     end
#     return idx_star
# end

# function propose_deletion(model, K)
#     (; g) = model
#     k = length(g)
#     g0 = deepcopy(g)
#     g1 = deepcopy(g)
#     lb_a = 1 / k
#     ub_a = k
#     idx_star = 0
#     log_w_star = -Inf
#     for idx in 1:k
#         !g1[idx] && continue
#         g1[idx] = false
#         log_w = log_B(model, g0, g1, K)
#         log_w = min(log_w, log(ub_a))
#         log_w = max(log_w, log(lb_a))
#         if log_w > log_w_star
#             log_w_star = log_w
#             idx_star = idx
#         end
#         g1[idx] = true
#     end
#     return idx_star
# end

# function propose_swap(model, K)
#     (; g) = model
#     idx_star1 = 0
#     idx_star2 = 0
#     if sum(g) < length(g)
#         idx_star1 = propose_addition(model, K)
#         idx_star2 = propose_deletion(model, K)
#     else
#         idx_star1 = propose_deletion(model, K)
#         idx_star2 = propose_addition(model, K)
#     end
#     return idx_star1, idx_star2
# end

# function log_kernel(model, g0, g1, K)
# #     k = length(g0)
# #     h_a = 0.0
# #     h_d = 0.0
# #     if sum(g0) == 0
# #         h_a = 1.0
# #         h_d = 0.0
# #     elseif sum(g0) == k
# #         h_a = 0.0
# #         h_d = 0.1
# #     else
# #         h_a = 0.5
# #         h_d = 0.5
# #     end
# #     # h_s = sum(g0) < length(g0) ? 0.2 : 0.5
# #     # w_s = log_B(model, g0, g1, K) |> x -> min(x, k^2) |> x -> max(x, k * s0)
# #     w_a = log_B(model, g0, g1, K) |> x -> min(x, k^2) |> x -> max(x, 1 / k^2)
# #     w_d = log_B(model, g0, g1, K) |> x -> min(x, k^1) |> x -> max(x, 1 / k^2)
# end

# function foo(idx, N)
#     i = 1 + (idx ÷ N)
#     j = 1 + (idx - 1) % (N - 1)
#     j += j >= i
#     (i, j)
# end

# [foo(idx, 4) for idx in 1:6]
