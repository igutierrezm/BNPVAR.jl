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
    S0::Matrix{Float64}
    v0::Int
    q0::Float64
    ζ0::Float64
    # Parameters
    β::Vector{Vector{Float64}}
    Σ::Vector{Matrix{Float64}}
    g::Vector{Bool}
    gdict::Dict{Int, Vector{Int}}
    gaugmented::Vector{Bool}
    # Skeleton
    skl::AbstractGSBPs.GSBPSkeleton{Vector{Float64}, Matrix{Float64}}
    function Model(; p, N, T, Z,
        S0::Matrix{Float64} = 1.0 * I(N) |> collect,
        v0::Int = N + 1,
        q0::Float64 = 1.0,
        ζ0::Float64 = 1.0,
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
        gaugmented = ones(Bool, k)
        for (key, value) in gdict
            gaugmented[value] .= g[key]
        end
        skl = AbstractGSBPs.GSBPSkeleton(; y = yvec, x = Xvec)
        new(p, N, T - p, y, X, yvec, Xvec, S0, v0, q0, ζ0, β, Σ, g, gdict, gaugmented, skl)
    end
end

function get_ij_pair(idx, N)
    i = ceil(Int, idx / (N - 1))
    j = 1 + (idx - 1) % (N - 1)
    j += j >= i
    (i, j)
end

function init_gdict(N, p)
    out = Dict{Int, Vector{Int}}()
    for idx in 1:(N * (N - 1))
        # i = 1 + (idx ÷ N)
        # j = 1 + (idx - 1) % (N - 1)
        # j += j >= i
        cause, effect = get_ij_pair(idx, N)
        tmp = [(p * N + 1) * (effect - 1) + 1 + cause + N * (m - 1) for m in 1:p]
        out[idx] = deepcopy(tmp)
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
    (; y, X, yvec, Xvec, N, T, p, S0, v0, q0, β, Σ, gaugmented) = model
    d = AbstractGSBPs.get_cluster_labels(model)
    k = N * (1 + N * p)
    while length(β) < K
        push!(β, zeros(k))
        push!(Σ, Matrix(2.0 * I(N)))
    end
    submodel = SubModel(; N, p, S0, v0, q0)
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
        Xveck = Xvec[d .== k, :]
        update_β!(submodel, yk, Xk, β[k], Σ[k], gaugmented)
        update_Σ!(submodel, yveck, Xveck, β[k], Σ[k])
    end
    update_g!(model, K)
    return nothing
end

struct SubModel
    # Dimensions
    N::Int
    p::Int
    # Hyperparameters
    S0::Matrix{Float64}
    v0::Int
    q0::Float64
    function SubModel(;
        N::Int,
        p::Int,
        S0::Matrix{Float64} = 2.0 * I(N) |> collect,
        v0::Int = N + 1,
        q0::Float64 = 2.0,
    )
        new(N, p, S0, v0, q0)
    end
end

# Según Chan
function update_β!(model::SubModel, y, X, β, Σ, γ)
    (; N, q0) = model
    T = length(y) ÷ N
    nγ = sum(γ)
    Xγ = X[:, γ]
    Ω0γ = q0 * I(nγ)
    chol_inv_Ω1γ = cholesky(Symmetric(inv(Ω0γ) + Xγ' * kron(I(T), inv(Σ)) * Xγ))
    m1γ = chol_inv_Ω1γ \ (Xγ' * kron(I(T), inv(Σ)) * y)
    newβγ = randn(nγ)
    ldiv!(chol_inv_Ω1γ.U, newβγ)
    newβγ .+= m1γ
    β .= 0.0
    β[γ] .= newβγ
end

# Según Chan
function update_Σ!(model::SubModel, yvec, Xvec, β, Σ)
    (; v0, S0) = model
    S1 = cholesky(S0)
    for t in eachindex(yvec)
        lowrankupdate!(S1, yvec[t] - Xvec[t] * β)
    end
    v1 = v0 + length(yvec)
    dist = InverseWishart(v1, Matrix(S1))
    rand!(dist, Σ)
end

function update_g!(model, K)
    (; N, g, ζ0, gdict, gaugmented) = model
    switched_index = rand(1:length(g))
    log_bf = get_log_bf(model, switched_index, K)
    pγ0 = ph0(N * (N - 1), ζ0)
    sumg_old = sum(g)
    sumg_new = sumg_old + 1 - 2 * g[switched_index]
    log_prior_odds = log(pγ0[sumg_new]) - log(pγ0[sumg_old])
    if log(rand()) < log_prior_odds + log_bf
        g[switched_index] = !g[switched_index]
    end
    for (key, value) in gdict
        gaugmented[value] .= g[key]
    end
    return nothing
end

function get_log_bf(model, switched_index, K)
    (; T, N, p, y, X, S0, v0, q0, Σ, g, gdict, gaugmented) = model
    d = AbstractGSBPs.get_cluster_labels(model)
    submodel = SubModel(; N, p, S0, v0, q0)
    gaugmented_old = deepcopy(gaugmented)
    gaugmented_new = deepcopy(gaugmented)
    gaugmented_new[gdict[switched_index]] .= 1 .- g[switched_index]
    idx = zeros(Bool, N * T)
    out = 0.0
    for cluster in 1:K
        row = 1
        for t = 1:T
            for j = 1:N
                idx[row] = d[t] == cluster
                row += 1
            end
        end
        yk = y[idx, :]
        Xk = X[idx, :]
        Λ1γU_old, m1γ_old = (
            get_Λ1γU_and_m1γ!(submodel, yk, Xk, Σ[cluster], gaugmented_old)
        )
        Λ1γU_new, m1γ_new = (
            get_Λ1γU_and_m1γ!(submodel, yk, Xk, Σ[cluster], gaugmented_new)
        )
        out += (
            length(m1γ_old) * log(q0) -
            length(m1γ_new) * log(q0) +
            logdet(Λ1γU_old) -
            logdet(Λ1γU_new) +
            norm(Λ1γU_new * m1γ_new, 2)^2 / 2 -
            norm(Λ1γU_old * m1γ_old, 2)^2 / 2
        )
    end
    return out
end

# Según Chan
function get_Λ1γU_and_m1γ!(model::SubModel, y, X, Σ, γ)
    (; N, q0) = model
    T = length(y) ÷ N
    nγ = sum(γ)
    Xγ = X[:, γ]
    Ω0γ = q0 * I(nγ)
    chol_inv_Ω1γ = cholesky(Symmetric(inv(Ω0γ) + Xγ' * kron(I(T), inv(Σ)) * Xγ))
    m1γ = chol_inv_Ω1γ \ (Xγ' * kron(I(T), inv(Σ)) * y)
    return chol_inv_Ω1γ.U, m1γ
end

# Extract Ak (in cluster cl) from the model current state
function get_Ak(m::Model, cl::Int, k::Int)
    (; N, β) = m
    B = reshape(β[cl], :, N)
    Ak = B[2 + N * (k - 1):(1 + N * k), :]' |> collect
    return Ak
end

# Extract Ak (in cluster cl) from a submodel
function get_Ak(m::SubModel, β::Vector{Float64}, k::Int)
    (; N) = m
    B = reshape(β, :, N)
    Ak = B[2 + N * (k - 1):(1 + N * k), :]' |> collect
    return Ak
end

# Extract c (in cluster cl) from the model current state
function get_c(m::Model, cl::Int)
    (; N, β) = m
    B = reshape(β[cl], :, N)
    c = B[1, :]
    return c
end

# Generate the "big C" matrix (for cluster cl) using the current state
function get_C(m::Model, cl::Int)
    (; N, p) = m
    Aks = [get_Ak(m, cl, k) for k in 1:p]
    C = [hcat(Aks...); I(N * (p - 1)) zeros(N * (p - 1), N)]
    return Matrix{Float64}(C)
end

# Generate the "big C" matrix (from a submodel) using the current state
function get_C(m::SubModel, β::Vector{Float64})
    (; N, p) = m
    Aks = [get_Ak(m, β, k) for k in 1:p]
    C = [hcat(Aks...); I(N * (p - 1)) zeros(N * (p - 1), N)]
    return Matrix{Float64}(C)
end

"""
    Return an offset vector `p`, where p[i] is the prior
    probability of any hypothesis such that sum_{j=1}^m γj = i,
    according to Womack (2015)'s proposal
"""
function ph0(m, ζ)
    p = OffsetArray([zeros(m); 1.0], 0:m)
    for l = m-1:-1:0
        for j = 1:m-l
            p[l] += ζ * p[l + j] * binomial(l + j, l)
        end
    end
    p /= sum(p)
    for l = 0:m-1
        p[l] /= binomial(m, l)
    end
    return p
end

# Return [D[τ - 1] for τ in 1:length(dpath)]
function get_all_Ds(m::Model, dpath::Vector{Int})
    (; β) = m
    nclus = length(β)
    hmax = length(dpath)
    C = [get_C(m, clus) for clus in 1:nclus]
    D = [C[dpath[hmax]]]
    for τ in 1:(hmax - 1)
        push!(D, D[end] * C[dpath[hmax - τ]])
    end
    return D
end

function get_irf(m::Model, hmax = 10)
    (; N, β) = m
    dpath = zeros(Int, hmax)
    nclus = length(β)
    for h in 1:hmax
        rnew = AG.rand_rnew(m)
        dnew = AG.rand_dnew(m, rnew)
        dpath[h] = min(dnew, nclus)
    end
    Ds = get_all_Ds(m, dpath)
    out = [Ds[h][1:N, 1:N] for h in 1:hmax]
    return out
end

function get_posterior_predictive_moments(m::Model, dpath::Vector{Int})
    (; p, N, Σ, yvec) = m
    hmax = length(dpath)
    Ds = get_all_Ds(m, dpath)
    zend = vcat([yvec[end + 1 - t] for t in 1:p]...)
    Phi = [Ds[τ][1:N, 1:N] for τ in 1:hmax]
    mhmax = Ds[hmax][1:N, :] * zend + get_c(m, dpath[hmax])
    Vhmax = deepcopy(Σ[dpath[hmax]])
    for τ = 1:(hmax - 1)
        mhmax .+= Phi[τ] * get_c(m, dpath[hmax - τ])
        Vhmax .= Vhmax .+ (Phi[τ] * Σ[dpath[hmax - τ]] * Phi[τ]')
    end
    return mhmax, LA.Symmetric(Vhmax)
end

function get_posterior_predictive_pdf_1d(
        y::Float64,
        k::Int,
        m::Model,
        dpath::Vector{Int}
    )
    mh, Vh = get_posterior_predictive_moments(m, dpath)
    return DT.pdf(DT.Normal(mh[k], Vh[k, k]), y)
end

function get_posterior_predictive_pdf_1d(
    y::Float64,
    k::Int,
    m::Model,
    hmax::Int
)
    (; β) = m
    dpath = zeros(Int, hmax)
    nclus = length(β)
    for h in 1:hmax
        rnew = AG.rand_rnew(m)
        dnew = AG.rand_dnew(m, rnew)
        dpath[h] = min(dnew, nclus)
    end
    return get_posterior_predictive_pdf_1d(y, k, m, dpath)
end
