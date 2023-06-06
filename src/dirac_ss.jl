struct DiracSSModel <: AbstractGSBPs.AbstractGSBP
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
    # Parameters
    β::Vector{Vector{Float64}}
    Σ::Vector{Matrix{Float64}}
    g::Vector{Bool}
    gdict::Dict{Int, Vector{Int}}
    gaugmented::Vector{Bool}
    # Skeleton
    skl::AbstractGSBPs.GSBPSkeleton{Vector{Float64}, Matrix{Float64}}
    function DiracSSModel(; p, N, T, Z,
        S0::Matrix{Float64} = 1.0 * I(N) |> collect,
        β0::Vector{Float64} = zeros(N * (1 + N * p)),
        v0::Int = N + 1,
        q0::Float64 = 2.0,
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
        new(p, N, T - p, y, X, yvec, Xvec, S0, v0, q0, β, Σ, g, gdict, gaugmented, skl)
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
        i = 1 + (idx ÷ N)
        j = 1 + (idx - 1) % (N - 1)
        j += j >= i
        tmp = [(p * N + 1) * (j - 1) + N * (m - 1) + i + 1 for m in 1:p]
        out[idx] = deepcopy(tmp)
    end
    return out
end

function AbstractGSBPs.get_skeleton(model::DiracSSModel)
    model.skl
end

function AbstractGSBPs.loglikcontrib(model::DiracSSModel, y0, x0, d0::Int)
    (; β, Σ) = model
    return Distributions.logpdf(Distributions.MvNormal(x0 * β[d0], Σ[d0]), y0)
end

function AbstractGSBPs.step_atoms!(model::DiracSSModel, K::Int)
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
    (; g, gdict, gaugmented) = model
    switched_index = rand(1:length(g))
    log_bf = get_log_bf(model, switched_index, K)
    if log(rand()) < log_bf
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
            logdet(q0 * I(length(m1γ_new))) -
            logdet(q0 * I(length(m1γ_old))) +
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
function get_Ak(m::DiracSSModel, cl::Int, k::Int)
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
function get_c(m::DiracSSModel, cl::Int)
    (; N, β) = m
    B = reshape(β[cl], :, N)
    c = B[1, :]
    return c
end

# Generate the "big A" matrix (for cluster cl) using the current state
function get_A(m::DiracSSModel, cl::Int)
    (; N, p) = m
    Aks = [get_Ak(m, cl, k) for k in 1:p]
    A = [hcat(Aks...); I(N * (p - 1)) zeros(N * (p - 1), N)]
    return Matrix{Float64}(A)
end

# Generate the "big A" matrix (from a submodel) using the current state
function get_A(m::SubModel, β::Vector{Float64})
    (; N, p) = m
    Aks = [get_Ak(m, β, k) for k in 1:p]
    A = [hcat(Aks...); I(N * (p - 1)) zeros(N * (p - 1), N)]
    return Matrix{Float64}(A)
end

function get_irf(m::DiracSSModel, max_horizon = 10)
    (; N, β) = m
    nclus = length(β)
    weights = [AbstractGSBPs.gen_mixture_weight(m, h) for h in 1:nclus]
    weighted_Ahs = [weights[h] * get_A(m, h) for h in 1:nclus]
    B = sum(weighted_Ahs)
    F = svd(B)
    irfs = [F.U * Diagonal(F.S .^ horizon) * F.Vt for horizon in 1:max_horizon]
    relevant_irfs = getindex.(irfs, Ref(1:N), Ref(1:N))
    return relevant_irfs
end
