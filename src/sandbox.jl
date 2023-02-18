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
    κ0::Float64
    κ1::Float64
    # Parameters
    β::Vector{Vector{Float64}}
    Σ::Vector{Matrix{Float64}}
    g::Vector{Bool}
    # Skeleton
    skl::AbstractGSBPs.GSBPSkeleton{Vector{Float64}, Matrix{Float64}}
    function Model(; p, N, T, Z,
        S0::Matrix{Float64} = 1.0 * I(N) |> collect,
        β0::Vector{Float64} = zeros(N * (1 + N * p)),
        v0::Int = N + 1,
        κ0::Float64 = 0.01,
        κ1::Float64 = 1.00,
    )
        k = N * (1 + N * p)
        Ω0 = κ1 * I(k) |> collect
        yvec = [Z[t, :] for t in (1 + p):T]
        lags = ones(T - p, 1)
        for j in 1:p
            lags = [lags Z[(1 + p - j):(T - j), :]]
        end
        Xvec = [kron(I(N), lags[t, :]') for t in 1:(T - p)]
        y = vcat(yvec...)
        X = vcat(Xvec...)
        β = [zeros(k)]
        Σ = [deepcopy(S0)]
        g = ones(Bool, k)
        skl = AbstractGSBPs.GSBPSkeleton(; y = yvec, x = Xvec)
        new(p, N, T - p, y, X, yvec, Xvec, Ω0, S0, β0, v0, κ0, κ1, β, Σ, g, skl)
    end
end

function AbstractGSBPs.get_skeleton(model::Model)
    model.skl
end

function AbstractGSBPs.loglikcontrib(model::Model, y0, x0, d0::Int)
    (; β, Σ) = model
    return Distributions.logpdf(Distributions.MvNormal(x0 * β[d0], Σ[d0]), y0)
end

function AbstractGSBPs.step_atoms!(model::Model, K::Int)
    (; y, X, yvec, Xvec, N, T, p, Ω0, S0, β0, v0, κ0, κ1, β, Σ, g) = model
    d = AbstractGSBPs.get_cluster_labels(model)
    k = N * (1 + N * p)
    while length(β) < K
        push!(β, zeros(k))
        push!(Σ, Matrix(1.0 * I(N)))
    end
    for row in 1:k
        Ω0[row, row] = g[row] ? κ1 : κ0
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
end
