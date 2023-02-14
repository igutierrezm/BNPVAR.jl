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
    # Parameters
    β::Vector{Vector{Float64}}
    Σ::Vector{Matrix{Float64}}
    # Skeleton
    skl::AbstractGSBPs.GSBPSkeleton{Vector{Float64}, Matrix{Float64}}
    function Model(; p, N, T, Z,
        Ω0::Matrix{Float64} = 100.0 * I(N * (1 + N * p)) |> collect,
        S0::Matrix{Float64} = 10.0 * I(N) |> collect,
        β0::Vector{Float64} = zeros(N * (1 + N * p)),
        v0::Int = N + 1
    )
        yvec = [Z[t, :] for t in (1 + p):T]
        lags = ones(T - p, 1)
        for j in 1:p
            lags = [lags Z[(1 + p - j):(T - j), :]]
        end
        Xvec = [kron(I(N), lags[t, :]') for t in 1:(T - p)]
        y = vcat(yvec...)
        X = vcat(Xvec...)
        β = [zeros(N * (1 + N * p))]
        Σ = [deepcopy(S0)]
        skl = AbstractGSBPs.GSBPSkeleton(; y = yvec, x = Xvec)
        new(p, N, T - p, y, X, yvec, Xvec, Ω0, S0, β0, v0, β, Σ, skl)
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
end
