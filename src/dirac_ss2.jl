struct DiracSSModel2 <: AbstractGSBPs.AbstractGSBP
    # Data
    p::Int
    N::Int
    T::Int
    Y::Matrix{Float64}
    # Transformed data
    M::Int
    gdict::Vector{Vector{Bool}}
    yvec::Vector{Vector{Float64}}
    Xvec::Vector{Matrix{Float64}}
    y::Vector{Float64}
    X::Matrix{Float64}
    # Hyperparameters
    S0::Matrix{Float64}
    v0::Float64
    o0::Float64
    # Parameters
    A::Vector{Vector{Matrix{Float64}}}
    c::Vector{Vector{Float64}}
    Σ::Vector{Matrix{Float64}}
    Γ::Matrix{Float64}
    # Transformed Parameters
    β::Vector{Vector{Float64}}
    g::Vector{Bool}
    ψ::Vector{Bool}
    # Skeleton
    skl::AbstractGSBPs.GSBPSkeleton{Vector{Float64}, Matrix{Float64}}
    function DiracSSModel2(;
        # Data
        p::Int,
        N::Int,
        T::Int,
        K::Int,
        Y::Matrix{Float64},
        # Hyperparameters
        S0::Matrix{Float64} = init_S0(N),
        v0::Float64 = init_v0(N),
        o0::Float64 = init_o0(),
        # Parameters
        A::Vector{Vector{Matrix{Float64}}} = init_A(N, K, p),
        c::Vector{Vector{Float64}} = init_c(N, K),
        Σ::Vector{Matrix{Float64}} = init_Σ(N, K),
        Γ::Matrix{Bool} = init_Γ(N),
    )
        # Transformed data
        M = gen_M(N, p)
        gdict = gen_gdict(N, p)
        yvec = gen_yvec(Y, T, p)
        Xvec = gen_Xvec(Y, N, T, p)
        y = vcat(yvec...)
        X = vcat(Xvec...)
        # Transformed parameters
        β = gen_β(c, A, K)
        g = gen_g(Γ)
        ψ = gen_ψ(g, gdict, N, M)
        skl = AbstractGSBPs.GSBPSkeleton(; y = yvec, x = Xvec)
        new(
            p, N, T, Y, M, gdict, yvec, Xvec, y, X,
            S0, v0, o0, A, c, Σ, Γ, β, g, ψ, skl
        )
    end
end

function init_S0(N)
   1.0 * I(N) |> collect
end

function init_v0(N)
    N + 1.0
end

function init_o0()
    2.0
end

function gen_M(N, p)
    N * (1 + N * p)
end

function init_Γ(N)
    ones(Bool, N, N)
end

function init_A(N, K, p)
    map(1:K) do _
        map(1:p) do _
            Uniform(-1, 1) |>
                x -> rand(x, N) |>
                Diagonal |>
                Matrix{Float64}
        end
    end
end

function init_c(N, K)
    map(1:K) do _
        Uniform(-2, 2) |>
            x -> rand(x, N)
    end
end

function init_Σ(N, K)
    map(1:K) do _
        Uniform(-2, 2) |>
            x -> rand(x, N) |>
            x -> exp.(x) |>
            Diagonal |>
            Matrix{Float64}
    end
end

function gen_gdict(N, p)
    c1 = zeros(N)
    A1 = zeros(N, N); A1[:] = 1:N^2
    β1 = [c1 kron(ones(1, p), A1)]' |> vec
    [Bool.(β1 .== idx) for idx in 1:N^2]
end

function gen_yvec(Y, T, p)
    [Y[t, :] for t in (1 + p):T]
end

function gen_Z(Y, T, p)
    Z = ones(T - p, 1)
    for m in 1:p
        Z = [Z Y[(1 + p - m):(T - m), :]]
    end
    return Z
end

function gen_Xvec(Y, N, T, p)
    Z = gen_Z(Y, T, p)
    [kron(I(N), Z[t, :]') for t in 1:(T - p)]
end

function gen_y(yvec)
    vcat(yvec...)
end

function gen_X(Xvec)
    vcat(Xvec...)
end

function gen_B(c, A, K)
    map(1:K) do k
        [c[k] hcat(A[k]...)]' |>
            Matrix{Float64}
    end
end

function gen_β(c, A, K)
    B = gen_B(c, A, K)
    map(1:K) do k
        vec(B[k])
    end
end

function gen_g(Γ)
    vec(Γ)
end

function gen_ψ(g, gdict, N, M)
    ψ = ones(Bool, M)
    for idx in 1:N^2
        ψ[gdict[idx]] .= g[idx]
    end
    return ψ
end
