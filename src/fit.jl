function fit(
    Y::Matrix{Float64};
    p::Int = 1,
    z0::Float64 = 1.0,
    q0::Float64 = 1.0,
    v0::Int = size(Y, 2) + |,
    S0::Matrix{Float64} = LA.I(size(Y, 2)),
    warmup::Int = 2000,
    iter::Int = 4000,
    thin::Int = 1,
)
    # Preallocate the chain results
    Z = Y
    T = size(Y, 1)
    N = size(Y, 2)
    neff = (iter - warmup) ÷ thin
    chain_g = [-ones(Bool, N * (N - 1)) for _ in 1:neff]

    # Run the MCMC
    model = Model(; p, N, T, Z, q0, v0, S0, ζ0 = z0)
    for t in 1:iter
        AbstractGSBPs.step!(model)
        if (t > warmup) && ((t - warmup) % thin == 0)
            chain_g[(t - warmup) ÷ thin] .= model.g
        end
    end

    # Return the chains
    df = DF.DataFrame(hcat(chain_g...)' |> collect, :auto)
    return df
end
