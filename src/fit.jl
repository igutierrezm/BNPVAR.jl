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
    hmax::Int = 1,
)
    # Preallocate the chain results
    Z = Y
    T = size(Y, 1)
    N = size(Y, 2)
    neff = (iter - warmup) ÷ thin
    chain_gamma = [-ones(Bool, N * (N - 1)) for _ in 1:neff]
    chain_irf = [[zeros(N, N) for _ in 1:hmax] for _ in 1:neff]

    # Run the MCMC
    model = Model(; p, N, T, Z, q0, v0, S0, ζ0 = z0)
    for t in 1:iter
        AbstractGSBPs.step!(model)
        if (t > warmup) && ((t - warmup) % thin == 0)
            chain_g[(t - warmup) ÷ thin] .= model.g
            irf = get_irf(model, hmax)
            for h in 1:hmax
                chain_irf[(t - warmup) ÷ thin][h] .= irf[h]
            end
        end
    end

    # Convert chain_gamma into a DataFrame
    df_gamma = DF.DataFrame(hcat(chain_gamma...)' |> collect, :auto)

    # Convert chain_irf into a DataFrame
    df_chain_irf =
        map(1:length(chain_irf)) do iter
            vec_irf = vcat(vec.(chain_irf[iter])...)
            ncells = length(vec_irf)
            df = DataFrame(irf = vec_irf)
            df[!, :horizon] = 1 .+ (0:ncells - 1) .÷ N^2
            df[!, :effect_id] = 1 .+ (0:ncells - 1) .% N
            df[!, :cause_id] = 1 .+ ((0:ncells - 1) .÷ N) .% N
            df[!, :iter] .= iter
            df
        end |>
        (x) -> reduce(vcat, x)

    # Return the chains as DataFrames
    return df_g
end
