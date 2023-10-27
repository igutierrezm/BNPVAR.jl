function fit(
    Y::Matrix{Float64};
    p::Int = 1,
    z0::Float64 = 1.0,
    q0::Float64 = 1.0,
    v0::Int = size(Y, 2) + 1,
    S0::Matrix{Float64} = LA.I(size(Y, 2)) |> Matrix{Float64},
    warmup::Int = 2000,
    iter::Int = 4000,
    thin::Int = 1,
    hmax::Int = 1,
    grid_npoints::Int = 50,
    grid_lbs::Vector{Float64} = eachcol(Y) .|> minimum,
    grid_ubs::Vector{Float64} = eachcol(Y) .|> maximum,
)
    # Preallocate the chain results
    Z = Y
    T = size(Y, 1)
    N = size(Y, 2)
    neff = (iter - warmup) ÷ thin
    ygrids = range.(grid_lbs, grid_ubs, grid_npoints) .|> collect
    chain_pdf = [[zeros(N, grid_npoints) for _ in 1:hmax] for _ in 1:neff]
    chain_irf = [[zeros(N, N) for _ in 1:hmax] for _ in 1:neff]
    chain_gamma = [-ones(Bool, N * (N - 1)) for _ in 1:neff]

    # Run the MCMC
    model = Model(; p, N, T, Z, q0, v0, S0, ζ0 = z0)
    for t in 1:iter
        AbstractGSBPs.step!(model)
        teff = (t - warmup) ÷ thin
        if (t > warmup) && ((t - warmup) % thin == 0)
            chain_gamma[teff] .= model.g
            irf = get_irf(model, hmax)
            for h in 1:hmax
                chain_irf[teff][h] .= irf[h]
            end
            for k in 1:N, igrid in 1:grid_npoints, h in 1:hmax
                chain_pdf[teff][h][k, igrid] =
                    BNPVAR.get_posterior_predictive_pdf_1d(
                        ygrids[k][igrid], k, model, h
                    )
            end
        end
    end

    # Get a DataFrame with the meaning of each gamma
    df_gamma_dict = DF.DataFrame(
        gamma_id = Int[],
        cause_id = Int[],
        effect_id = Int[]
    )
    for gamma_id in 1:6
        cause_id, effect_id = get_ij_pair(gamma_id, N)
        push!(df_gamma_dict, (gamma_id, cause_id, effect_id))
    end

    # Convert chain_gamma into a DataFrame
    df_chain_gamma = DF.DataFrame(hcat(chain_gamma...)' |> collect, :auto)
    df_chain_gamma[!, :iter] = collect(1:size(df_chain_gamma, 1))
    df_chain_gamma = DF.stack(df_chain_gamma, DF.Not(:iter))
    df_chain_gamma[!, :gamma_id] =
        df_chain_gamma[!, :variable] .|>
        (x) -> strip(x, 'x') .|>
        (x) -> parse(Int, x)
    df_chain_gamma =
        df_chain_gamma |>
        (x) -> DF.innerjoin(x, df_gamma_dict, on = :gamma_id) |>
        (x) -> DF.select!(x, :iter, :gamma_id, :cause_id, :effect_id, :value)

    # Convert chain_irf into a DataFrame
    df_chain_irf =
        map(1:length(chain_irf)) do iter
            vec_irf = vcat(vec.(chain_irf[iter])...)
            ncells = length(vec_irf)
            df = DF.DataFrame(value = vec_irf)
            df[!, :horizon] = 1 .+ (0:ncells - 1) .÷ N^2
            df[!, :effect_id] = 1 .+ (0:ncells - 1) .% N
            df[!, :cause_id] = 1 .+ ((0:ncells - 1) .÷ N) .% N
            df[!, :iter] .= iter
            df
        end |>
        (x) -> reduce(vcat, x) |>
        (x) -> DF.select!(x, :iter, :horizon, :cause_id, :effect_id, :value)

    # Convert chain_pdf into a DataFrame
    begin
        df_chain_pdf = DF.DataFrame(
            iter = Int[],
            horizon = Int[],
            var_id = Int[],
            y = Float64[],
            value = Float64[]
        )
        for iter in 1:neff, var_id in 1:N
            for horizon in 1:hmax, igrid in 1:grid_npoints
                f0 = chain_pdf[iter][horizon][var_id, igrid]
                ith_row = (iter, horizon, var_id, ygrids[var_id][igrid], f0)
                push!(df_chain_pdf, ith_row)
            end
        end
    end

    # Return the chains as DataFrames
    out =
        Dict(
            "gamma" => df_chain_gamma,
            "irf" => df_chain_irf,
            "pdf" => df_chain_pdf,
        )
    return out
end
