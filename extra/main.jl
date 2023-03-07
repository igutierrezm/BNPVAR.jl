begin
    using AbstractGSBPs
    using BNPVAR
    using CSV
    using RCall
    using Random
    using LinearAlgebra
end

R"""
options(repos = "http://cran.rstudio.com/")
install.packages("vars")
"""


# begin
#     R"""
#     fit <-
#         vars::Canada |>
#         vars::VAR(p = 3, type = "const")
#     # out <-
#     #     fit |>
#     #     bruceR::granger_causality(
#     #         var.y = "prod",
#     #         var.x = "e"
#     #     )
#     # pval <- out$result$p.Chisq
#     out <-
#         fit |>
#         vars::causality(
#             cause = "prod",
#             boot = TRUE
#         )
#     pval <- out$Granger$p.value
#     """;
#     # @rget pval
# end

# Generate 100 samples
function generate_sample(idx)
    T, N, p = 200, 3, 2
    i, j = get_ij_pair(idx, N)
    Z = randn(T, N)
    c = 0.5 * ones(N)
    Σ = Matrix{Float64}(I(N))
    A1 = 0.5 * Matrix{Float64}(I(N))
    A2 = deepcopy(A1)
    A2[j, i] = 0.5
    d = rand(T) .<= 0.3
    for t in 3:T
        if d[t]
            Z[t, :] .= c + A1 * Z[t - 1, :] + cholesky(Σ).L * randn(N)
        else
            Z[t, :] .= c + A2 * Z[t - 1, :] + cholesky(Σ).L * randn(N)
        end
    end
    yvec = [Z[t, :] for t in 2:T]
    Xvec = [kron(I(N), [1 vec(Z[(t-p):(t-1), :]')']) for t in (1 + p):T]
    y = vcat(yvec...)
    X = vcat(Xvec...)
    return y, X, Z
end

begin
    Random.seed!(1)
    samples = [[generate_sample(idx) for _ in 1:100] for idx in 1:6];
end;

# Run our test
for idx in 1:6
    Random.seed!(1)
    nsims = 100
    T, N, p = 200, 3, 2
    warmup = 5000
    neff = 100
    thin = 10
    iter = warmup + neff * thin
    chain_g = [-ones(Bool, N * (N - 1)) for _ in 1:neff]
    scores = [-ones(Int, N * (N - 1)) for _ in 1:nsims]
    # bay_scores = -ones(Int, nsims, 8)
    for sim in 1:nsims
        println(sim)
        y, X, Z = samples[idx][sim]
        model = BNPVAR.Model(; p, N, T, Z)
        for t in 1:iter
            AbstractGSBPs.step!(model)
            if (t > warmup) && ((t - warmup) % thin == 0)
                chain_g[(t - warmup) ÷ thin] .= model.g
            end
        end
        # bay_scores[sim, idx] = findmax(sum(chain_g))[2]
        scores[sim] .= mode(chain_g)
    end
    scores
    CSV.write("data$idx.csv", DataFrame(hcat(scores...)' |> collect, :auto))
end

# Run a Granger causality test on each sample
begin
    freq_scores = -ones(100)
    for i in 1:100
        y, X, Y = samples[i]
        R"""
        df <- data.frame($Y)
        names(df) <- c("y1", "y2")
        fit <-
            df |>
            vars::VAR(p = 2, type = "const")
        # out <-
        #     fit |>
        #     bruceR::granger_causality(
        #         var.y = "y1",
        #         var.x = "y2"
        #     )
        # pval <- out$result$p.Chisq
        out <-
            fit |>
            vars::causality(
                cause = "y1",
                boot = TRUE,
                boot.runs = 1000
            )
        pval <- out$Granger$p.value
        """
        @rget pval
        freq_scores[i] = pval >= 0.05
    end
    sum(freq_scores) / 100
end
