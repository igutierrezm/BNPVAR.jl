begin
    using AbstractGSBPs
    using BNPVAR
    using DataFrames
    using RCall
    using Random
    using LinearAlgebra
    using StatsBase
end

R"""
sessionInfo()
# install.packages('vars', repos = "http://cran.rstudio.com/")
# install.packages('tidytext', repos = "http://cran.rstudio.com/")
# install.packages('bruceR', repos = "http://cran.rstudio.com/")
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
    T, N, p = 200, 3, 1
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
for idx in 1:1
    Random.seed!(1)
    nsims = 100
    T, N, p = 200, 3, 2
    warmup = 5000
    neff = 500
    thin = 10
    iter = warmup + neff * thin
    chain_g = [-ones(Bool, N * (N - 1)) for _ in 1:neff]
    scores = [-ones(Int, N * (N - 1)) for _ in 1:nsims]
    for sim in 1:nsims
        println("idx: $idx, sim: $sim")
        y, X, Z = samples[idx][sim]
        model = BNPVAR.DiracSSModel(; p, N, T, Z)
        for t in 1:iter
            AbstractGSBPs.step!(model)
            if (t > warmup) && ((t - warmup) % thin == 0)
                chain_g[(t - warmup) ÷ thin] .= model.g
            end
        end
        # Save results
        filename = "dirac_gamma_$(idx)_$(sim).csv"
        df = DataFrame(hcat(chain_g...)' |> collect, :auto)
        R"""
        readr::write_csv($df, $filename)
        """
        @show idx sim mode(chain_g)
    end
    # # scores
    # filename = "data$(idx)_dirac.csv"
    # df = DataFrame(hcat(scores...)' |> collect, :auto)
    # R"""
    # readr::write_csv($df, $filename)
    # """
end

# Run a Granger causality test on each sample
begin
    for idx in 1:6
        freq_scores = -ones(100)
        for i in 1:100
            y, X, Y = samples[idx][i]
            R"""
            df <- data.frame($Y)
            names(df) <- c("y1", "y2", "y3")
            fit <-
                df |>
                vars::VAR(p = 1, type = "const")
            out <-
                fit |>
                bruceR::granger_causality(
                    var.y = "y1",
                    var.x = "y2"
                )
            # pval <- out$result$p.Chisq
            # out <-
            #     fit |>
            #     vars::causality(
            #         cause = "y1",
            #         boot = TRUE,
            #         boot.runs = 1000
            #     )
            # pval <- out$Granger$p.value
            """
            # @rget pval
            # freq_scores[i] = pval >= 0.05
        end
        # sum(freq_scores) / 100
    end
end

R"""
df <-
    list.files(pattern = "*.csv") |>
    readr::read_csv(id = "filename") |>
    tidyr::unite("gamma", x1:x6, sep = "") |>
    dplyr::mutate(id =  substring(filename, 5, 5) |> as.integer()) |>
    dplyr::count(gamma, .by = id) |>
    dplyr::arrange(.by, dplyr::desc(n))

for (id in 1:6) {
    true_gamma <- rep(0, 6)
    true_gamma[id] <- 1
    true_gamma <- paste0(true_gamma, collapse = "")
    df |>
        dplyr::filter(.by == id) |>
        dplyr::mutate(gamma = reorder(gamma, n)) |>
        ggplot2::ggplot(ggplot2::aes(y = gamma, x = n)) +
        ggplot2::geom_col() +
        ggplot2::labs(
            title = "Distribucion empirica de las hipotesis seleccionadas por el algoritmo",
            subtitle = paste0("usando 100 muestras simuladas con gamma = ", true_gamma)
        )
    ggplot2::ggsave(paste0("plot", id, ".png"))
}
"""