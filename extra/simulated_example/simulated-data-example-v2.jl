begin
    using AbstractGSBPs
    using BNPVAR
    using DataFrames
    using RCall
    using Random
    using LinearAlgebra
    using StatsBase
end

# R"""
# sessionInfo()
# install.packages('vars', repos = "http://cran.rstudio.com/")
# install.packages('tidytext', repos = "http://cran.rstudio.com/")
# install.packages('bruceR', repos = "http://cran.rstudio.com/")
# """

# Simulate a sample, as described in the main article
function generate_sample()
    T, N, p = 200, 3, 1
    Z = randn(T, N)
    c = 0.5 * ones(N)
    Σ = 0.5 * Matrix{Float64}(I(N))
    A1 = 0.5 * Matrix{Float64}(I(N))
    A2 = deepcopy(A1)
    A1[1, 2] = -0.5
    A2[1, 2] = +0.5
    A1[3, 1] = -0.5
    A2[3, 1] = +0.5
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

# Generate 100 samples
begin
    Random.seed!(1)
    samples = [generate_sample() for _ in 1:100]
end;

# Run a Granger causality test on each sample
begin
    df = DataFrame(i = Int[], j = Int[], p = Int[], rejected = Float64[])
    for cause in 1:3, effect in 1:3, p in 1:2, idx in 1:100
        y, X, Y = samples[idx]
        R"""
        sink("tmp.txt")
        df <- data.frame($Y)
        names(df) <- c("y1", "y2", "y3")
        fit <-
            df |>
            vars::VAR(p = $p, type = "const")
        out <-
            fit |>
            bruceR::granger_causality(
                var.y = paste0("y", $effect),
                var.x = paste0("y", $cause)
            )
        pval <- out$result$p.Chisq
        sink()
        """
        @rget pval
        push!(df, (cause, effect, p, pval <= 0.05))
    end
end

# Plot the results of the frequentist test
R"""
fig <-
    $df |>
    dplyr::summarize(
        freq = sum(rejected),
        .by = c(i, j, p)
    ) |>
    dplyr::mutate(
        i = paste0("y", i),
        j = paste0("y", j)
    ) |>
    dplyr::filter(i != j) |>
    dplyr::rename(`# lags` = p) |>
    ggplot2::ggplot(
        ggplot2::aes(
            x = i,
            y = j,
            fill = freq,
            label = freq
        )
    ) +
    ggplot2::geom_tile(stat = "identity") +
    ggplot2::geom_text(
        ggplot2::aes(color = ifelse(freq > 50, "1", "2"))
    ) +
    ggplot2::facet_wrap(
        ggplot2::vars(`# lags`),
        labeller = "label_both",
        ncol = 2
    ) +
    ggplot2::scale_fill_distiller(
        type = "seq",
        direction = 1,
        palette = "Greys"
    ) +
    ggplot2::scale_colour_manual(
        values = c("white", "black")
    ) +
    ggplot2::theme_classic() +
    ggplot2::guides(color = FALSE) +
    ggplot2::theme(
        legend.position = 'top',
        legend.justification = 'left',
        legend.direction = 'horizontal'
    ) +
    ggplot2::labs(
        x = "cause",
        y = "effect",
        fill = "frequency\n"
    )
    fig |>
    ggplot2::ggsave(
        filename = "extra/simulated_example/fig/fig-simulated-freq-1vs1.png",
        height = 3.5,
        width = 5.5,
        dpi = 1200,
    )
"""

# Run our test
begin
    Random.seed!(1)
    nsims = 100
    T, N, p = 200, 3, 2
    warmup = 10000
    neff = 500
    thin = 5
    iter = warmup + neff * thin
    chain_g = [-ones(Bool, N * (N - 1)) for _ in 1:neff]
    scores = [-ones(Int, N * (N - 1)) for _ in 1:nsims]
    for p in 1:2, sim in 1:nsims
        println("sim: $sim")
        y, X, Z = samples[sim]
        model = BNPVAR.DiracSSModel(; p, N, T, Z)
        for t in 1:iter
            AbstractGSBPs.step!(model)
            if (t > warmup) && ((t - warmup) % thin == 0)
                chain_g[(t - warmup) ÷ thin] .= model.g
            end
        end
        # Save results
        filename = "extra/simulated_example/gamma/gamma_$(p)_$(sim).csv"
        df = DataFrame(hcat(chain_g...)' |> collect, :auto)
        R"""
        readr::write_csv($df, $filename)
        """
    end
end

# # Compute the joint posterior
# R"""
# df_joint_posterior <-
#     list.files(
#         "extra/simulated_example/gamma",
#         full.names = TRUE
#     ) |>
#     magrittr::extract(1:10) |>
#     purrr::map_df(
#         ~ .x |>
#             readr::read_csv(show_col_types = FALSE) |>
#             dplyr::count(x1, x2, x3, x4, x5, x6) |>
#             magrittr::set_colnames(c(paste0("g", 1:6), "n")) |>
#             dplyr::mutate(
#                 sim = stringr::str_extract_all(.x, "\\d+")[[1]][[1]],
#                 sim = as.integer(sim)
#             )
#     ) |>
#     dplyr::summarize(
#         prob = mean(n / 500),
#         .by = g1:g6
#     )
# """

# Compute the cause-effect relationship associated to each gamma
begin
    df_gamma_dict = DataFrame(cause = Int[], effect = Int[], idx = Int[])
    for idx in 1:6
        cause, effect = get_ij_pair(idx, N)
        push!(df_gamma_dict, (cause, effect, idx))
    end
    @rput df_gamma_dict
end

# Compute the 1vs1 decisions of our test
R"""
df_bayes_1vs1_decisions <-
    list.files(
        "extra/simulated_example/gamma",
        full.names = TRUE
    ) |>
    purrr::map_df(
        ~ .x |>
            readr::read_csv(show_col_types = FALSE) |>
            magrittr::set_colnames(paste0("g", 1:6)) |>
            dplyr::summarize(
                dplyr::across(g1:g6, mean)
            ) |>
            dplyr::mutate(
                p = stringr::str_extract_all(.x, "\\d+")[[1]][[1]],
                sim = stringr::str_extract_all(.x, "\\d+")[[1]][[2]],
                sim = as.integer(sim),
                p = as.integer(p)
            )
    ) |>
    tidyr::pivot_longer(g1:g6) |>
    dplyr::summarize(
        freq = sum(value > 0.5),
        .by = c(p, name)
    ) |>
    dplyr::rename(idx = name) |>
    dplyr::mutate(idx = gsub("g", "", idx) |> as.integer()) |>
    dplyr::inner_join(df_gamma_dict) |>
    dplyr::mutate(
        cause = paste0("y", cause),
        effect = paste0("y", effect)
    )
"""

# Create a 1vs1 analog of the frequentist summary
# Plot the results of the frequentist test
R"""
fig <-
    df_bayes_1vs1_decisions |>
    dplyr::rename(`# lags` = p) |>
    ggplot2::ggplot(
        ggplot2::aes(
            x = cause,
            y = effect,
            fill = freq,
            label = round(freq, 2)
        )
    ) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(
        ggplot2::aes(color = ifelse(freq > 80, "1", "2"))
    ) +
    ggplot2::facet_wrap(
        ggplot2::vars(`# lags`),
        labeller = "label_both",
        ncol = 2
    ) +
    ggplot2::scale_fill_distiller(
        type = "seq",
        direction = -1,
        palette = "Greys"
    ) +
    ggplot2::scale_colour_manual(values = c("white", "black")) +
    ggplot2::theme_classic() +
    ggplot2::guides(color = FALSE) +
    ggplot2::theme(
        legend.position = 'top',
        legend.justification = 'left',
        legend.direction = 'horizontal'
    ) +
    ggplot2::labs(
        x = "cause",
        y = "effect",
        fill = "frequency\n"
    )
    fig |>
    ggplot2::ggsave(
        filename = "extra/simulated_example/fig/fig-simulated-bayes-1vs1.png",
        height = 3.5,
        width = 5.5,
        dpi = 1200,
    )
"""
