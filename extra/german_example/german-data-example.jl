begin
    using Revise
    using AbstractGSBPs
    using BNPVAR
    using DataFrames
    using RCall
    using Random
    using LinearAlgebra
    using StatsBase
end

R"""
library(vars)
library(bruceR)
"""

R"""
cleaned_data <-
    'extra/german_example/data/data.xlsx' |>
    readxl::read_xlsx() |>
    dplyr::as_tibble() |>
    janitor::clean_names() |>
    dplyr::mutate(
        income = log(income),
        investment = log(invest),
        consumption = log(cons)
    ) |>
    dplyr::select(investment, income, consumption) |>
    dplyr::mutate(
        dplyr::across(
            c(income, investment, consumption),
            ~ (.x - dplyr::lag(.x, 1))
        )
    ) |>
    na.omit() |>
    dplyr::mutate(
        dplyr::across(
            c(income, investment, consumption),
            ~ (.x - mean(.x)) / sd(.x)
        )
    )
cleaned_data |>
    readr::write_csv(file = "extra/german_example/data/cleaned_data.xlsx")
"""

begin
    Random.seed!(1)
    cleaned_data = rcopy(R"cleaned_data")
    Z = Matrix{Float64}(cleaned_data)
    T, N = size(Z)
end;

# Run our test
begin
    Random.seed!(1)
    warmup = 10000
    neff = 2000
    thin = 5
    iter = warmup + neff * thin
    chain_g = [-ones(Bool, N * (N - 1)) for _ in 1:neff]
    for p in [2, 4]
        model = BNPVAR.DiracSSModel(; p, N, T, Z)
        for t in 1:iter
            AbstractGSBPs.step!(model)
            if (t > warmup) && ((t - warmup) % thin == 0)
                chain_g[(t - warmup) ÷ thin] .= model.g
            end
        end
        df = DataFrame(hcat(chain_g...)' |> collect, :auto)
        filename = "extra/german_example/gamma/gamma_var_$p.csv"
        R"""
        $df |>
            dplyr::mutate(nlags = $p) |>
            readr::write_csv($filename)
        """
    end
end

# Compute the relationship between the variable id and its name
R"""
varnames_df <-
    data.frame(
        id = c(1, 2, 3),
        varname = c("investment", "income", "consumption") |> factor()
    )
"""

# Compute the relationship between the x's and the cause/effect relationships
begin
    cause_id = getindex.(get_ij_pair.(1:(N * (N - 1)), N), 1)
    effect_id = getindex.(get_ij_pair.(1:(N * (N - 1)), N), 2)
    R"""
    cause_effect_df <-
        data.frame(
            name = paste0("x", 1:6),
            cause_id = $cause_id,
            effect_id = $effect_id
        ) |>
        dplyr::inner_join(
            varnames_df,
            by = dplyr::join_by(cause_id == id)
        ) |>
        dplyr::rename(cause_var = varname) |>
        dplyr::inner_join(
            varnames_df,
            by = dplyr::join_by(effect_id == id)
        ) |>
        dplyr::rename(effect_var = varname)
    """
end

# Reload the results
R"""
df <-
    "extra/german_example/gamma/" |>
    list.files("gamma_var*", full.names = TRUE) |>
    purrr::map_df(readr::read_csv)
"""

# Summarize the results: 1vs1 heatmap
R"""
fig <-
    df |>
    tidyr::pivot_longer(x1:x6) |>
    dplyr::summarize(prob = mean(value), .by = c(name, nlags)) |>
    dplyr::inner_join(cause_effect_df) |>
    dplyr::rename(`# lags` = nlags) |>
    ggplot2::ggplot(
        ggplot2::aes(
            x = cause_var,
            y = effect_var,
            fill = prob,
            label = round(prob, 2)
        )
    ) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(
        ggplot2::aes(color = ifelse(prob > 0.8, "1", "2"))
    ) +
    ggplot2::facet_wrap(
        ggplot2::vars(`# lags`),
        labeller = "label_both",
        ncol = 2
    ) +
    ggplot2::scale_colour_manual(values = c("white", "black")) +
    ggplot2::scale_fill_distiller(
        type = "seq",
        direction = 1,
        palette = "Greys",
        limits = c(0, 1),
        breaks = c(0, 1)
    ) +
    ggplot2::theme_classic() +
    ggplot2::guides(color = "none") +
    ggplot2::theme(
        legend.position = 'top',
        legend.justification = 'left',
        legend.direction = 'horizontal',
        axis.text.y = ggplot2::element_text(angle = 90, hjust = 0.5),
    ) +
    ggplot2::labs(
        x = "cause",
        y = "effect",
        fill = "posterior probability\n"
    )
fig |>
    ggplot2::ggsave(
        filename = "extra/german_example/fig/fig-prob-1vs1.png",
        dpi = 1200,
        height = 3.5,
        width = 5.5,
    )
"""

# Summarize the results: 2vs1 heatmap (effect: investment)
R"""
fig <-
    df |>
    dplyr::mutate(.id = 1:dplyr::n()) |>
    tidyr::pivot_longer(x1:x6) |>
    dplyr::inner_join(cause_effect_df) |>
    dplyr::filter(effect_var == "investment") |>
    dplyr::select(-effect_var, -name, -cause_id, -effect_id) |>
    tidyr::pivot_wider(names_from = cause_var, values_from = value) |>
    dplyr::summarize(freq = dplyr::n(), .by = c(income, consumption, nlags)) |>
    dplyr::mutate(
        prob = freq / sum(freq),
        income = ifelse(income == 1, "yes", "no"),
        consumption = ifelse(consumption == 1, "yes", "no"),
        .by = nlags
    ) |>
    dplyr::rename(`# lags` = nlags) |>
    ggplot2::ggplot(
        ggplot2::aes(
            x = income,
            y = consumption,
            fill = prob,
            label = round(prob, 2)
        )
    ) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(
        ggplot2::aes(color = ifelse(prob > 0.9, "1", "2"))
    ) +
    ggplot2::facet_wrap(
        ggplot2::vars(`# lags`),
        labeller = "label_both",
        ncol = 2
    ) +
    ggplot2::scale_fill_distiller(
        type = "seq",
        direction = 1,
        palette = "Greys",
        limits = c(0, 1),
        breaks = c(0, 1)
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
        fill = "posterior probability\n"
    )
fig |>
    ggplot2::ggsave(
        filename = "extra/german_example/fig/fig-prob-2vs1-investment.png",
        height = 3.5,
        width = 5.5,
        dpi = 1200,
    )
"""

# Run a competitor test
R"""
    for (p in c(2, 4)) {
        fit <-
        cleaned_data |>
        vars::VAR(p = p, type = "const")

    varnames <-
    dplyr::tibble(cause = c("investment", "income", "consumption"))

    varname_pairs <-
    varnames |>
    dplyr::cross_join(varnames) |>
    dplyr::rename(cause = cause.x, effect = cause.y) |>
    dplyr::filter(cause != effect)

    sink("tmp.txt")
    pvals <-
    varname_pairs |>
    dplyr::rowwise() |>
    dplyr::mutate(
        nlags = p,
        pval =
        bruceR::granger_causality(
            varmodel = fit,
            var.y = effect,
            var.x = cause,
            test = "Chisq"
        ) |>
        magrittr::extract2("result") |>
        magrittr::extract2("p.Chisq")
    )
    sink()

    readr::write_csv(
        x = pvals,
        file = paste0("extra/german_example/pvals/pvals-1vs1-", p, ".csv")
    )
}
"""

# Plot the results
R"""
fig <-
    "extra/german_example/pvals" |>
    list.files("pvals-1vs1-*", full.names = TRUE) |>
    purrr::map_df(readr::read_csv) |>
    dplyr::rename(`# lags` = nlags) |>
    ggplot2::ggplot(
        ggplot2::aes(
            x = cause,
            y = effect,
            fill = pval,
            label = round(pval, 2)
        )
    ) +
    ggplot2::facet_wrap(
        ggplot2::vars(`# lags`),
        labeller = "label_both",
        ncol = 2
    ) +
    ggplot2::geom_tile() +
    ggplot2::geom_text() +
    ggplot2::scale_fill_distiller(
        type = "seq",
        direction = -1,
        palette = "Greys",
        limits = c(0, 1),
        breaks = c(0, 1)
    ) +
    ggplot2::geom_text(
        ggplot2::aes(color = ifelse(pval < 0.5, "1", "2"))
    ) +
    ggplot2::scale_colour_manual(values = c("white", "black")) +
    ggplot2::theme_classic() +
    ggplot2::guides(color = FALSE) +
    ggplot2::theme(
        legend.position = 'top',
        legend.justification = 'left',
        legend.direction = 'horizontal',
        axis.text.y = ggplot2::element_text(angle = 90, hjust = 0.5),
    ) +
    ggplot2::labs(
        x = "cause",
        y = "effect",
        fill = "p-value\n"
    )
fig |>
    ggplot2::ggsave(
        filename = "extra/german_example/fig/fig-pval-1vs1.png",
        dpi = 1200,
        height = 3.5,
        width = 5.5,
    )
"""

# Try our IRF function
begin
    Random.seed!(1)
    warmup = 20000
    neff = 5000
    thin = 10
    hmax = 16
    iter = warmup + neff * thin
    chain_g = [-ones(Bool, N * (N - 1)) for _ in 1:neff]
    chain_irf = [[zeros(N, N) for _ in 1:hmax] for _ in 1:neff]
    for p in [2, 4]
        model = BNPVAR.DiracSSModel(; p, N, T, Z)
        for t in 1:iter
            AbstractGSBPs.step!(model)
            if (t > warmup) && ((t - warmup) % thin == 0)
                chain_g[(t - warmup) ÷ thin] .= model.g
                irf = BNPVAR.get_irf(model, hmax)
                for h in 1:hmax
                    chain_irf[(t - warmup) ÷ thin][h] .= irf[h]
                end
            end
        end
    end
end

# Convert our IRF into a data.frame
begin
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
end

# Plot the IRF
R"""
fig <-
    $df_chain_irf |>
    dplyr::inner_join(
        varnames_df,
        by = dplyr::join_by(cause_id == id)
    ) |>
    dplyr::rename(cause_var = varname) |>
    dplyr::inner_join(
        varnames_df,
        by = dplyr::join_by(effect_id == id)
    ) |>
    dplyr::rename(effect_var = varname) |>
    dplyr::mutate(
        effect_var =
            factor(
                effect_var,
                levels = effect_var |> levels() |> rev()
            )
    ) #|>
    dplyr::group_by(cause_var, effect_var, horizon) |>
    dplyr::summarize(
        irf_mean = mean(irf),
        irf_lb = quantile(irf, 0.05),
        irf_ub = quantile(irf, 0.95)
    ) #|>
    ggplot2::ggplot(
        ggplot2::aes(
            x = horizon,
            y = irf_mean,
            ymin = irf_lb,
            ymax = irf_ub
        )
    ) +
    ggplot2::geom_ribbon(fill = "grey80") +
    ggplot2::geom_line() +
    ggplot2::geom_hline(
        yintercept = 0,
        linetype = "dashed",
        alpha = 0.3
    ) +
    ggplot2::facet_grid(
        cols = ggplot2::vars(cause_var),
        rows = ggplot2::vars(effect_var)
    ) +
    ggplot2::theme_classic() +
    ggplot2::labs(
        x = "cause",
        y = "effect",
        fill = "IRF"
    )
fig |>
    ggplot2::ggsave(
        filename = "extra/german_example/fig/fig-irf.png",
        dpi = 1200,
        height = 3.5,
        width = 5.5,
    )
""";

d