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
library(vars)
library(bruceR)
"""

R"""
cleaned_data <-
    'extra/chilean_example/data/data.xlsx' |>
    readxl::read_xlsx(skip = 2) |>
    dplyr::as_tibble() |>
    janitor::clean_names() |>
    dplyr::mutate(
        income = log(x7_ingreso_nacional_bruto_disponible),
        investment = log(x12_formacion_bruta_de_capital_fijo),
        consumption = log(x8_consumo_total)
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
    readr::write_csv(file = "extra/chilean_example/data/cleaned_data.xlsx")
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
    neff = 4000
    thin = 10
    iter = warmup + neff * thin
    chain_g = [-ones(Bool, N * (N - 1)) for _ in 1:neff]
    for p in [2, 4]
        model = BNPVAR.Model(; p, N, T, Z)
        for t in 1:iter
            AbstractGSBPs.step!(model)
            if (t > warmup) && ((t - warmup) % thin == 0)
                chain_g[(t - warmup) ÷ thin] .= model.g
            end
        end
        df = DataFrame(hcat(chain_g...)' |> collect, :auto)
        filename = "extra/chilean_example/gamma/gamma_var_$p.csv"
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
        varname = c("investment", "income", "consumption")
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
        dplyr::inner_join(varnames_df, by = dplyr::join_by(cause_id == id)) |>
        dplyr::rename(cause_var = varname) |>
        dplyr::inner_join(varnames_df, by = dplyr::join_by(effect_id == id)) |>
        dplyr::rename(effect_var = varname) #|>
        # dplyr::select(-cause_id, -effect_id)
    """
end

# Reload the results
R"""
df <-
    "extra/chilean_example/gamma/" |>
    list.files("gamma_var*", full.names = TRUE) |>
    purrr::map_df(readr::read_csv)
"""

# # Summarize the results: modal-model
# R"""
# mod_gamma <-
#     df |>
#     dplyr::mutate(gamma = paste0(x1, x2, x3, x4, x5, x6)) |>
#     dplyr::mutate(n = dplyr::n(), .by = c(gamma, nlags)) |>
#     dplyr::arrange(dplyr::desc(n)) |>
#     dplyr::slice(1) |>
#     dplyr::select(-gamma, -n) |>
#     tidyr::pivot_longer(x1:x6) |>
#     dplyr::inner_join(cause_effect_df) |>
#     dplyr::select(cause_var, effect_var, value)
# """

# # Summarize the results: median-model
# R"""
# med_gamma <-
#     df |>
#     tidyr::pivot_longer(x1:x6) |>
#     dplyr::summarize(med = median(value), .by = "name") |>
#     dplyr::inner_join(cause_effect_df) |>
#     dplyr::select(cause_var, effect_var, med)
# """

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
        ggplot2::aes(color = ifelse(prob > 0.9, "1", "2"))
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
        filename = "extra/chilean_example/fig/fig-prob-1vs1.png",
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
        filename = "extra/chilean_example/fig/fig-prob-2vs1-investment.png",
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
        file = paste0("extra/chilean_example/pvals/pvals-1vs1-", p, ".csv")
    )
}
"""

# Plot the results
R"""
fig <-
    "extra/chilean_example/pvals" |>
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
        filename = "extra/chilean_example/fig/fig-pval-1vs1.png",
        dpi = 1200,
        height = 3.5,
        width = 5.5,
    )
"""
