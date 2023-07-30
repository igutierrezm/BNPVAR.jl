begin
    using AbstractGSBPs
    using BNPVAR
    using DataFrames
    using RCall
    using Random
    using LinearAlgebra
    using StatsBase
end

begin
    Random.seed!(1)
    data = rcopy(R"readxl::read_xlsx('extra/german-data.xlsx')")
    Z = Matrix{Float64}(data)
    T, N = size(Z)
    p = 2
    model = BNPVAR.DiracSSModel(; p, N, T, Z)
end;

# Plot the data

# Run our test
begin
    Random.seed!(1)
    warmup = 5000
    neff = 4000
    thin = 10
    iter = warmup + neff * thin
    chain_g = [-ones(Bool, N * (N - 1)) for _ in 1:neff]
    for t in 1:iter
        AbstractGSBPs.step!(model)
        if (t > warmup) && ((t - warmup) % thin == 0)
            chain_g[(t - warmup) ÷ thin] .= model.g
        end
    end
    df = DataFrame(hcat(chain_g...)' |> collect, :auto)
    filename = "extra/dirac_gamma_german-data.csv"
    R"readr::write_csv($df, $filename)"
end

# Compute the relationship between the variable id and its name
R"""
varnames_df <-
    data.frame(
        id = c(1, 2, 3),
        varname = c("invest", "income", "cons")
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
filename <- "extra/dirac_gamma_german-data.csv"
df <- readr::read_csv(filename)
"""

# Summarize the results: modal-model
R"""
mod_gamma <-
    df |>
    dplyr::mutate(gamma = paste0(x1, x2, x3, x4, x5, x6)) |>
    dplyr::mutate(n = dplyr::n(), .by = "gamma") |>
    dplyr::arrange(dplyr::desc(n)) |>
    dplyr::slice(1) |>
    dplyr::select(-gamma, -n) |>
    tidyr::pivot_longer(x1:x6) |>
    dplyr::inner_join(cause_effect_df) |>
    dplyr::select(cause_var, effect_var, value)
"""

# Summarize the results: median-model
R"""
med_gamma <-
    df |>
    tidyr::pivot_longer(x1:x6) |>
    dplyr::summarize(med = median(value), .by = "name") |>
    dplyr::inner_join(cause_effect_df) |>
    dplyr::select(cause_var, effect_var, med)
"""

# Summarize the results: 1vs1 heatmap
R"""
df |>
    tidyr::pivot_longer(x1:x6) |>
    dplyr::summarize(prob = mean(value), .by = "name") |>
    dplyr::inner_join(cause_effect_df) |>
    ggplot2::ggplot(
        ggplot2::aes(x = cause_var, y = effect_var, fill = prob)
    ) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_distiller(
        type = "seq",
        direction = 1,
        palette = "Greys"
    )
"""

# Summarize the results: 2vs1 heatmap (effect: invest)
R"""
df |>
    dplyr::mutate(.id = 1:dplyr::n()) |>
    tidyr::pivot_longer(x1:x6) |>
    dplyr::inner_join(cause_effect_df) |>
    dplyr::filter(effect_var == "invest") |>
    dplyr::select(-effect_var, -name, -cause_id, -effect_id) |>
    tidyr::pivot_wider(names_from = cause_var, values_from = value) |>
    dplyr::summarize(freq = dplyr::n(), .by = c(income, cons)) |>
    dplyr::mutate(
        prob = freq / sum(freq),
        income = ifelse(income == 1, "yes", "no"),
        cons = ifelse(cons == 1, "yes", "no"),
        case = paste0(income, "/", cons)
    ) |>
    ggplot2::ggplot(
        ggplot2::aes(y = reorder(case, -prob), x = prob)
    ) +
    ggplot2::geom_col() +
    ggplot2::labs(
        y = "does incomde/consumption granger-cause investment?",
        x = "posterior probability"
    )
ggplot2::ggsave("extra/fig-2vs1-investment.png")
"""
