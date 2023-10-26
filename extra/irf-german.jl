begin
    using AbstractGSBPs
    using BNPVAR
    using DataFrames
    using RCall
    using Random
    using LinearAlgebra
    using StatsBase
end

# Prepare the data
R"""
data <-
    readxl::read_xlsx('extra/german-data.xlsx') |>
    log() |>
    dplyr::mutate(
        dplyr::across(
            dplyr::everything(),
            ~ .x - dplyr::lag(.x, 1)
        )
    ) |>
    dplyr::slice(-1) |>
    dplyr::mutate(t = 1:dplyr::n()) |>
    tidyr::pivot_longer(cols = -t) |>
    dplyr::mutate(value = (value - mean(value)) / sd(value), .by = name) |>
    tidyr::pivot_wider(names_from = name, values_from = value) |>
    dplyr::arrange(t)
"""

# Plot the data
R"""
data |>
    dplyr::mutate(t = 1:dplyr::n()) |>
    tidyr::pivot_longer(cols = -t) |>
    ggplot2::ggplot(ggplot2::aes(x = t, y = value, color = name)) +
    ggplot2::geom_line()
"""

# Plot the acf
R"""
acf(data$cons) # income, invest
"""

# Create a model
begin
    Random.seed!(1)
    data = rcopy(R"data |> dplyr::select(-t)")
    Z = Matrix{Float64}(data)
    T, N = size(Z)
    p = 1
    model = BNPVAR.Model(; p, N, T, Z)
end;

# Run our test
begin
    Random.seed!(1)
    warmup = 5000
    neff = 4000
    thin = 10
    iter = warmup + neff * thin
    chain_irf = Vector{Matrix{Float64}}[]
    for t in 1:iter
        @show t
        AbstractGSBPs.step!(model)
        if (t > warmup) && ((t - warmup) % thin == 0)
            push!(chain_irf, BNPVAR.get_irf(model, 9))
        end
    end
end

# Compute the median irf
irf_med = mean(chain_irf)

# Reshape the IRF
R"""
df <-
    1:length($irf_med) |>
    purrr::map_df(\(.x) {
        varnames <- c("invest", "income", "cons")
        df <- data.frame(($irf_med)[[.x]])
        colnames(df) <- varnames
        result <- df |>
            dplyr::mutate(
                effect_var = varnames,
                horizon = as.integer(.x)
            ) |>
            tidyr::pivot_longer(
                cols = -c(effect_var, horizon),
                names_to = "cause_var",
                values_to = "irf"
            )
    })
"""

# Compute the impulse response functions
R"""
df |>
    ggplot2::ggplot(
        ggplot2::aes(x = horizon, y = irf)
    ) +
    ggplot2::geom_line() +
    ggplot2::facet_grid(
        cols = ggplot2::vars(cause_var),
        rows = ggplot2::vars(effect_var),
        scales = "free"
    )
"""
