begin
    using RCall
    using Random
    using LinearAlgebra
    Random.seed!(1)
    R"""
    library("bruceR")
    library("vars")
    """
end

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
function generate_sample()
    T, N, p = 100, 2, 2
    Z = randn(T, N)
    c = [0.5, -0.5]
    Σ = [1.0 0.0; 0.0 1.0]
    A1 = [0.5 0.0; 0.0 0.5]
    A2 = [0.5 0.1; 0.0 0.5]
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
    samples = [generate_sample() for _ in 1:100];
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

# Run our test
begin
    N = 2
    p = 2
    warmup = 5000
    neff = 1000
    thin = 1
    iter = warmup + neff * thin
    chain_g = [zeros(Bool, N * (N - 1)) for _ in 1:neff]
    bay_scores = -ones(100)
    for i in 1:100
        model = Model(; p, N, T, Z)
        for t in 1:iter
            AbstractGSBPs.step!(model)
            if (t > warmup) && ((t - warmup) % thin == 0)
                chain_g[(t - warmup) ÷ thin] .= model.g
            end
        end
        bay_scores[i] = sum(chain_g)[1] / neff <= 0.5
    end
    sum(bay_scores) / 100
end
