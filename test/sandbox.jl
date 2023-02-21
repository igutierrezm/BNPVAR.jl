# Setup
begin
    Random.seed!(1)
    T, N, p = 100, 2, 2
    k = N * (1 + N * p)
    Z = randn(T, N)
    c = [0.5, -0.5]
    Σ = [1.0 0.0; 0.0 1.0]
    A1 = [0.9 0.0; 0.0 0.9]
    A2 = [0.0 0.5; 0.0 0.0]
    for t in 3:T
        Z[t, :] .= c + A1 * Z[t - 1, :] + A2 * Z[t - 2, :] + cholesky(Σ).L * randn(N)
    end
    yvec = [Z[t, :] for t in 2:T]
    Xvec = [kron(I(N), [1 vec(Z[(t-p):(t-1), :]')']) for t in (1 + p):T]
    y = vcat(yvec...)
    X = vcat(Xvec...)
end

model = Model(; p, N, T, Z)
AbstractGSBPs.get_skeleton(model)
AbstractGSBPs.loglikcontrib(model, yvec[1], Xvec[1], 1)
AbstractGSBPs.step_atoms!(model, 5)
AbstractGSBPs.step!(model)

warmup = 5000
neff = 1000
thin = 1
iter = warmup + neff * thin
chain_g = [zeros(Bool, N * (N - 1)) for _ in 1:neff]
for t in 1:iter
    AbstractGSBPs.step!(model)
    if (t > warmup) && ((t - warmup) % thin == 0)
        chain_g[(t - warmup) ÷ thin] .= model.g
    end
end
@show sum(chain_g) / neff
@show get_ij_pair.(eachindex(model.g), Ref(N))