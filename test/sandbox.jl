# Setup
begin
    Random.seed!(1)
    T, N, p = 100, 2, 1
    k = N * (1 + N * p)
    Z = randn(T, N)
    c = [0.0, 0.0]
    Σ = [1.0 0.0; 0.0 2.0]
    A = [0.5 0.0; 0.0 0.5]
    for t in 2:T
        Z[t, :] .= c + A * Z[t - 1, :] + cholesky(Σ).L * randn(N)
    end
    yvec = [Z[t, :] for t in 2:T]
    Xvec = [kron(I(N), [1 Z[t-1, :]']) for t in 2:T]
    y = vcat(yvec...)
    X = vcat(Xvec...)
end

model = Model(; p, N, T, Z)
AbstractGSBPs.get_skeleton(model)
AbstractGSBPs.loglikcontrib(model, yvec[1], Xvec[1], 1)
AbstractGSBPs.step_atoms!(model, 5)
AbstractGSBPs.step!(model)
@show BNPVAR.propose_addition(model, 2)
@show BNPVAR.propose_deletion(model, 2)
@show BNPVAR.propose_swap(model, 2)

warmup = 5000
neff = 100
thin = 1
iter = warmup + neff * thin
chain_g = [zeros(Bool, k) for _ in 1:neff]
for t in 1:iter
    AbstractGSBPs.step!(model)
    if (t > warmup) && ((t - warmup) % thin == 0)
        chain_g[(t - warmup) ÷ thin] .= model.g
    end
end
@show model.β[1]
@show sum(chain_g) / neff
