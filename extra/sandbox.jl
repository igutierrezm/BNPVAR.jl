using Revise
using BNPVAR
using DataFrames
DF = DataFrames
Y = randn(10, 2)
out = BNPVAR.fit(Y; iter = 2, warmup = 1)
begin
    df_chain_gamma = out["gamma"]
end
