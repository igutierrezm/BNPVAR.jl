using Revise
using BNPVAR
using DataFrames
DF = DataFrames
Y = randn(10, 2)
out = BNPVAR.fit(Y; iter = 2, warmup = 1)
# begin
#     df_chain_gamma[!, :iter] = collect(1:size(df_chain_gamma, 1))
#     df_chain_gamma = DF.stack(df_chain_gamma, DF.Not(:iter))
#     df_chain_gamma[!, :var_id] =
#         df_chain_gamma[!, :variable] .|>
#         String |>
#         #(x) -> strip(x, 'x') .|>
#         #Int
# end
