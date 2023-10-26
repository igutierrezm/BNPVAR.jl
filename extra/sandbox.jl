using Revise
using BNPVAR
Y = randn(10, 2)
BNPVAR.fit(Y; iter = 2, warmup = 1)