module BNPVAR

using LinearAlgebra: I, Diagonal, Symmetric
using LinearAlgebra: cholesky, inv, kron, ldiv!
using LinearAlgebra: logdet, lowrankupdate!, norm, svd
using Distributions: Distributions, Gamma, InverseWishart, rand!
using AbstractGSBPs: AbstractGSBPs
using BayesVAR: BayesVAR

# export Model, get_ij_pair
export DiracSSModel, get_ij_pair

# include("sandbox.jl")
include("dirac_ss.jl")

end
