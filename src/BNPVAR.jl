module BNPVAR

using LinearAlgebra: I, Symmetric, cholesky, inv, kron, ldiv!, logdet, lowrankupdate!, norm
using Distributions: Distributions, Gamma, InverseWishart, rand!
using AbstractGSBPs: AbstractGSBPs
using BayesVAR: BayesVAR

# export Model, get_ij_pair
export DiracSSModel, get_ij_pair

# include("sandbox.jl")
include("dirac_ss.jl")

end
