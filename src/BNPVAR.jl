module BNPVAR

using LinearAlgebra: kron, I
using Distributions: Distributions
using AbstractGSBPs: AbstractGSBPs
using BayesVAR: BayesVAR

export Model, get_ij_pair

include("sandbox.jl")

end
