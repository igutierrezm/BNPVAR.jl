module BNPVAR

using LinearAlgebra: LinearAlgebra, I, Diagonal, Symmetric
using LinearAlgebra: cholesky, inv, kron, ldiv!
using LinearAlgebra: logdet, lowrankupdate!, norm, svd
using Distributions: Distributions, Gamma, InverseWishart, Uniform, rand!
using AbstractGSBPs: AbstractGSBPs
using BayesVAR: BayesVAR
using OffsetArrays: OffsetArray
# using

const AG = AbstractGSBPs
const LA = LinearAlgebra

# export Model, get_ij_pair
export DiracSSModel, get_ij_pair

# include("sandbox.jl")
include("dirac_ss.jl")
# include("dirac_ss2.jl")

end
