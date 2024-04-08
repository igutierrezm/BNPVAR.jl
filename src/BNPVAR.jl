module BNPVAR

using LinearAlgebra: LinearAlgebra, I, Diagonal, Symmetric
using LinearAlgebra: cholesky, inv, kron, ldiv!
using LinearAlgebra: logdet, lowrankupdate!, norm, svd
using DataFrames: DataFrames
using Distributions: Distributions, Gamma, InverseWishart, Uniform, rand!
using AbstractGSBPs: AbstractGSBPs
using OffsetArrays: OffsetArray
# using

const AG = AbstractGSBPs
const DT = Distributions
const LA = LinearAlgebra
const DF = DataFrames

# export Model, get_ij_pair
export Model, get_ij_pair

# include("sandbox.jl")
include("model.jl")
include("fit.jl")

end
