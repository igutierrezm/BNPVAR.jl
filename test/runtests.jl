using BNPVAR
using Test

using AbstractGSBPs
using LinearAlgebra
using Random

const BV = BNPVAR

@testset "BNPVAR.jl" begin
    # include("sandbox.jl")
    # include("dirac_ss.jl")
    include("dirac_ss2.jl")
end
