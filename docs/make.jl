using BNPVAR
using Documenter

DocMeta.setdocmeta!(BNPVAR, :DocTestSetup, :(using BNPVAR); recursive=true)

makedocs(;
    modules=[BNPVAR],
    authors="Iván Gutiérrez <ivangutierrez1988@gmail.com> and contributors",
    repo="https://github.com/igutierrezm/BNPVAR.jl/blob/{commit}{path}#{line}",
    sitename="BNPVAR.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://igutierrezm.github.io/BNPVAR.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/igutierrezm/BNPVAR.jl",
    devbranch="main",
)
