using Pkg

# Activate the docs environment in docs/Project.toml
Pkg.activate(@__DIR__)
Pkg.instantiate()  # optional but good to keep

@show Base.active_project()  # temporary debug, can be removed later

using Documenter
using Utils, Distributions, Statistics

DocMeta.setdocmeta!(Utils, :DocTestSetup, :(using Utils); recursive=true)

makedocs(
    modules = [Utils],
    authors="Mattias Villani",
    sitename = "Utils.jl",
    format=Documenter.HTML(;
        canonical="https://mattiasvillani.github.io/Utils.jl",
        edit_link="main",
        assets=String[],
    ),
    checkdocs = :exports,  # or :none if you want no check
    pages = Any[
        "Home" => "index.md",
        "Distributions" => "Distr.md",
        "Linear algebra" => "LinAlgMisc.md",
        "Misc" => "Misc.md"    
    ]
)

deploydocs(
   repo  = "github.com/mattiasvillani/Utils.jl",
   devbranch = "main"
)
