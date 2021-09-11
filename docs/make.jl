push!(LOAD_PATH,"../src/")

using Documenter
using Utils, Distributions, Statistics

#DocMeta.setdocmeta!(Utils, :DocTestSetup, :(using Utils, Statistics, Distributions); recursive=true)

makedocs(
    sitename = "Utils",
    modules = [Utils],
    format = Documenter.HTML(prettyurls = false),
    doctest = false, # disabling for now since there is some issue with modules here
    pages = Any[
        "Home" => "index.md",
        "Data/array manipulation" => "ArrayManip.md",
        "Distributions" => "Distr.md",
        "Misc" => "Misc.md"    
    ]
)

#deploydocs(
#    repo  = "github.com/mattiasvillani/Utils.jl",
#)
