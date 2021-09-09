push!(LOAD_PATH,"../src/")

using Documenter
using SpecTools, Distributions

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


# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
