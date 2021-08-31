push!(LOAD_PATH,"../src/")

using Documenter
using SpecTools

makedocs(
    sitename = "Utils",
    format = Documenter.HTML(prettyurls = false),
    doctest = true,
    pages = Any[
        "Home" => "index.md",
        "Data/array manipulation" => "ArrayManip.md",
        "Misc" => "Misc.md"    
    ]
)


# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
