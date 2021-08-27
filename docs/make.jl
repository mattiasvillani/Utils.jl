push!(LOAD_PATH,"../src/")

using Documenter
using Utils

makedocs(
    sitename = "Utils",
    format = Documenter.HTML(),
    modules = [Utils]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
