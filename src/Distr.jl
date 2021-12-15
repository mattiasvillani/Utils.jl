# Some distributions not available in Distributions.jl

using Distributions: InverseGamma, LocationScale, TDist, pdf
using Statistics

""" 
    ScaledInverseChiSq(ν,τ²) 

Defines the Scaled inverse Chi2 distribution with location ν and scale τ. 

This is a convenience function that is just calling InverseGamma(ν/2,ν*τ²/2) 

# Examples
```jldoctest
julia> using Statistics: mean;
julia> using Distributions: pdf;
julia> dist = ScaledInverseChiSq(10,3^2);
julia> mean(dist)
11.25
julia> pdf(dist, 12)
0.06055632954714239
```
"""
ScaledInverseChiSq(ν,τ²) = InverseGamma(ν/2,ν*τ²/2)

""" 
    TDist(μ, σ, ν) 

Defines the three parameter version of the Student-t distribution.


The distribution is parameterized so that the variance is σ²ν/(ν-2). The distribution is constructed by extending the standard student-t in Distributions.jl using the `LocationScale` construction in that same package. 

# Examples
```jldoctest
julia> using Statistics: mean;
julia> using Distributions: pdf;
julia> dist = TDist(1, 2, 5)
julia> mean(dist) 
1.0
julia> pdf(dist, 1)
0.18980334491124723
```
""" 
TDist(μ, σ, ν) = LocationScale(μ, σ, TDist(ν))



""" 
    SimDirProcess(P₀, α, ϵ) 

Simulates one realization from the Dirichlet Process DP(α⋅P₀) using the Stick-breaking construction.

ϵ>0 is the remaining stick length when the simulation terminates. 

# Examples
```julia-repl
julia> θ, π = SimDirProcess(Normal(), 2, 0.001);
julia> plot(θ, cumsum(π), linetype = :steppost, label = nothing, xlab = "θ", ylab = "F(θ)")
```
""" 
function SimDirProcess(P₀, α, ϵ)
    remainStickLength = 1
    θ = Float64[];
    π = Float64[];
    while remainStickLength > ϵ
        θₕ = rand(P₀)
        Vₕ = rand(Beta(1, α))
        πₕ = Vₕ*remainStickLength
        remainStickLength = remainStickLength*(1-Vₕ) 
        push!(θ, θₕ)
        push!(π, πₕ)
    end
    sortIdx = sortperm(θ)
    return θ[sortIdx], π[sortIdx]
end

export ScaledInverseChiSq, TDist, SimDirProcess

