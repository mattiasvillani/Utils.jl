# Some distributions not available in Distributions.jl

using Distributions: InverseGamma, LocationScale, TDist, pdf

""" 
    ScaledInverseChiSq(ν,τ²) 

Defines the Scaled inverse Chi2 distribution with location ν and scale τ. 

This is a convenience function that is just calling InverseGamma(ν/2,ν*τ²/2) 

# Examples
```jldoctest
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
julia> dist = TDist(1, 2, 5)
julia> mean(dist) 
1.0
julia> pdf(dist, 1)
0.18980334491124723
```
""" 
TDist(μ, σ, ν) = LocationScale(μ, σ, TDist(ν))



export ScaledInverseChiSq, TDist

