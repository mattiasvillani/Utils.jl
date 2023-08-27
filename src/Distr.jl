using Distributions, SpecialFunctions, LogExpFunctions, Random 
using Distributions: InverseGamma, LocationScale, TDist
import Distributions: logpdf, pdf, cdf, quantile, std, mean
using Statistics
import Base.rand
include("normalinvchisq.jl") # Taken from ConjugatePriors.jl (which downgrades too many packages)

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
TDist(μ, σ, ν) = μ + TDist(ν)*σ



""" 
    SimDirProcess(P₀, α, ϵ) 

Simulates one realization from the Dirichlet Process DP(α⋅P₀) using the Stick-breaking construction.

ϵ>0 is the remaining stick length when the simulation terminates. 

# Examples
```julia-repl
julia> θ, π = SimDirProcess(Normal(), 5, 0.001);
julia> plot(-3:0.01:3, cdf.(Normal(), -3:0.01:3), label = "base", xlab = "x", 
    ylab = "F(x)", c = :red)
julia> plot!(θ, cumsum(π), linetype = :steppost, xlab = "θ", 
    ylab = "F(θ)", label = "realization")
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


""" 
    ZDist(α, β) 

Define the Z(α, β, 0, 1)-distribution. 

This standardized case of the Z-distribution is the same as log(x/(1-x)) for x ∼ Beta(α, β).
The general Z(α, β, μ, σ) is obtained by the Distributions.jl location-scale construction:
μ + σ*Z(α, β)

# Examples
```julia-repl
julia> zdist = ZDist(1/2, 1/2)
julia> rand(zdist, 4)'
1×4 adjoint(::Vector{Float64}) with eltype Float64:
 1.00851  0.640297  0.566234  2.16941
julia> pdf(zdist, 1)
julia> cdf(zdist, 1)
julia> zdist_general = 3 + 2*ZDist(1/2, 1/2)
julia> pdf(zdist_general, 1)zdist = ZDist(3/2,3/2)
```
""" 
struct ZDist <: ContinuousUnivariateDistribution
    α::Real
    β::Real
end

ZDist(α::Real, β::Real, μ::Real, σ::Real) = μ + σ*ZDist(α, β)

# function rand(zdist::ZDist, n::Int = 1)
#     x = rand(Beta(zdist.α, zdist.β), n)
#     return logit.(x) # this is log.(x./(1 .- x))
# end

function rand(rng::Random.AbstractRNG, d::ZDist)
    x = rand(Beta(d.α, d.β))
    return logit.(x) # this is log.(x./(1 .- x))
end

function pdf(zdist::ZDist, x::Real)
    return (logistic(x)^zdist.α * logistic(-x)^zdist.β)/beta(zdist.α,zdist.β)
end

function logpdf(zdist::ZDist, x::Real)
    return -logbeta(zdist.α, zdist.β) + zdist.α*x - (zdist.α + zdist.β)*log1pexp(x) 
                                                            # log1pexp(x) = log(1 + exp(x))
end

autodiff = true
if autodiff == false

    function cdf(zdist::ZDist, x::Real)
        # return cdf(Beta(zdist.α, zdist.β), logistic(x)) # no love from autodiff + slower
        return beta_inc(zdist.α, zdist.β, logistic(x))[1]
    end

    function quantile(zdist::ZDist, p)
        quantBeta = quantile(Beta(zdist.α, zdist.β), p)
        return logit(quantBeta)  
    end

else # autodiff does not work with beta_inc

    function cdf(zdist::ZDist, x::Real)
        if zdist.α ≈ zdist.β ≈ 1/2 # Mixture approx of Z(1/2,1/2) for speed
            cdf(MixtureModel([3.4236*TDist(10),1.8417*TDist(10)],[0.5414,1-0.5414]), x)
        elseif zdist.α ≈ zdist.β ≈ 1 # Mixture approx of Z(1,1) for speed
            cdf(MixtureModel([3.91662*Normal(),20.6839*Normal(),10.3208*Normal()],
            [0.1719,0.3198, 1-(0.1719+0.3198)]), x)
        else
            return quadgk(y -> pdf(zdist, y), -Inf, x, rtol=1e-8)[1]
        end
    end

    function quantile(zdist::ZDist, p)
        if zdist.α ≈ zdist.β ≈ 1/2 # Mixture approx of Z(1/2,1/2) for speed
            quantile(MixtureModel([3.4236*TDist(10),1.8417*TDist(10)],[0.5414,1-0.5414]),p)
        elseif zdist.α ≈ zdist.β ≈ 1 # Mixture approx of Z(1,1) for speed
            quantile(MixtureModel([3.91662*Normal(),20.6839*Normal(),10.3208*Normal()],
            [0.1719,0.3198, 1-(0.1719+0.3198)]), p)
        else
            return find_zero(x -> cdf(zdist, x) - p, (-100, 100))
        end
    end

end

function mean(zdist::ZDist)
    return digamma(zdist.α) - digamma(zdist.β)
end

function var(zdist::ZDist)
    return trigamma(zdist.α) + trigamma(zdist.β)
end

function std(zdist::ZDist)
    return sqrt(trigamma(zdist.α) + trigamma(zdist.β))
end

function params(zdist::ZDist)
    return zdist.α, zdist.β
end

""" 
    GaussianCopula(CorrMat, f)

Construct a Gaussian Copula with correlation matrix `CorrMat` and marginal distributions given by the elements in the vector of distributions in `f`. 

If `f` is a singleton, then this distribution is used for all margins.

# Examples
```doctests 
julia> using PDMats
julia> f = [Normal(2, 3), Normal()]
julia> CorrMat = PDMat([1 -0.8; -0.8 1])
julia> GC = GaussianCopula(CorrMat, f)
```
""" 
mutable struct GaussianCopula <: ContinuousMultivariateDistribution
    CorrMat::Matrix
    f::Vector{UnivariateDistribution}
end

GaussianCopula(CorrMat::Matrix, f::UnivariateDistribution) = GaussianCopula(CorrMat, 
    [f for _ in 1:size(CorrMat,1)])
GaussianCopula(f::Vector{UnivariateDistribution}) = GaussianCopula(1.0*I(length(f)), f)

length(d::GaussianCopula) = size(d.CorrMat)[1]

""" 
    pdf(d:GaussianCopula, x)

Compute probability density at `x` for the Gaussian copula with correlation matrix `CorrMat` and marginal distributions in `f` (vector of Distributions).

# Examples
The density of the Gaussian copula with Gaussian margins is:
```doctests 
julia> f = [Normal(2, 3), Normal()]
julia> CorrMat = PDMat([1 -0.8; -0.8 1])
julia> GC = GaussianCopula(CorrMat, f)
julia> pdf(GC, [1,-1])
0.009008250957272087
```
""" 
function pdf(d::GaussianCopula, x::AbstractVector{<:Real})
    u = cdf.(d.f, x)
    q = quantile.(Normal(), u)
    L = cholesky(d.CorrMat).L
    invL = inv(L)
    CorrMatInv = invL'invL  
    return exp(-logdet(L) + 0.5*q'*(I(length(q)) - CorrMatInv)*q + sum(logpdf.(d.f, x)))
end


""" 
    logpdf(d:GaussianCopula, x)

Compute the log probability density at x for the Gaussian copula with correlation matrix CorrMat and marginal distributions in f (vector of Distributions).

See also [`pdf(d:GaussianCopula, x)`](@ref).
""" 
function logpdf(d::GaussianCopula, x::AbstractVector{<:Real})
    u = cdf.(d.f, x)
    q = quantile.(Normal(), u)
    L = cholesky(d.CorrMat).L
    invL = inv(L)
    CorrMatInv = invL'invL  
    return -logdet(L) + 0.5*q'*(I(length(q)) - CorrMatInv)*q + sum(logpdf.(d.f, x))
end

function rand(rng::Random.AbstractRNG, d::GaussianCopula)
    p = size(d.CorrMat,1)
    if !isdiag(d.CorrMat)
        u = cdf.(Normal(), rand(MvNormal(d.CorrMat)))
        x = quantile.(d.f, u)
    else
        x = rand.(d.f, p)
    end   
    return x
end

""" 
    rand(d:GaussianCopula, n::Int)

Simulate from a Gaussian copula with correlation matrix `CorrMat` and marginal distributions in `f` (vector of Distributions).

See also [`pdf(d:GaussianCopula, x)`](@ref).
"""
function rand(rng::Random.AbstractRNG, d::GaussianCopula, n::Int)
    p = size(d.CorrMat,1)
    X = zeros(p,n)
    for i in 1:n
        X[:,i] = rand(d::GaussianCopula)
    end
    return X
end


