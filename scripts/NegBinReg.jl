# Main file for posterior sampling from the negative binomial regression model
#    yᵢ | β,τ ∼ NegativeBinomial(exp(τ), logit(xᵢ'β))
#    β ∼ N(μᵦ,Ωᵦ) and τ ∼ N(μₜ,σₜ)

using Distributions, LaTeXStrings, Plots, LinearAlgebra, PDMats, StatsPlots
using ForwardDiff, Optim, ProgressMeter, MCMCChains, AdvancedMH, KernelDensity
import ColorSchemes: Paired_12; colors = Paired_12[[1,2,7,8,3,4,5,6,9,10]]

"""
    PostSampNegBinReg(y, X, priorSet, algoSet)
    
Gibbs sampling with finite step Newton for negative binomial regression:
yᵢ | β,τ ∼ NegativeBinomial(exp(τ), logit(xᵢ'β))
with prior β ∼ N(μᵦ,Ωᵦ) and τ ∼ N(μₜ,σₜ)
"""
function PostSampNegBinReg(y, X, priorSet, algoSet)
    
    # Unpack algorithmic settings
    nIter, fracBurnin, nNewton, df = algoSet
    nBurnin = Int(floor(fracBurnin*nIter))
    nIter += nBurnin

    # Unpack prior settings
    μᵦ, Ωᵦ, μₜ, σₜ = priorSet

    # Initial values
    τ = 0
    β = zeros(size(X,2))

    # Set up full conditional for β with gradient and Hessian functions using AD.
    πᵦargs = [y,X,τ]
    function ℓπᵦ(β, πᵦargs...)
        p = exp.(X*β)./(1 .+ exp.(X*β))
        logLik = sum( logpdf.(NegativeBinomial.(exp(τ), p), y) )
        logPrior = logpdf(MvNormal(μᵦ, Ωᵦ), β)
        return logLik + logPrior
    end
    ∇ᵦ(β,πᵦargs...) = ForwardDiff.gradient(β -> ℓπᵦ(β, πᵦargs...), β)
    Hᵦ(β,πᵦargs...) = ForwardDiff.hessian(β -> ℓπᵦ(β, πᵦargs...), β)

    # Set up full conditional for τ with gradient and Hessian functions using AD.
    πₜargs = [y,X,β]
    function ℓπₜ(τ, πₜargs...)
        p = exp.(X*β)./(1 .+ exp.(X*β))
        logLik = sum( logpdf.(NegativeBinomial.(exp(τ), p), y) )
        logPrior = logpdf(Normal(μₜ,σₜ), τ)
        return logLik + logPrior
    end
    ∇ₜ(τ,πₜargs...) = ForwardDiff.derivative(τ -> ℓπₜ(τ, πₜargs...), τ)
    Hₜ(τ,πₜargs...) = ForwardDiff.derivative(τ -> ∇ₜ(τ,πₜargs...), τ)

    # Set up storage 
    βpost = zeros(size(X,2), nIter)
    τpost = zeros(nIter)
    ᾱ = zeros(2)  # Mean MH acceptance probability

    for i = 1:nIter
        # Sample τ
        τ, α = finiteNewtonMH(τ, ℓπₜ, ∇ₜ, Hₜ, nNewton[1], df, πₜargs...)
        τpost[i] = πᵦargs[3] = τ
        ᾱ[1] = (ᾱ[1]*(i-1) + α)/i # updating mean accept prob τ

        # Sample β
        β, α = finiteNewtonMH(β, ℓπᵦ, ∇ᵦ, Hᵦ, nNewton[2], df, πᵦargs...)
        βpost[:,i] = πₜargs[3] = β
        ᾱ[2] = (ᾱ[2]*(i-1) + α)/i # updating mean accept prob β

    end

    return (βpost[:,nBurnin+1:end], τpost[nBurnin+1:end], ᾱ)

end

# Simulate some NegativeBinomial regression data
β = [1, -1, 0.2]
τ = log(3)
n = 1000
X = [ones(n) randn(n,2)]
p = exp.(X*β)./(1 .+ exp.(X*β))
y = [rand(NegativeBinomial(exp(τ),p[i])) for i ∈ 1:n]
scatter(X[:,2],y, label = "data", ylabel = L"y", xlabel = L"x_1", color = :black)

# Algorithmic settings
nIter = 10000
fracBurnin = 0.1
nNewton = [1,1]
df = 10 
algoSet = (nIter, fracBurnin, nNewton, df)

# Prior settings
q = size(X,2)
μᵦ = zeros(q) 
Ωᵦ = 1.0*Symmetric(I(q))
μₜ = 0 # Prior for τ ∼ Normal(μₜ,σₜ) prior
σₜ = 1  # Prior stdev for τ
priorSet = (μᵦ, Ωᵦ, μₜ, σₜ)

# Optimizing all parameters jointly
function logPostNegBinReg(θ, y, X, μᵦ, Ωᵦ, μₜ, σₜ)
    τ = θ[1]
    β = θ[2:end]
    p = exp.(X*β)./(1 .+ exp.(X*β))
    logLik = sum( logpdf.(NegativeBinomial.(exp(τ), p), y) )
    logPrior = logpdf(MvNormal(μᵦ, Ωᵦ), β) + logpdf(Normal(μₜ,σₜ), τ)
    return logLik + logPrior
end
πargs = (y, X, priorSet...)
ℓπ(θ, πargs...) = logPostNegBinReg(θ, πargs...)

# Posterior mode and Hessian
∇(θ,πargs...) = ForwardDiff.gradient(θ -> ℓπ(θ, πargs...), θ)
H(θ,πargs...) = ForwardDiff.hessian(θ -> ℓπ(θ, πargs...), θ)
θinit = [1 β']'
optimRes = maximize(θ -> ℓπ(θ, πargs...), θinit)
θmode = Optim.maximizer(optimRes)
θcov = -inv(H(θmode,πargs...))
τmode = θmode[1]
τstd = √θcov[1,1]
βmode = θmode[2:end]
βcov = θcov[2:end,2:end]

  
# Plotting the log posterior and gradient for τ
τGrid = -3:0.1:3
p1 = plot(τGrid, [ℓπ([τ β']',πargs...) for τ in τGrid], 
    xlabel = L"\tau", ylabel = L"\log \pi(y\vert X, \tau)", color = colors[2], title = "log-likelihood");
# Plotting the gradient log posterior for τ
p2 = plot(τGrid, [∇([τ β']',πargs...)[1] for τ in τGrid], 
    xlabel = L"\tau", ylabel = L"\nabla \log \pi(y\vert X, \tau)", color = colors[4], title = "gradient log-likelihood");
plot(p1,p2, layout = (1,2))

# Sample the posterior by a two-block MH-within-Gibbs sampler, using finite Newton proposals.
βpost, τpost, ᾱ = PostSampNegBinReg(y, X, priorSet, algoSet)
print("Accept prob of β was $(round(ᾱ[1], digits = 3)) using $(nNewton[1]) Newton steps
Accept prob of τ was $(round(ᾱ[2], digits = 3)) using $(nNewton[2]) Newton steps")

chn = Chains([βpost' τpost],[:β₀, :β₁, :β₂, :τ])
cor([βpost' τpost], dims = 1)
plot(chn)