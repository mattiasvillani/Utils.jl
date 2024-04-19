#Gibbs sampling with finite step Newton for one-parameter Polya-gamma distribution PG(b,0)
#    yᵢ | b iid∼ PG(b,0)
#    b ∼ LogNormal(m,s)

using Distributions, LaTeXStrings, Plots
using PolyaGammaSamplers
using ForwardDiff, Optim, ProgressMeter
using Utils
import ColorSchemes: Paired_12; colors = Paired_12[[1,2,7,8,3,4,5,6,9,10]]

# Set up posterior, gradient and Hessian for one-parameter Polya-gamma distribution PG(b,0)
# Reparametrizing b = exp(b̃) to ensure positivity
function ℓπ(b̃, y, m, s, nterms)
    b = exp(b̃)
    logLik = sum( logpdf.(PGDistOneParam(b, nterms), y) )
    logPrior = logpdf(LogNormal(m, s), b)
    return logLik + logPrior
end
∇(b̃, y, m, s, nterms) = ForwardDiff.derivative(b̃ -> ℓπ(b̃, y, m, s, nterms), b̃)
H(b̃, y, m, s, nterms) = ForwardDiff.derivative(b̃ -> ∇(b̃, y, m, s, nterms), b̃)

# Simulate some PolyaGamma(b,0) data and plot
b = 2
n = 100
y = rand(PolyaGammaPSWSampler(b, 0), n)
histogram(y, xlabel = L"y", fill = colors[1])

# Set up prior and plot it
m = 1/2 
s = 1
plot(0.01:0.01:10, x -> pdf(LogNormal(m,s),x), label = "LogNormal($m,$s)", lw = 2,
        xlabel = L"b", ylabel = L"p(b)", color = colors[2], title = "prior")
nterms = 10 # Number of terms in the truncation of the Polya-gamma density.

# Algorithmic settings
nIter = 1000
fracBurnin = 0.1 # Fraction of burnin
nNewton = 3      # Number of Newton steps
df = 10          # Degrees of freedom for t-distribution proposal
algoSet = (nIter, fracBurnin, nNewton, df)
nBurnin = floor(Int, fracBurnin*nIter)
nIter += nBurnin

 # Set up storage 
 b̃post = zeros(nIter)
 ᾱ = 0  # Mean MH acceptance probability for MH

 # Initial values from optimization
optimRes = maximize(b̃ -> ℓπ(b̃[1], y, m, s, nterms), [log(2)])
b̃ = Optim.maximizer(optimRes)

@showprogress "Posterior sampling" for i = 1:nIter
    # Sample unrestricted b̃
    b̃, α = finiteNewtonMH(b̃, ℓπ, ∇, H, nNewton, df, y, m, s, nterms)
    b̃post[i] = b̃
    ᾱ = (ᾱ*(i-1) + α)/i # updating mean accept prob 
end
bpost = exp.(b̃post[nBurnin+1:end])
print("Accept prob of b was $(round(ᾱ, digits = 3)) using $(nNewton) Newton steps")

histogram(bpost, xlabel = L"b", fill = (0, 0.5,colors[1]), linecolor = colors[1], 
    normalize = true, label = "Posterior sample", 
    title = "True "*L"b=%$b"*", sample size "*L"n=%$n")
binwidth = 0.01
bgrid = 0.01:binwidth:maximum(bpost)
post = exp.(ℓπ.(bgrid, Ref(y), λ, nterms))
post = post/(binwidth*sum(post))
plot!(bgrid, post, color = colors[2], label = "True Posterior", linewidth = 2)
scatter!([b], [0], color = colors[3], label = "True b", ms = 5)
