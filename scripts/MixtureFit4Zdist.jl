# Script for approximating the Z(α,β) distribution by a mixture.

using Distributions, Distances, Optim, Utils, SpecialFunctions, Plots
using LaTeXStrings 

figFolder = "/home/mv/Dropbox/Julia/dev/SpecLocalStat/scripts/figs/"

function distScaleMix2Target(θ, targetDist, compDist, xGrid, distFunc)
    K = length(compDist)
    σ = exp.(θ[1:K]) 
    w = exp.(θ[(K+1):end]) ./ sum(exp.(θ[(K+1):end]))
    dMix = MixtureModel(σ .* compDist, w)
    return evaluate(distFunc, pdf.(targetDist, xGrid), pdf.(dMix, xGrid) )
end

function distMix2Target(θ, targetDist, compDist, xGrid, distFunc)
    K = length(compDist)
    μ = θ[1:K]
    σ = exp.(θ[(K+1):2K]) 
    w = exp.(θ[(2K+1):end]) ./ sum(exp.(θ[(2K+1):end]))
    dMix = MixtureModel(σ .* compDist .+ μ, w)
    return evaluate(distFunc, pdf.(targetDist, xGrid), pdf.(dMix, xGrid) )
end

α = β = 0.1
targetDist = ZDist(α, β)
stdev = sqrt(trigamma(α) + trigamma(β))
xGrid = -5*stdev:0.05:5*stdev
distFunc = Euclidean()
distFunc = KLDivergence()
symmetric = pdf(targetDist, -1) ≈ pdf(targetDist, 1)
maxK = 4
selQuants = [10.0^j for j ∈ -4:0.01:-2]
quants = zeros(length(selQuants),maxK)
p = []
for K = 1:maxK
    compDist = [TDist(100) for _ ∈ 1:K];
    if symmetric
        θ₀ = [zeros(K);repeat([1/K],K)];
        optRes = maximize(θ -> -distScaleMix2Target(θ, targetDist, compDist, xGrid, distFunc), θ₀);
        θopt = optRes.res.minimizer;
        μ = zeros(K);
        σ = exp.(θopt[1:K]);
        w = exp.(θopt[(K+1):end]) ./ sum(exp.(θopt[(K+1):end]));
        global dMix = MixtureModel(σ .* compDist, w)
    else
        θ₀ = [zeros(2K);repeat([1/K],K)];
        optRes = maximize(θ -> -distMix2Target(θ, targetDist, compDist, xGrid, distFunc), θ₀);
        θopt = optRes.res.minimizer;
        μ = θopt[1:K]
        println(μ)
        σ = exp.(θopt[(K+1):2K]);
        w = exp.(θopt[(2K+1):end]) ./ sum(exp.(θopt[(2K+1):end]));
        dMix = MixtureModel(σ .* compDist .+ μ, w)
    end

    ptmp = plot(xGrid, pdf.(targetDist, xGrid), label = "Target", 
        title = "K = $K, distance = $(round(optRes.res.minimum, digits = 4))")
    plot!(ptmp, xGrid, pdf.(dMix, xGrid), label = "Mixture")
    push!(p, ptmp)
    quants[:,K] = quantile.(dMix, selQuants)
end
plot(size = (1000, 800), p..., layout = (2,2))
savefig(figFolder*"Densities.pdf")

q = []
for K = 1:maxK
    qtmp = plot(selQuants, quantile.(targetDist, selQuants), color = :black, label = "Target", title = L"K = %$K", ylab = "quantiles", xlab = "tail probability")
    plot!(qtmp, selQuants, quants[:,K], label = "Mixture", legend = :bottomright)
    push!(q, qtmp)
end
plot(size = (1000, 800), q..., layout = (2,2))
savefig(figFolder*"Quantiles.pdf")


# Find mixture approx for all α=β
distFunc = Euclidean()
distFunc = KLDivergence()
symmetric = pdf(targetDist, -1) ≈ pdf(targetDist, 1)
K = 2
selQuants = [10.0^j for j ∈ -4:0.01:-2]
quants = zeros(length(selQuants), maxK)
compDist = [TDist(10) for _ ∈ 1:K];
p = []
αgrid = 0.05:0.01:2
nα = length(αgrid)
optResAll = zeros(nα, 1 + 2*K + 1)
for (i,α) = enumerate(αgrid)
    stdev = sqrt(trigamma(α) + trigamma(β))
    xGrid = -5*stdev:0.05:5*stdev
    β = α
    targetDist = ZDist(α, β)
    θ₀ = [zeros(K);repeat([1/K],K)];
    optRes = maximize(θ -> -distScaleMix2Target(θ, targetDist, compDist, xGrid, distFunc), θ₀);
    θopt = optRes.res.minimizer;
    dist = optRes.res.minimum;
    σ = exp.(θopt[1:K]) 
    w = exp.(θopt[(K+1):end]) ./ sum(exp.(θopt[(K+1):end]))
    optResAll[i, :] = [α; σ; w; dist]
    println(optResAll[i, :])
    μ = zeros(K);
    σ = exp.(θopt[1:K]);
    w = exp.(θopt[(K+1):end]) ./ sum(exp.(θopt[(K+1):end]));
    dMix = MixtureModel(σ .* compDist .+ μ, w)
    quants[:,K] = quantile.(dMix, selQuants)
    #quantile.(targetDist, selQuants)
end

plot(αgrid, optResAll[:,2], label = L"\sigma_1")
plot!(αgrid, 0.5 .+ (1.35 ./αgrid).^(1), c = "red")

plot(αgrid, optResAll[:,3], label = L"\sigma_2")
plot!(αgrid, 0.5 .+ (0.42./αgrid).^(1.0), c = "red")


plot(αgrid, optResAll[:,4], label = L"w_1")
plot(αgrid, optResAll[:,5], label = L"w_2")
