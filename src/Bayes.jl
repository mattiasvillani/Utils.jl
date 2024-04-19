# General functions, e.g. MCMC samplers, for Bayesian inference.


"""
    finiteNewtonMH(θₛ::Vector, ℓπ, ∇, H, nNewton, df, args...)

Finite step Newton Metropolis-Hastings update a vector parameter

- θₛ is the current value of the parameter **vector** θ
- ℓπ(θ, πargs...) is the log posterior function
- ∇(θ, πargs...) is the gradient function (perhaps obtained by automatic differentiation)
- H(θ, πargs...) is the Hessian matrix function
- nNewton is the number of Newton steps
- df is the degrees of freedom in the multivariate student-t proposal
- args are the arguments (data etc) needed to compute the log posterior ℓπ

See also the method when θₛ is a scalar.
"""
function finiteNewtonMH(θₛ::Vector, ℓπ, ∇, H, nNewton, df, πargs...)

    jitter = 1e-8 # To ensure positive definite covariance matrix
    ℓπₛ = ℓπ(θₛ, πargs...)

    # Iterate nNewton steps from starting (current) θₛ
    μ = θₛ
    for i in 1:nNewton
        μ = μ - H(μ,πargs...)\∇(μ,πargs...)
    end

    # Proposal draw from multivariate t
    Σ = PDMat(Symmetric(-inv(H(μ, πargs...) .+ jitter*I(length(μ))) )) # Cov at terminal
    θₚ = rand(MvTDist(df, μ, Σ))
    ℓqₚ = logpdf(MvTDist(df, μ, Σ), θₚ) # This is log q(θₚ|θₛ)
    ℓπₚ = ℓπ(θₚ, πargs...)

    # Now take nNewton Newton steps, this time starting from θₚ
    μ = θₚ
    for i in 1:nNewton
        μ = μ - H(μ, πargs...)\∇(μ, πargs...)
    end
    Σ = PDMat(Symmetric(-inv(H(μ,πargs...) .+ jitter*I(length(μ)) )  ))
    ℓqₛ = logpdf(MvTDist(df, μ, Σ), θₛ) # This is log q(θₛ|θₚ)

    # Accept-reject step
    α = minimum([1,exp((ℓπₚ - ℓπₛ) + (ℓqₛ - ℓqₚ))])
    if rand() < α
        θₛ = θₚ
    end
    return (θₛ,α)

end

"""
    finiteNewtonMH(θₛ, ℓπ, ∇, H, nNewton, df, args...)

Finite step Newton Metropolis-Hastings update for a **scalar** parameter

- θₛ is the current value of the parameter vector θ
- ℓπ(θ, πargs...) is the log posterior function
- ∇(θ, πargs...) is the gradient function (typically obtained by AutoDiff)
- H(θ, πargs...) is the Hessian matrix function
- nNewton is the number of Newton steps
- df is the degrees of freedom in the multivariate student-t proposal
- args are the arguments (data etc) needed to compute the log posterior ℓπ

See also the method when θₛ is a vector.
"""
function finiteNewtonMH(θₛ::Real, ℓπ, ∇, H, nNewton, df, πargs...)
    
    ℓπₛ = ℓπ(θₛ, πargs...)

    # Iterate nNewton steps from starting (current) θₛ
    μ = θₛ
    for i in 1:nNewton
        μ = μ - H(μ, πargs...)\∇(μ, πargs...)
    end

    # Proposal draw from multivariate t
    σ = √(-1/H(μ,πargs...)) # Cov at terminal point
    θₚ = rand(TDist(μ,σ,df))
    ℓqₚ = logpdf(TDist(μ,σ,df), θₚ) # This is log q(θₚ|θₛ)
    ℓπₚ = ℓπ(θₚ,πargs...)

    # Now take nNewton Newton steps, this time starting from θₚ
    μ = θₚ
    for i in 1:nNewton
        μ = μ - H(μ,πargs...)\∇(μ,πargs...)
    end
    σ = √(-1/H(μ,πargs...))
    ℓqₛ = logpdf(TDist(μ,σ,df), θₛ) # This is log q(θₛ|θₚ)

    # Accept-reject step
    α = minimum([1,exp((ℓπₚ - ℓπₛ) + (ℓqₛ - ℓqₚ))])
    if rand() < α
        θₛ = θₚ
    end
    return (θₛ,α)

end

""" 
    HPDregions(data::AbstractArray, coverage) 

Compute the highest posterior density (HPD) regions based on a kernel density estimate of the `data`. 

`coverage` ∈ (0,1) is the probability mass to be included in the HPD region.

# Examples
```julia-repl
julia> hpdregion, actualCoverage = HPDregions(randn(100), 0.95)
```
""" 
function HPDregions(data::AbstractArray, coverage)

    if size(data,2) == 1 
        data = data[:]
    else
        error("Data must be a vector")
    end
    kdeObject = kde(data) 
    binSize = step(kdeObject.x)
    sortidx = sortperm(kdeObject.density, rev=true) # descending
    xSort = kdeObject.x[sortidx] 
    densSort = kdeObject.density[sortidx]
    
    finalpointin = findfirst(cumsum(densSort*binSize) .>= coverage)
    actualCoverage = cumsum(densSort*binSize)[finalpointin]
    hpdPoints = sort(xSort[1:finalpointin])
    breakpoints = findall(diff(hpdPoints) .> 1.9*binSize)
    nIntervals = length(breakpoints) + 1
    breakpoints = [0;breakpoints;length(hpdPoints)]

    hpd = zeros(nIntervals,2)
    for j = 1:nIntervals
        hpd[j,:] = [ hpdPoints[breakpoints[j]+1] hpdPoints[breakpoints[j+1]] ]
    end

    return hpd, actualCoverage
end

""" 
    HPDregions(d::UnivariateDistribution, coverage) 

Compute the highest posterior density (HPD) regions for the distribution `d`. 

`coverage` ∈ (0,1) is the probability mass to be included in the HPD region.

# Examples
```julia-repl
julia> hpdregion, actualCoverage = HPDregions(Normal(0,1), 0.95)
```
""" 
function HPDregions(d::UnivariateDistribution, coverage)

    min, max = quantile.(d,[0.001,0.999])
    # Check if end of support has higher density
    if pdf(d, min) < pdf(d, minimum(Distributions.support(d)))
        min = minimum(Distributions.support(d))
    end
    if pdf(d, max) < pdf(d, maximum(Distributions.support(d)))
        max = maximum(Distributions.support(d))
    end
    xGrid = range(min, max, length = 1000)
    binSize = step(xGrid)
    dens = pdf.(d, xGrid)
    sortidx = sortperm(dens, rev=true) # descending
    xSort = xGrid[sortidx] 
    densSort = dens[sortidx]
    
    finalpointin = findfirst(cumsum(densSort*binSize) .>= coverage)
    actualCoverage = cumsum(densSort*binSize)[finalpointin]
    hpdPoints = sort(xSort[1:finalpointin])
    breakpoints = findall(diff(hpdPoints) .> 1.9*binSize)
    nIntervals = length(breakpoints) + 1
    breakpoints = [0;breakpoints;length(hpdPoints)]

    hpd = zeros(nIntervals,2)
    for j = 1:nIntervals
        hpd[j,:] = [ hpdPoints[breakpoints[j]+1] hpdPoints[breakpoints[j+1]] ]
    end

    return hpd, actualCoverage 
end