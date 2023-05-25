# General functions, e.g. MCMC samplers, for Bayesian inference.


"""
    finiteNewtonMH(θₛ, ℓπ, ∇, H, nNewton, df, args...)

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

    ℓπₛ = ℓπ(θₛ, πargs...)

    # Iterate nNewton steps from starting (current) θₛ
    μ = θₛ
    for i in 1:nNewton
        μ = μ - H(μ,πargs...)\∇(μ,πargs...)
    end

    # Proposal draw from multivariate t
    Σ = PDMat(Symmetric(-inv(H(μ,πargs...)))) # Cov at terminal point
    θₚ = rand(MvTDist(df, μ, Σ))
    ℓqₚ = logpdf(MvTDist(df, μ, Σ), θₚ) # This is log q(θₚ|θₛ)
    ℓπₚ = ℓπ(θₚ,πargs...)

    # Now take nNewton Newton steps, this time starting from θₚ
    μ = θₚ
    for i in 1:nNewton
        μ = μ - H(μ,πargs...)\∇(μ,πargs...)
    end
    Σ = PDMat(Symmetric(-inv(H(μ,πargs...))))
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
        μ = μ - H(μ,πargs...)\∇(μ,πargs...)
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

