@testset "Distr.jl" begin

    ## Z-distribution
    zdist = ZDist(1/2,1/2)
    @test Utils.cdf.(zdist, Utils.quantile.(zdist, 0.1:0.1:0.9)) ≈ 0.1:0.1:0.9

    zdist = ZDist(3/2,3/2)
    @test Utils.cdf.(zdist,Utils.quantile.(zdist, 0.1:0.1:0.9)) ≈ 0.1:0.1:0.9

    @test pdf(zdist, 1) ≈ exp(logpdf(zdist, 1))

    @test cdf(zdist,-1) ≈ 1 - cdf(zdist, 1) # symmetry test

    @test Utils.var(zdist) ≈ Utils.std(zdist).^2

    # Z-distribution, location scale variant
    x = 1; μ = 2; σ = 3;
    @test mean(ZDist(1/2,1/2, μ, σ)) ≈ μ 
    @test std(ZDist(1/2,1/2, 0, σ)) ≈ σ*std(ZDist(1/2,1/2))
    
    @test pdf(ZDist(1/2,1/2, μ, σ), x) ≈ (1/σ)*pdf(ZDist(1/2,1/2), (x-μ)/σ)
    @test cdf(ZDist(1/2,1/2, μ, σ), x) ≈ cdf(ZDist(1/2,1/2), (x-μ)/σ)

    @test Distributions.var(TDist(3, 2, 5)) == 2^2*(5/(5-2))

    @test Distributions.mean(ScaledInverseChiSq(10, 5)) == 5*(10/(10-2))

    # Test if pdf of Gaussian Copula with Gaussian margins equals Multivariate Gaussian pdf
    μ₁ = 10; μ₂ = 0; σ₁ = 1; σ₂ = 3; σ12 = -1.9;
    ρ = σ12/(σ₁*σ₂)
    μ = [μ₁; μ₂]
    Σ = [σ₁^2 σ12; σ12 σ₂^2]
    Ω = PDMat([1 ρ; ρ 1])
    MvN = MvNormal(μ, Σ)
    x = rand(MvN)
    GC = GaussianCopula(Ω, [Normal(μ₁, σ₁), Normal(μ₂, σ₂)])
    @test pdf(GC, x) ≈ pdf(MvN, x)

    @test logpdf(GC, x) ≈ log(pdf(GC, x))

end
