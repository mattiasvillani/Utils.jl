@testset "Utils.jl" begin

    @test invvech([11,21,22], 2) == [11 21;21 22]
    @test invvech([11,21,31,22,32,33], 3) == [11 21 31;21 22 32;31 32 33]

    @test invvech_byrow([11,21,22], 2) == [11 21;21 22]
    @test invvech_byrow([11,21,22,31,32,33], 3) == [11 21 31;21 22 32;31 32 33]

    @test CovMatEquiCorr([1.0,1.0], [0.0,0.0], [1,1]) == I(2)

    σₓ = rand(5)
    ρ = rand(2)
    pBlock = [2,3]
    CovMat = CovMatEquiCorr(σₓ, ρ, pBlock)
    @test CovMat[1,3] == 0
    @test CovMat[1,2]/(√CovMat[1,1]*√CovMat[2,2]) ≈ ρ[1]

    ρComputed, σComputed = Cov2Corr(CovMat);
    @test ρComputed[1,2] ≈ ρ[1]
    @test ρComputed[4,5] ≈ ρ[2]
    @test σComputed[1] ≈ σₓ[1]

    # Misc.jl
    A = [1 2 3; 4 5 6; 7 8 9]
    @test find_min_matrix(A, 3)[3] == CartesianIndex(1, 3)
    @test find_max_matrix(A, 3)[1] == CartesianIndex(3, 3)

end
