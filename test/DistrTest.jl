@testset "Distr.jl" begin
    
    @test var(TDist(3, 2, 5)) == 2^2*(5/(5-2))

    @test mean(ScaledInverseChiSq(10, 5)) == 5*(10/(10-2))
end