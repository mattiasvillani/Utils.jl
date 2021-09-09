@testset "Utils.jl" begin

    @test invvech([11,21,22], 2) == [11 21;21 22]
    @test invvech([11,21,31,22,32,33], 3) == [11 21 31;21 22 32;31 32 33]

    @test invvech_byrow([11,21,22], 2) == [11 21;21 22]
    @test invvech_byrow([11,21,22,31,32,33], 3) == [11 21 31;21 22 32;31 32 33]

end
