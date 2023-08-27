"""
    invvech(v, p; fillupper = true)

Construct a symmetric matrix from its diagonal and lower diagonal elements. Fills by columns.

- v is a vector with the unique elements
- p is the number of rows or columns of the returned symmetric matrix.

See also [`invvech_byrow(v, p; fillupper = true)`](@ref)

# Examples
```julia-repl
julia> invvech([11,21,31,22,32,33], 3)
3×3 Matrix{Int64}:
 11  21  31
 21  22  32
 31  32  33
```
"""
function invvech(v, p; fillupper = true)
    Cov = zeros(eltype(v), p, p)
    j = 1;
    for i in 1:p
        Cov[:,i] = [zeros(eltype(v), i-1, 1) ; v[j:j+(p-i)]];
        j = j + (p-i) + 1;
    end
    if fillupper
        Cov = Cov + Cov' - diagm(diag(Cov));
    end
    return Cov
end


"""
    invvech_byrow(v, p; fillupper = true)

Construct a symmetric matrix from its diagonal and lower diagonal elements. Fills by row.

- v is a vector with the unique elements
- p is the number of rows or columns of the returned symmetric matrix.

See also [`invvech(v, p; fillupper = true)`](@ref)

# Examples
```julia-repl
julia> invvech_byrow([11,21,22,31,32,33], 3)
3×3 Matrix{Int64}:
 11  21  31
 21  22  32
 31  32  33
```
"""
function invvech_byrow(v, p; fillupper = true)
    Cov = zeros(eltype(v), p, p)
    count = 0
    for i in 1:p
        for j in 1:i
            count = count + 1
            Cov[i,j] = v[count]
        end
    end
    if fillupper
        Cov = Cov + Cov' - diagm(diag(Cov));
    end
    return Cov
end

""" 
    CovMatEquiCorr(σₓ, ρ, pBlock) 

Set up a covariance matrix with equi-correlation within blocks of variables

- σₓ is a p-vector with standard deviations 
- ρ[i] is the correlation within block i 
- pBlock[j] is the number of variables in block j

# Examples
```julia-repl
julia> σₓ = [1,2,3]; ρ = [0.5,0.8]; pBlock = [1,2];
julia> CovMatEquiCorr(σₓ, ρ, pBlock)
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  4.0  4.8
 0.0  4.8  9.0
```
""" 
function CovMatEquiCorr(σₓ, ρ, pBlock)
    p = sum(pBlock)
    corrMat = zeros(p,p)
    count = 0 
    println(length(ρ))
    for i in 1:length(ρ)
        idx = (1 + count):(count + pBlock[i])
        corrMat[idx, idx] = ρ[i]*ones(pBlock[i], pBlock[i])
        count = count + pBlock[i]
    end
    corrMat[diagind(corrMat)] .= 1
    return diagm(σₓ)*corrMat*diagm(σₓ)
end

""" 
    ρ, σ = Cov2Corr(Σ) 

Compute the correlation matrix `ρ` and vector standard deviations `σ` for the covariance matrix Σ.

# Examples
```julia-repl
julia> Σ = CovMatEquiCorr([1,2,3], [0.7], [3]) # Covariance matrix with all corr = 0.7
3×3 Matrix{Float64}:
 1.0  1.4  2.1
 1.4  4.0  4.2
 2.1  4.2  9.0

julia> ρ, σ = Cov2Corr(Σ); ρ
3×3 Matrix{Float64}:
 1.0  0.7  0.7
 0.7  1.0  0.7
 0.7  0.7  1.0
```
""" 
function Cov2Corr(Σ)
    StdDev = .√diag(Σ)
    D = diagm(1 ./StdDev)
    return D*Σ*D, StdDev
end