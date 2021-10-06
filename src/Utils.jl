module Utils

# Python functions import
using PyCall, LinearAlgebra, Distributions, Statistics, PDMats, ForwardDiff, DataFrames

include("Distr.jl") # some extra distributions
include("Bayes.jl") # Bayesian inference utilities, e.g. posterior samplers.

# Importing the pickle functionality from Python using PyCall
py"""
import pickle
def unpickle(filename):
    with open(filename + '.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

"""

# Make julia function that pickles (reads a Python file with data)
"""
    unpickle(filename)

Read a Python data file using Python's Pickle via PyCall.jl.
"""
function unpickle(filename)
    return py"unpickle"(filename)
end

# invvech - inverse operation to the vech operator for symmetric matrices
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

"""
    plotFcnGrid(f, xGrid, xNames, fcnArgs...;ylabel="", title ="", levels = 10,fill=:viridis)

Plotting a function of a 1D or 2D grid.
"""
function plotFcnGrid(f, xGrid, xNames, fcnArgs...;ylabel="", title ="",
        levels = 10, fill=:viridis)
    griddedx = [typex<: StepRangeLen for typex in typeof.(xGrid)]
    xGridReduced = [x for x in xGrid if typeof(x)<: StepRangeLen]
    xFixed = [x for x in xGrid if typeof(x)<: Number]
    xNamesGrid = xNames[griddedx]
    xNamesFixed = xNames[griddedx.==false]
    function fcnFrozen(xSubset, fcnArgs...)
        x = zeros(length(xGrid))
        x[griddedx.==false] = xFixed
        x[griddedx.==true] = xSubset
        return f(x, fcnArgs...)
    end

    if sum(griddedx.==true) == 1 # 1D
        fValGrid = [fcnFrozen([x1], fcnArgs...) for x1 in xGridReduced[1]]
        p = plot(xGridReduced[1], fValGrid, xlabel = xNamesGrid[1], ylabel = ylabel,
            title = title)
    elseif sum(griddedx.==true) == 2 # 2D
        fValGrid = [fcnFrozen([x1, x2], fcnArgs...)
            for x1 in xGridReduced[1], x2 in xGridReduced[2]]
        p = contourf(xGridReduced[2], xGridReduced[1], fValGrid;
            levels = levels, fill=fill, xlabel = xNamesGrid[2],
            ylabel = xNamesGrid[1], title = title, cgrad = logscale)
    elseif sum(griddedx.==true) == 3 # 3D FIXME: Not implemented yet
        xGridTmp = xGrid
        p = zeros(0)
        for par in xGridReduced[3]
            xGridTmp[findall(griddedx)[end]] = par # Fix the third input
            subp = plotFcnGrid(f, xGridTmp, xNames, fcnArgs...;ylabel="", title ="",
                    levels = 10, fill=:viridis);
            append!(p, subp)
        end
        #l = @layout [a ; b c]
        plot(p)
    else error("Can only grid over one or two arguments. Please check xGrid")
    end
    return p
end


""" 
    plotClassifier2D(y, X, predictFunc; gridSize = [100,100], colors = missing, axisLabels = missing) 

Plots the decision boundaries of a classifier predictFunc() with two inputs

- predictFunc should take x1, x2 as inputs and returns a class in unique(y)
- class labels can be Categorical or Int

""" 
function plotClassifier2D(y, X, predictFunc; gridSize = [100,100], 
        colors = missing, axisLabels = missing)
    if ismissing(colors)
        colors = Paired_12[[1,2,7,8,3,4,5,6,9,10,11,12]];
    end
    classes = unique(y)
    if ismissing(labels)
        axisLabels = names(X)
    end
    markerLarge = 3
    markerSmall = 3
    x1 = X[:,1]
    x2 = X[:,2]
    # Compute predictions over a grid in x1-x2 space
    x1Grid = range(minimum(x1),maximum(x1), length = gridSize[1])
    x2Grid = range(minimum(x2),maximum(x2), length = gridSize[2])
    yPreds = Matrix{Union{Int, String}}(undef, length(x2Grid), length(x1Grid))
    for (i,x1) in enumerate(x1Grid)
        for (j,x2) in enumerate(x2Grid)
            yPreds[j,i] = predictFunc([x1,x2])
        end
    end
    x1Vec = (x1Grid' .* ones(length(x2Grid)))[:]
    x2Vec = (ones(length(x1Grid))' .* x2Grid )[:]
    yPredsVec = yPreds[:]

    # Plot classifications as big points with light color
    p = scatter(xlabel = axisLabels[1], ylabel = axisLabels[2])
    for (i,c) in enumerate(classes)
        scatter!(p, x1Vec[yPredsVec .== c], x2Vec[yPredsVec .== c], 
            color = colors[2*i-1], markersize = markerLarge, label = "")
    end

    # Plot the labelled training data as dark small points
    for (i,c) in enumerate(classes)
        scatter!(p, x1[y .== c], x2[y .== c], color = colors[2*i], 
            markersize = markerSmall, label = c)
    end

    return p
end

export unpickle, invvech, invvech_byrow, CovMatEquiCorr, Cov2Corr, plotFcnGrid, plotClassifier2D

end
