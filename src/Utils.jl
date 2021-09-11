module Utils

# Python functions import
using PyCall, LinearAlgebra, Distributions, Statistics

include("Distr.jl") # some extra distributions

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



export unpickle, invvech, invvech_byrow, plotFcnGrid

end
