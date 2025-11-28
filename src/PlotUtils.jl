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


""" 
    plot_braces!(x, y, width, height, up = true, horizontal = true) 

Plots curly braces with tip at point (`x`,`y`) with braces total width of `width` and `height`. Accepts additional keywords arguments for plot styling. If `up` is true, the braces point upward. If `horizontal` = false, the braces are vertical.

# Examples
```julia-repl
julia> plot(-2π:0.01:2π, sin.(-2π:0.01:2π))
julia> plot_braces!(π/2, 1, 2, 0.1; lw = 1, color = :black) 
julia> annotate!(π/2, 1.15, text(L"f(1.57) = 1", :black, :middle, 8))  
```
""" 
function plot_braces!(x, y, width, height, up = true, horizontal = true; plotSettings...)

    if !up
        height = -height
    end
    if !horizontal
        width = -width
    end
    if horizontal
        # left brace
        bezxl = x .- [width/2, width/2, 0, 0]
        bezyl = y .+ [height, 0, height, 0]
        # right brace
        bezxr = x .+ [0, 0, width/2, width/2]
        bezyr = y .+ [0, height, 0, height]
    else
        # upper brace
        bezxl = x .+ [height, 0, height, 0]
        bezyl = y .+ [width/2, width/2, 0, 0]
        # lower brace
        bezxr = x .+ [0, height, 0, height]
        bezyr = y .- [0, 0, width/2, width/2]
    end

    curves!(bezxl, bezyl; plotSettings...)
    curves!(bezxr, bezyr;  plotSettings...)

end