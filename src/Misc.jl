""" 
    subscript(i::Integer) 

Set up string with integer `i` as subscript (for printing). 

From https://stackoverflow.com/questions/46671965/printing-variable-subscripts-in-julia 

# Examples
```julia-repl
julia> println("Studio"*subscript(45))
Studio₄₅
```
""" 
subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))


""" 
    pad_digits(x) 

Returns a vector with string representations of the numbers in x such that they have the same number of digits, including trailing zeros.

Useful when printing numbers in latex tables and annotating figures.

# Examples
```julia-repl
julia> pad_digits([1.21, 13.3, 123.456])
3-element Vector{String}:
 "1.210"
 "13.300"
 "123.456"
```
""" 
function pad_digits(x)
    strs = string.(x)
    digits = [length(str[(findfirst.('.', str) + 1):end]) for str in strs]
    maxDigits = maximum(digits)
    lengthNumber = [length(str) for str in strs]
    for i = 1:length(strs)
        strs[i] = strs[i]*repeat("0",maxDigits-digits[i])
    end
    return strs
end

""" 
    find_min_matrix(matrix, k) 

Returns the Cartesian indices of the k:th smallest values in `matrix`. 

# Examples
```julia-repl
julia> A = [10 8 12; 5 4 9; 3 6 2]
julia> find_min_matrix(A, 3)
```
""" 
function find_min_matrix(matrix, k)
    flattened_matrix = vec(matrix)
    sorted_indices = sortperm(flattened_matrix)
    return CartesianIndices(size(matrix))[sorted_indices[1:k]]
end


""" 
    find_max_matrix(matrix, k) 

Returns the Cartesian indices of the k:th largest values in `matrix`. 

# Examples
```julia-repl
julia> A = [10 8 12; 5 4 9; 3 6 2]
julia> find_max_matrix(A, 3)
```
""" 
function find_max_matrix(matrix, k)
    flattened_matrix = vec(matrix)
    sorted_indices = sortperm(flattened_matrix, rev=true)
    return CartesianIndices(size(matrix))[sorted_indices[1:k]]
end


function ConstructOptimalSubplot(NumberOfPlots)

    # Given a number of plots, this function determines the 'optimal' number of
    # rows and columns of the subplot.

    #TODO: also output the optimal size of the figure
    
    if NumberOfPlots == 1 return 1, 1 end
    if NumberOfPlots == 2 return 2, 1 end
    if NumberOfPlots == 3 return 2, 2 end
    if NumberOfPlots == 4 return 2, 2 end
    if NumberOfPlots == 5 return 3, 3 end
    if NumberOfPlots == 6 return 3, 3 end
    if NumberOfPlots == 7 return 3, 3 end
    if NumberOfPlots == 8 return 3, 3 end
    if NumberOfPlots == 9 return 3, 3 end
    if NumberOfPlots == 10 return 3, 4 end
    if NumberOfPlots == 11 return 3, 4 end
    if NumberOfPlots == 12 return 3, 4 end
    if NumberOfPlots == 13 return 4, 4 end
    if NumberOfPlots == 14 return 4, 4 end
    if NumberOfPlots == 15 return 4, 4 end
    if NumberOfPlots == 16 return 4, 4 end
    if NumberOfPlots == 14 return 4, 4 end
    return ceil(sqrt(NumberOfPlots)), ceil(sqrt(NumberOfPlots))

end

quantileMultiDim(A, p; dims, kwargs...) = mapslices(x -> quantile(x, p; kwargs...), A; dims)