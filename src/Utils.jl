module Utils

# Exporting
using Plots, LinearAlgebra, Distributions, Statistics, DataFrames
using LaTeXStrings, KernelDensity
using PDMats

include("PlotSettings.jl") # Color schemes and default plot settings
export mvcolors

include("Distr.jl") # some extra distributions
export ScaledInverseChiSq, TDist, NormalInverseChisq
export GaussianCopula
export PGDistOneParam
export SimDirProcess

include("Bayes.jl") # Bayesian inference utilities
export HPDregions

include("Misc.jl") # Miscellaneous utilities
export quantile_multidim, find_min_matrix, find_max_matrix, ConstructOptimalSubplot

include("LinAlgMisc.jl")   
export invvech, invvech_byrow, CovMatEquiCorr, Cov2Corr

include("PlotUtils.jl") 
export plotFcnGrid, plotClassifier2D, plot_braces!

end 