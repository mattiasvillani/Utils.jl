module Utils

# Exporting
using Plots, LinearAlgebra, Distributions, Statistics, DataFrames, RCall
using LaTeXStrings, KernelDensity
using PyCall, PDMats, QuadGK, Roots


include("PlotSettings.jl") # Color schemes and default plot settings

include("Distr.jl") # some extra distributions
export ScaledInverseChiSq, TDist, NormalInverseChisq, SimDirProcess
export ZDist, GaussianCopula
export PGDistOneParam

include("Bayes.jl") # Bayesian inference utilities, e.g. posterior samplers.
export finiteNewtonMH, HPDregions

include("DataWrangling.jl") # Data wrangling utilities
export unpickle # use the pickle.jl package instead of rolling own

include("Misc.jl") # Miscellaneous utilities
export quantileMultiDim, find_min_matrix, find_max_matrix, ConstructOptimalSubplot

include("LinAlgMisc.jl")   
export invvech, invvech_byrow, CovMatEquiCorr, Cov2Corr

include("PlotUtils.jl") 
export plotFcnGrid, plotClassifier2D

end