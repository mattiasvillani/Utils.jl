module Utils

# Exporting
using Reexport
@reexport using Plots, LinearAlgebra, Distributions, Statistics, DataFrames, RCall
@reexport using LaTeXStrings
using PyCall, PDMats, QuadGK, Roots


include("PlotSettings.jl") # Color schemes and default plot settings
export colors

include("Distr.jl") # some extra distributions
export ScaledInverseChiSq, TDist, NormalInverseChisq, SimDirProcess
export ZDist, GaussianCopula

include("Bayes.jl") # Bayesian inference utilities, e.g. posterior samplers.
export finiteNewtonMH

include("DataWrangling.jl") # Data wrangling utilities
export unpickle

include("Misc.jl") # Miscellaneous utilities

include("LinAlgMisc.jl")   
export invvech, invvech_byrow, CovMatEquiCorr, Cov2Corr

include("PlotUtils.jl") 
export plotFcnGrid, plotClassifier2D

end