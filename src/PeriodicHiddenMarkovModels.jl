module PeriodicHiddenMarkovModels

using Distributions
using HMMBase
using CyclicArrays: CyclicArray
# Write your package code here.

using Base: OneTo
using ArgCheck
using Random: AbstractRNG, GLOBAL_RNG

import Base: ==, copy, rand, size
import Distributions: fit_mle

export
    # periodichmm.jl
    PeriodicHMM,
    copy,
    rand,
    size,
    permute,
    # messages.jl
    forward,
    backward,
    posteriors,
    # likelihoods.jl
    loglikelihoods,
    likelihoods,
    # mle.jl
    fit_mle,
    # viterbi.jl
    viterbi

for fname in ["periodichmm.jl", "mle.jl", "messages.jl",
    "viterbi.jl", "likelihoods.jl"], foldername in ["periodic"]
    include(joinpath(foldername, fname))
end


end