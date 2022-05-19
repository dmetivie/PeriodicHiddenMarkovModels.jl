module PeriodicHiddenMarkovModels

using Distributions
using HMMBase
using CyclicArrays: CyclicArray
using HMMBase: posteriors!, vec_maximum, EMHistory, update_a!, isprobvec # function not exported by default by HHMBase

using Base: OneTo
using ArgCheck
using Random: AbstractRNG, GLOBAL_RNG

import Base: ==, copy, rand, size
import HMMBase: fit_mle!

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