module PeriodicHiddenMarkovModels

using Distributions
using HMMBase
using HMMBase: posteriors!, vec_maximum, EMHistory, update_a!, isprobvec # function not exported by default by HHMBase

using Base: OneTo
using ArgCheck
using Random: AbstractRNG, GLOBAL_RNG

import Base: ==, copy, rand, size
import HMMBase: fit_mle!

export
    # periodichmm.jl
    AbstractPeriodicHMM,
    PeriodicHMM,
    copy,
    rand,
    size,
    permute,
    # messages.jl
    forward,
    backward,
    posteriors,
    # mle.jl
    fit_mle

include("utilities.jl")
for fname in ["periodichmm.jl", "mle.jl", "likelihoods.jl"], foldername in ["periodic"]
    include(joinpath(foldername, fname))
end
include("messages.jl")
include("viterbi.jl")


end