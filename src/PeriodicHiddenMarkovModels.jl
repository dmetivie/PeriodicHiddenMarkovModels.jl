module PeriodicHiddenMarkovModels

using Distributions
# using HMMBase
# using HMMBase: posteriors!, vec_maximum, EMHistory, update_a!, isprobvec # function not exported by default by HHMBase

using Base: OneTo
using ArgCheck
using Random: AbstractRNG, GLOBAL_RNG

import Base: ==, copy, size
# import HMMBase: rand, fit_mle!, viterbi, viterbilog! # viterbi! is not in HHMBase (anymore?)
#? should I import rand from Base or HMMBase? (I will be elaborating on the rand method devlopped in HHMBase)
import Distributions: fit_mle, rand

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
    viterbi,
    # utilities.jl
    n_to_t,
    randPeriodicHMM

include("utilities.jl")
for fname in ["periodichmm.jl", "mle.jl", "likelihoods.jl"], foldername in ["periodic"]
    include(joinpath(foldername, fname))
end
include("messages.jl")
include("viterbi.jl")


end