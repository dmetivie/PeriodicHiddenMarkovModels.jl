module PeriodicHiddenMarkovModels

using Distributions
using HMMBase
# Write your package code here.

using Base: OneTo
using ArgCheck
using Random: AbstractRNG, GLOBAL_RNG

import Base: ==, copy, rand, size
import Distributions: fit_mle, loglikelihood

export
    # periodichmm.jl
    copy,
    rand,
    size,
    permute,
    # messages.jl
    forward,
    backward,
    posteriors,
    # likelihoods.jl
    likelihood,
    loglikelihood,
    loglikelihoods,
    loglikelihoods,
    # mle.jl
    fit_mle,
    # viterbi.jl
    viterbi

include("periodichmm.jl")
include("mle.jl")
include("messages.jl")
include("viterbi.jl")
include("likelihoods.jl")

end