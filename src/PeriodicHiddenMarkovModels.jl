module PeriodicHiddenMarkovModels

using ArgCheck: @argcheck
import HiddenMarkovModels as HMMs
import StatsAPI
using HiddenMarkovModels
using HiddenMarkovModels: AbstractVectorOrNTuple

export
    # utilities.jl
    n_to_t,
    PeriodicHMM,
    transition_matrix,
    obs_distributions

include("utilities.jl")
include("periodichmm.jl")
include("fit.jl")

end