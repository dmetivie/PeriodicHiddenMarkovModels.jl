# PeriodicHiddenMarkovModels

This package is an extension of the package [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl) that originally define, use, fit Hidden Markov Models.
The extension adds the subtype `PeriodicHMM` to the type `HMMBase.AbstractHMM` that deals with non-constant transition matrix `A(t)` and emission distribution `B(t)`.
