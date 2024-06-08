# PeriodicHiddenMarkovModels

This package is an extension of the package [HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl) that defines a lot of the Hidden Markov Models tools (Baum Welch, Viterbi, etc.).
The extension adds the subtype `PeriodicHMM` to the type `HiddenMarkovModels.AbstractHMM` that deals with non-constant transition matrix `A(t)` and emission distribution `B(t)`.

Before v0.2 this package depended on the no longer maintained [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl). It is now inspired by this [Tutorial](https://gdalle.github.io/HiddenMarkovModels.jl/dev/examples/temporal/) of the [HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl) model, meaning that it should benefit from all the good stuff there. In particular mutli-sequence support.
The major notable difference is that `PeriodicHMM` here do not have to be periodic, they can be completely time inhomogeneous.
This is controlled by the `n2t` vector which indicates the correspondance between observation `n` and the associated element of the HMM `t`. See example bellow.

## Simple example

```julia
using PeriodicHiddenMarkovModels
using Distributions
using Random
```

### Creating matrix

```julia
Random.seed!(2022)
K = 2 # Number of Hidden states
T = 10 # Period
N = 49_586 # Length of observation
Q = zeros(K, K, T)
Q[1, 1, :] = [0.25 + 0.1 + 0.5cos(2π / T * t + 1)^2 for t in 1:T]
Q[1, 2, :] = [0.25 - 0.1 + 0.5sin(2π / T * t + 1)^2 for t in 1:T]
Q[2, 2, :] = [0.25 + 0.2 + 0.5cos(2π / T * (t - T / 3))^2 for t in 1:T]
Q[2, 1, :] = [0.25 - 0.2 + 0.5sin(2π / T * (t - T / 3))^2 for t in 1:T]

dist = [Normal for i in 1:K]
ν = [dist[i](2i * cos(2π * t / T), i + cos(2π / T * (t - i / 2 + 1))^2) for i in 1:K, t in 1:T]

init = [1 / 2, 1 / 2]
trans_per = tuple(eachslice(Q; dims=3)...)
dists_per = tuple(eachcol(ν)...)
hmm = PeriodicHMM(init, trans_per, dists_per)   
```

### Creating guess matrix (initial condition for the EM algorithm)

Here we add noise to the true matrix (not too far to not end up in far away local minima).

```julia
ν_guess = [dist[i](2i * cos(2π * t / T) + 0.01 * randn(), i + cos(2π / T * (t - i / 2 + 1))^2 + 0.05 * randn()) for i in 1:K, t in 1:T]
Q_guess = copy(Q)

ξ = rand(Uniform(0, 0.1))
Q_guess[1, 1, :] .+= ξ
Q_guess[1, 2, :] .-= ξ

ξ = rand(Uniform(0, 0.05))
Q_guess[1, 1, :] .+= ξ
Q_guess[1, 2, :] .-= ξ
hmm_guess = PeriodicHMM(init, tuple(eachslice(Q_guess; dims=3)...), tuple(eachcol(ν_guess)...))
```

### Sampling from the HMM

The `n2t` vector of length `N` controls the correspondence between the index of the sequence `n` and `t∈[1:T]`.
The function `n_to_t(N,T)` creates a vector of length `N` and periodicity `T` but arbitrary non-periodic `n2t` are accepted.

```julia
n2t = n_to_t(N, T)
state_seq, obs_seq = rand(hmm, n2t)
```

### Fitting the HMM

```julia
hmm_fit, hist = baum_welch(hmm_guess, obs_seq, n2t)
```

### Plotting the results

```julia
using Plots, LaTeXStrings
default(fontfamily="Computer Modern", linewidth=2, label=nothing, grid=true, framestyle=:default)
```

#### Transition matrix

```julia
begin
    p = [plot(xlabel="t") for i in 1:K]
    for i in 1:K, j in 1:K
        plot!(p[i], 1:T, [transition_matrix(hmm, t)[i, j] for t in 1:T], label=L"Q_{%$(i)\to %$(j)}", c=j)
        plot!(p[i], 1:T, [transition_matrix(hmm_fit, t)[i, j] for t in 1:T], label=L"\hat{Q}_{%$(i)\to %$(j)}", c=j, s=:dash)
    end
    plot(p..., size=(1000, 500))
end
```

![Time dependent transition matrix coefficient](img/Q_estiamated.svg)

#### Emission distribution

```julia
begin
    p = [plot(xlabel="t", title=L"K = %$(i)") for i in 1:K]
    for i in 1:K
        plot!(p[i], 1:T, mean.(ν[i, :]), label="mean", c=1)
        plot!(p[i], 1:T, mean.([obs_distributions(hmm_fit, t)[i] for t in 1:T]), label="Estimated mean", c=1, s=:dash)
        plot!(p[i], 1:T, std.(ν[i, :]), label="std", c=2)
        plot!(p[i], 1:T, std.([obs_distributions(hmm_fit, t)[i] for t in 1:T]), label="Estimated std", c=2, s=:dash)
    end
    plot(p..., size=(1000, 500))
end
```

![Emission distribution parameters](img/nu_estiamated.svg)

> [!WARNING]
> As it is `fit_mle` does not enforce smoothness of hidden states with `t` i.e. because HMM are identifiable up to a relabeling nothing prevents that after fitting `ν[k=1, t=1]` and `ν[k=1, t=2]` mean the same hidden state (same for `Q` matrix).
> To enforce smoothness and identifiability (up to a global index relabeling), one can be inspired by seasonal Hidden Markov Model, see [A. Touron (2019)](https://link.springer.com/article/10.1007/s11222-019-09854-4). This is already implemented in [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl).but I plan to add this feature to `PeriodicHiddenMarkovModels.jl` soon.
