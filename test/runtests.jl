using PeriodicHiddenMarkovModels
using Test
using HMMBase, Distributions
using Random

Random.seed!(2022)

dist = [Normal, Normal]
K = length(dist)
T = 1
N = 5_000
Q = zeros(K, K, T)
Q[1, 1, :] = [0.8 for t in 1:T]
Q[1, 2, :] = [0.2 for t in 1:T]
Q[2, 2, :] = [0.4 for t in 1:T]
Q[2, 1, :] = [0.6 for t in 1:T]

ν = Matrix{Distribution{Univariate,Continuous}}(undef, (K, T))
for i in 1:K
    ν[i, :] = [dist[i](4 * (i - 1), 4 * (i - 1) + 1) for t in 1:T]
end

ν_guess = Matrix{Distribution{Univariate,Continuous}}(undef, (K, T))
for i in 1:K
    ν_guess[i, :] = [dist[i](i + 1 / 2, 3i + cos(2π / T * (t - i / 2 + 1))^2) for t in 1:T]
end

Q_guess = zeros(K, K, T)
Q_guess[1, 1, :] = [0.65 for t in 1:T]
Q_guess[1, 2, :] = [0.35 for t in 1:T]
Q_guess[2, 2, :] = [0.45 for t in 1:T]
Q_guess[2, 1, :] = [0.55 for t in 1:T]

@testset "Basic functions" begin
    hmm = PeriodicHMM(Q, ν)

    global z, y = rand(hmm, N, seq=true)

    hmm_guess0 = PeriodicHMM(Q_guess, ν_guess)
    global hmm_fit, hist = fit_mle(hmm_guess0, y, display=:none, robust=true, maxiter=1000)

    @test hmm_fit.A ≈ Q rtol = 5e-2
    # test parameter value (array of tuple) 
    # https://discourse.julialang.org/t/isapprox-for-array-of-tuples/18527/2
    @test isapprox(collect.(params.(hmm_fit.B)), collect.(params.(ν)), rtol=5e-2)
    # or
    # all(isapprox.(flatten(params.(hmm_fit.B)), flatten(params.(ν)), rtol=5e-2))
end

# Not tested because it belongs to HMMBase.jl
hmm_base = HMM(dropdims(Q, dims=ndims(Q)), dropdims(ν, dims=ndims(ν)))
hmm_base_guess0 = HMM(dropdims(Q_guess, dims=ndims(Q_guess)), dropdims(ν_guess, dims=ndims(ν_guess)))
hmm_base_fit, hist_base = fit_mle(hmm_base_guess0, y, display=:none, robust=true, maxiter=1000)

@testset "Comparing with HMMBase in trivial case T = 1 (aperiodic)" begin
    @test hist.logtots == hist_base.logtots
    @test hmm_fit.a == hmm_base_fit.a
    @test hmm_fit.A[:, :, 1] == hmm_base_fit.A
    @test hmm_fit.B[:, 1] == hmm_base_fit.B
end

@testset "Periodic example" begin
    K = 2
    T = 10
    N = 50_000
    Q = zeros(K, K, T)
    Q[1, 1, :] = [0.25 + 0.1 + 0.5cos(2π / T * t + 1)^2 for t in 1:T]
    Q[1, 2, :] = [0.25 - 0.1 + 0.5sin(2π / T * t + 1)^2 for t in 1:T]
    Q[2, 2, :] = [0.25 + 0.2 + 0.5cos(2π / T * (t - T / 3))^2 for t in 1:T]
    Q[2, 1, :] = [0.25 - 0.2 + 0.5sin(2π / T * (t - T / 3))^2 for t in 1:T]

    dist = [Normal for i in 1:K]
    ν = [dist[i](2i * cos(2π * t / T), i + cos(2π / T * (t - i / 2 + 1))^2) for i in 1:K, t in 1:T]

    ν_guess = [dist[i](2i * cos(2π * t / T) + 0.01 * randn(), i + cos(2π / T * (t - i / 2 + 1))^2 + 0.05 * randn()) for i in 1:K, t in 1:T]
    Q_guess = copy(Q)

    ξ = rand(Uniform(0, 0.15))
    Q_guess[1, 1, :] .+= ξ
    Q_guess[1, 2, :] .-= ξ

    ξ = rand(Uniform(0, 0.05))
    Q_guess[1, 1, :] .+= ξ
    Q_guess[1, 2, :] .-= ξ

    #* PeriodicHiddenMarkovModels #####

    hmm = PeriodicHMM(Q, ν)
    y = rand(hmm, N)

    hmm_guess0 = PeriodicHMM(Q, ν)
    hmm_fit, hist = fit_mle(hmm_guess0, y, display=:none, robust=true, maxiter=1000)

    @test hmm_fit.A ≈ Q rtol = 10e-2
    @test isapprox(collect.(params.(hmm_fit.B)), collect.(params.(ν)), rtol=10e-2)
end