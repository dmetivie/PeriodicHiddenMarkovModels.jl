using PeriodicHiddenMarkovModels
using Test
using HiddenMarkovModels, Distributions
using Random
using StableRNGs


@testset "Comparing with HiddenMarkovModels.jl in trivial case T = 1 (homogeneous)" begin
    Random.seed!(2022)
    dist = [Normal, Normal]
    K = length(dist)
    T = 1
    N = 4_786
    init = [1 / 2, 1 / 2]

    Q = zeros(K, K, T)
    Q[1, 1, :] = [0.8]
    Q[1, 2, :] = [0.2]
    Q[2, 2, :] = [0.4]
    Q[2, 1, :] = [0.6]
    ν = Matrix{Distribution{Univariate,Continuous}}(undef, (K, T))
    for i in 1:K
        ν[i, :] = [dist[i](4 * (i - 1), 4 * (i - 1) + 1) for t in 1:T]
    end

    trans_per = tuple(eachslice(Q; dims=3)...)
    dists_per = tuple(eachcol(ν)...)

    hmm = PeriodicHMM(init, trans_per, dists_per)

    @inferred PeriodicHiddenMarkovModels.transition_matrix(hmm, 1)
    @inferred PeriodicHiddenMarkovModels.obs_distributions(hmm, 1)

    Q_guess = zeros(K, K, T)
    Q_guess[1, 1, :] = [0.65]
    Q_guess[1, 2, :] = [0.35]
    Q_guess[2, 2, :] = [0.45]
    Q_guess[2, 1, :] = [0.55]
    ν_guess = Matrix{Distribution{Univariate,Continuous}}(undef, (K, T))
    for i in 1:K
        ν_guess[i, :] = [dist[i](i + 1 / 2, 3i + cos(2π / T * (t - i / 2 + 1))^2) for t in 1:T]
    end

    trans_per_guess = tuple(eachslice(Q_guess; dims=3)...)
    dists_per_guess = tuple(eachcol(ν_guess)...)

    hmm_guess = PeriodicHMM(init, trans_per_guess, dists_per_guess)

    ## Basic functions

    n2t = fill(1, N)

    state_seq, obs_seq = rand(hmm, n2t)

    hmm_fit, loglikelihood_evolution = baum_welch(hmm_guess, obs_seq, n2t)

    # test parameter value (array of tuple) 
    # https://discourse.julialang.org/t/isapprox-for-array-of-tuples/18527/2

    # HiddenMarkovModels.jl is a well tested package
    hmm_0 = HiddenMarkovModels.HMM([1 / 2, 1 / 2], dropdims(Q, dims=ndims(Q)), dropdims(ν, dims=ndims(ν)))
    hmm_0_guess = HiddenMarkovModels.HMM([1 / 2, 1 / 2], dropdims(Q_guess, dims=ndims(Q_guess)), dropdims(ν_guess, dims=ndims(ν_guess)))
    hmm_0_fit, loglikelihood_evolution_0 = baum_welch(hmm_0_guess, obs_seq)

    @test loglikelihood_evolution ≈ loglikelihood_evolution_0 rtol = 1e-2

    @test transition_matrix(hmm_fit, 1) ≈ Q rtol = 5e-2
    @test transition_matrix(hmm_fit, 1) ≈ transition_matrix(hmm_0_fit) rtol = 5e-2
    @test isapprox(collect.(Distributions.params.(obs_distributions(hmm_fit, 1))), collect.(Distributions.params.(ν)), rtol=5e-2)
    @test isapprox(collect.(Distributions.params.(obs_distributions(hmm_fit, 1))), collect.(Distributions.params.(obs_distributions(hmm_0_fit))), rtol=5e-2)
end


@testset "Periodic example" begin
    Random.seed!(2022)
    K = 2
    T = 10
    N = 49_586
    Q = zeros(K, K, T)
    Q[1, 1, :] = [0.25 + 0.1 + 0.5cos(2π / T * t + 1)^2 for t in 1:T]
    Q[1, 2, :] = [0.25 - 0.1 + 0.5sin(2π / T * t + 1)^2 for t in 1:T]
    Q[2, 2, :] = [0.25 + 0.2 + 0.5cos(2π / T * (t - T / 3))^2 for t in 1:T]
    Q[2, 1, :] = [0.25 - 0.2 + 0.5sin(2π / T * (t - T / 3))^2 for t in 1:T]

    dist = [Normal for i in 1:K]
    ν = [dist[i](2i * cos(2π * t / T), i + cos(2π / T * (t - i / 2 + 1))^2) for i in 1:K, t in 1:T]

    ν_guess = [dist[i](2i * cos(2π * t / T) + 0.01 * randn(), i + cos(2π / T * (t - i / 2 + 1))^2 + 0.05 * randn()) for i in 1:K, t in 1:T]
    Q_guess = copy(Q)

    ξ = rand(Uniform(0, 0.1))
    Q_guess[1, 1, :] .+= ξ
    Q_guess[1, 2, :] .-= ξ

    ξ = rand(Uniform(0, 0.05))
    Q_guess[1, 1, :] .+= ξ
    Q_guess[1, 2, :] .-= ξ

    init = [1 / 2, 1 / 2]
    #* PeriodicHiddenMarkovModels #####
    trans_per = tuple(eachslice(Q; dims=3)...)
    dists_per = tuple(eachcol(ν)...)

    hmm = PeriodicHMM(init, trans_per, dists_per)

    n2t = n_to_t(N, T)
    state_seq, obs_seq = rand(hmm, n2t)

    hmm_guess = PeriodicHMM(init, tuple(eachslice(Q_guess; dims=3)...), tuple(eachcol(ν_guess)...))

    hmm_fit, hist = baum_welch(hmm_guess, obs_seq, n2t)
    @test all(diff(hist) .> 0)
    for t in 1:T
        @test transition_matrix(hmm_fit, t) ≈ transition_matrix(hmm, t) rtol = 15e-2
    end

    @test isapprox(collect.(Distributions.params.(obs_distributions(hmm_fit, 1))), collect.(Distributions.params.(obs_distributions(hmm, 1))), rtol=5e-2)
end

# begin
#     p = [plot(xlabel="t") for i in 1:K]
#     for i in 1:K, j in 1:K
#         plot!(p[i], 1:T, [transition_matrix(hmm, t)[i, j] for t in 1:T], label="Q_{%$(i)\to %$(j)}", c=j)
#         plot!(p[i], 1:T, [transition_matrix(hmm_fit, t)[i, j] for t in 1:T], label="hat{Q}_{%$(i)\to %$(j)}", c=j, s=:dash)
#     end
#     plot(p..., size=(1000, 500))
# end

# begin
#     p = [plot(xlabel="t", title="K = %$(i)") for i in 1:K]
#     for i in 1:K
#         plot!(p[i], 1:T, mean.(ν[i, :]), label="mean", c=1)
#         plot!(p[i], 1:T, mean.([obs_distributions(hmm_fit, t)[i] for t in 1:T]), label="Estimated mean", c=1, s=:dash)
#         plot!(p[i], 1:T, std.(ν[i, :]), label="std", c=2)
#         plot!(p[i], 1:T, std.([obs_distributions(hmm_fit, t)[i] for t in 1:T]), label="Estimated std", c=2, s=:dash)
#     end
#     plot(p..., size=(1000, 500))
# end

@testset "HiddentMarlovModel Tuto time" begin
    rng = StableRNG(63);
    init = [0.6, 0.4]
    trans_per = ([0.7 0.3; 0.2 0.8], [0.3 0.7; 0.8 0.2])
    dists_per = ([Normal(-1.0), Normal(-2.0)], [Normal(+1.0), Normal(+2.0)])
    T = length(trans_per)
    hmm = PeriodicHMM(init, trans_per, dists_per)

    control_seq = repeat(1:2, 5)
    state_seq, obs_seq = rand(hmm, control_seq)

    control_seqs = [repeat(1:2, rand(100:200)) for k in 1:1000]
    obs_seqs = [rand(hmm, control_seqs[k]).obs_seq for k in eachindex(control_seqs)]

    obs_seq = reduce(vcat, obs_seqs)
    control_seq = reduce(vcat, control_seqs)
    seq_ends = cumsum(length.(obs_seqs))

    best_state_seq, _ = viterbi(hmm, obs_seq, control_seq; seq_ends)

    vcat(obs_seq', best_state_seq')


    init_guess = [0.7, 0.3]
    trans_per_guess = ([0.6 0.4; 0.3 0.7], [0.4 0.6; 0.7 0.3])
    dists_per_guess = ([Normal(-1.1), Normal(-2.1)], [Normal(+1.1), Normal(+2.1)])
    hmm_guess = PeriodicHMM(init_guess, trans_per_guess, dists_per_guess)

    hmm_est, loglikelihood_evolution = baum_welch(hmm_guess, obs_seq, control_seq; seq_ends)
    first(loglikelihood_evolution), last(loglikelihood_evolution)

    for t in 1:T
        @test transition_matrix(hmm_est, t) ≈ transition_matrix(hmm, t) rtol = 15e-2
    end

    map(mean, hcat(obs_distributions(hmm_est, 1), obs_distributions(hmm, 1)))

    map(mean, hcat(obs_distributions(hmm_est, 2), obs_distributions(hmm, 2)))
    #TODO: complete the test
end

#TODO: add multi sequences test