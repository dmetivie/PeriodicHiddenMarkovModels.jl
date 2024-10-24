function StatsAPI.fit!(
    hmm::PeriodicHMM{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector,
    n2t::AbstractVector;
    seq_ends::AbstractVector{Int},
) where {T}
    (; γ, ξ) = fb_storage
    L, N = period(hmm), length(hmm)

    hmm.init .= zero(T)
    for l in 1:L
        hmm.trans[l] .= zero(T)
    end
    @views for k in eachindex(seq_ends)
        t1, t2 = HMMs.seq_limits(seq_ends, k)
        hmm.init .+= γ[:, t1]
        n2t_k = n2t[t1:t2]

        for l in 1:L
            hmm.trans[l] .+= sum(ξ[first(parentindices(n2t_k))[findall(n2t_k .== l)]])
        end
    end
    hmm.init ./= sum(hmm.init)
    for l in 1:L, row in eachrow(hmm.trans[l])
        row ./= sum(row)
    end
    for l in 1:L
        times_l = Int[]
        @views for k in eachindex(seq_ends)
            t1, t2 = HMMs.seq_limits(seq_ends, k)
            n2t_k = n2t[t1:t2]
            append!(times_l, first(parentindices(n2t_k))[findall(n2t_k .== l)])
        end
        for i in 1:N
            HMMs.fit_in_sequence!(hmm.dists[l], i, obs_seq[times_l], γ[i, times_l])
        end
    end
    for l in 1:L
        @assert HMMs.valid_hmm(hmm, l)
    end
    return nothing
end
