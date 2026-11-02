function StatsAPI.fit!(
    hmm::PeriodicHMM{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector,
    n2t::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
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

        # `ξ[t]` is the marginal of the step `t -> t+1`, and `forward_backward!` builds it with
        # `transition_matrix(hmm, control_seq[t + 1])`: a transition matrix is indexed by the
        # period of the step it *arrives at*. So `trans[l]` collects the `ξ[t]` whose successor
        # has period `l`, not those whose own period is `l`. The two differ by one step, and
        # grouping on the wrong one re-estimates `A(l)` from the transitions belonging to
        # `A(l+1)`, which is a decreasing likelihood rather than an error. It is invisible when
        # `L == 1`, which is why the homogeneous testset never caught it.
        # (`t2` has no successor inside the sequence, and `ξ[t2]` is zeroed by forward-backward.)
        for t in t1:(t2 - 1)
            hmm.trans[n2t[t + 1]] .+= ξ[t]
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
