struct PeriodicHMM{T<:Number,D,L,V<:AbstractVector{T},AM<:AbstractMatrix{T},AV<:AbstractVector{D}} <: AbstractHMM
    init::V
    trans::NTuple{L,AM}
    dists::NTuple{L,AV}
    function PeriodicHMM(init, trans_per, dists_per) 
        L = length(trans_per)
        @assert L == length(dists_per)
        hmm = new{eltype(init), eltype(eltype((dists_per))), L, typeof(init), eltype(typeof(trans_per)), eltype(typeof(dists_per))}(init, trans_per, dists_per)
        for t in 1:L
            @argcheck HiddenMarkovModels.valid_hmm(hmm, t)
        end
        return hmm
    end
end

period(::PeriodicHMM{T,D,L}) where {T,D,L} = L

function HMMs.initialization(hmm::PeriodicHMM)
    return hmm.init
end

function HMMs.transition_matrix(hmm::PeriodicHMM, t::Integer)
    return hmm.trans[t]
end

function HMMs.obs_distributions(hmm::PeriodicHMM, t::Integer)
    return hmm.dists[t]
end