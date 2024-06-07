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

#TODO
# HMMs.valid_hmm

# struct PeriodicHMM{T<:Number,D,L,AM<:AbstractMatrix{T},AV<:AbstractVector{D}} <: AbstractHMM
#     init::Vector{T}
#     trans_per::NTuple{L,AM}
#     dists_per::NTuple{L,AV}
#     function PeriodicHMM2(init, trans_per, dists_per) 
#         L = length(trans_per)
#         @assert L == length(dists_per)
#         new{eltype(init), eltype(eltype(trans_per)), eltype(eltype(dists_per)), L, typeof(AM), typeof(AV)}(init, trans_per, dists_per)
#     end
# end

# struct PeriodicHMM{T<:Number,D,L,AM<:AbstractMatrix{T},AV<:AbstractVector{D}} <: AbstractHMM
#     init::Vector{T}
#     trans_per::NTM#NTuple{L,AM}
#     dists_per::NTD#NTuple{L,AV}
#     function PeriodicHMM2(init, trans_per, dists_per) 
#         L = length(trans_per)
#         @assert L == length(dists_per)
#         new{eltype(init), eltype(eltype(trans_per)), eltype(eltype(dists_per)), L, typeof(AM), typeof(AV)}(init, trans_per, dists_per)
#     end
# end

# struct PeriodicHMM{
#     V<:AbstractVector,
#     M<:AbstractMatrix,
#     VD<:AbstractVector,
#     L<:Integer
#     # Vl<:AbstractVector,
#     # Ml<:AbstractMatrix,
# } <: AbstractHMM
#     "initial state probabilities"
#     init::V
#     "state transition probabilities"
#     trans::NTuple{L,M}
#     "observation distributions"
#     dists::NTuple{L,VD}
#     L::Integer
#     # "logarithms of initial state probabilities"
#     # loginit::Vl
#     # "logarithms of state transition probabilities"
#     # logtrans::Ml

#     function PeriodicHMM(init::AbstractVector, trans::NTuple{L,AM}, dists::NTuple{L,AV}) where {L,AM,AV}
#         # log_init = elementwise_log(init)
#         # log_trans = elementwise_log(trans)

#         hmm = new{
#             typeof(init),AM,AV,typeof(L)#,typeof(log_init),typeof(log_trans)
#         }(
#             init, trans, dists, L#, log_init, log_trans
#         )
#         @argcheck all([valid_hmm(hmm, t) for t in 1:L])
#         return hmm
#     end
# end

# struct PeriodicHMM{T<:Number,D,L,AM<:AbstractMatrix{T},AV<:AbstractVector{D}} <: AbstractHMM
#     init::Vector{T}
#     trans_per::NTuple{L,AM}
#     dists_per::NTuple{L,AV}
# end

# struct PeriodicHMM{T<:Number,D,L} <: AbstractHMM
#     init::Vector{T}
#     trans_per::NTuple{L,<:AbstractMatrix{T}}
#     dists_per::NTuple{L,<:AbstractVector{D}}
# end
