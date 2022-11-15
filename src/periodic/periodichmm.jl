"""
    PeriodicHMM([a, ]A, B) -> PeriodicHMM

Build an PeriodicHMM with transition matrix `A(t)` and observation distributions `B(t)`.  
If the initial state distribution `a` is not specified, a uniform distribution is assumed. 

Observations distributions can be of different types (for example `Normal` and `Exponential`),  
but they must be of the same dimension.

Alternatively, `B(t)` can be an emission matrix where `B[i,j,t]` is the probability of observing symbol `j` in state `i`.

**Arguments**
- `a::AbstractVector{T}`: initial probabilities vector.
- `A::AbstractArray{T,3}`: transition matrix.
- `B::AbstractMatrix{<:Distribution{F}}`: observations distributions.
- or `B::AbstractMatrix`: emission matrix.

**Example**
```julia
using Distributions, PeriodicHMM
# from distributions
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
# from an emission matrix
hmm = HMM([0.9 0.1; 0.1 0.9], [0. 0.5 0.5; 0.25 0.25 0.5])
```
"""
abstract type AbstractPeriodicHMM{F<:VariateForm} <: AbstractHMM{F} end

struct PeriodicHMM{F,T} <: AbstractPeriodicHMM{F}
    a::Vector{T}
    A::Array{T,3}
    B::Matrix{<:Distribution{F}}
    PeriodicHMM{F,T}(a, A, B) where {F,T} = assert_hmm(a, A, B) && new(a, A, B)
end

PeriodicHMM(a::AbstractVector{T}, A::AbstractArray{T,3}, B::AbstractMatrix{<:Distribution{F}}) where {F,T} =
    PeriodicHMM{F,T}(a, A, B)

PeriodicHMM(A::AbstractArray{T,3}, B::AbstractMatrix{<:Distribution{F}}) where {F,T} =
    PeriodicHMM{F,T}(ones(size(A, 1)) ./ size(A, 1), A, B)

function PeriodicHMM(a::AbstractVector{T}, A::AbstractArray{T,3}, B::AbstractArray{T,3}) where {T}
    ν = map((i, t) -> Categorical(B[i, :, t]), Iterators.product(OneTo(size(B, 1)), OneTo(size(B, 3))))
    PeriodicHMM{Univariate,T}(a, A, ν)
end

function assert_hmm(hmm::PeriodicHMM)
    assert_hmm(hmm.a, hmm.A, hmm.B)
end

"""
    assert_hmm(a, A, B)

Throw an `ArgumentError` if the initial state distribution and the transition matrix rows does not sum to 1,
and if the observation distributions do not have the same dimensions.
"""
function assert_hmm(a::AbstractVector, A::AbstractArray{T,3} where {T}, B::AbstractMatrix{<:Distribution})
    @argcheck isprobvec(a)
    @argcheck all(t -> istransmat(A[:, :, t]), OneTo(size(A, 3))) ArgumentError("All transition matrice A(t) for all t must be transition matrix")
    @argcheck all(length.(B) .== length(B[1])) ArgumentError("All distributions must have the same dimensions")
    @argcheck length(a) == size(A, 1) == size(B, 1) ArgumentError("Number of transition rates must match length of chain")
    @argcheck size(A, 3) == size(B, 2) ArgumentError("Period length must be the same for transition matrix and distribution")
    return true
end

rand(hmm::AbstractHMM, n2t::AbstractVector{<:Integer}; kwargs...) = rand(GLOBAL_RNG, hmm, n2t; kwargs...)

rand(rng::AbstractRNG, hmm::AbstractPeriodicHMM, N::Integer; kwargs...) = rand(rng, hmm, n_to_t(N, size(hmm, 3)); kwargs...)    

function rand(
    rng::AbstractRNG,
    hmm::AbstractPeriodicHMM,
    n2t::AbstractVector{<:Integer};
    z_ini=rand(rng, Categorical(hmm.a))::Integer,
    seq=false,
    kwargs...
)
    N = length(n2t)
    z = zeros(Int, N)
    (N >= 1) && (z[1] = z_ini)
    for n = 2:N
        tₙ₋₁ = n2t[n-1] # periodic t-1
        z[n] = rand(rng, Categorical(hmm.A[z[n-1], :, tₙ₋₁]))
    end
    y = rand(rng, hmm, z, n2t; kwargs...)
    return seq ? (z, y) : y
end

function rand(rng::AbstractRNG, 
    hmm::PeriodicHMM{Univariate}, 
    z::AbstractVector{<:Integer}, 
    n2t::AbstractVector{<:Integer}
)
    y = Vector{eltype(eltype(hmm.B))}(undef, length(z)) #! Change compare to HHMBase where Vector{Float64} is used
    for n in eachindex(z)
        t = n2t[n] # periodic t
        y[n] = rand(rng, hmm.B[z[n], t])
    end
    return y
end

function rand(
    rng::AbstractRNG,
    hmm::PeriodicHMM{Multivariate},
    z::AbstractVector{<:Integer};
    n2t::AbstractVector{<:Integer}
)
    y = Matrix{eltype(eltype(hmm.B))}(undef, length(z), size(hmm, 2))
    for n in eachindex(z)
        t = n2t[n] # periodic t
        y[n, :] = rand(rng, hmm.B[z[n], t])
    end
    return y
end

"""
    size(hmm, [dim]) -> Int | Tuple
Return the number of states in `hmm`, the dimension of the observations and the length of the chain.
"""
size(hmm::PeriodicHMM, dim=:) = (size(hmm.B, 1), length(hmm.B[1]), size(hmm.B, 2))[dim]

copy(hmm::PeriodicHMM) = PeriodicHMM(copy(hmm.a), copy(hmm.A), copy(hmm.B))

"""
    permute(hmm, perm) -> HMM

Permute the states of `hmm` according to `perm`.

**Arguments**

- `perm::Vector{<:Integer}`: permutation of the states.

"""
function permute(hmm::AbstractHMM, perm::Vector{<:Integer})
    @argcheck length(perm) == length(hmm.a) == size(hmm.A, 1) == size(hmm.B, 1)
    a = hmm.a[perm]
    B = copy(hmm.B)
    T = size(B, 3)
    A = zeros(size(hmm.A))
    for t in 1:T
        B[:, t] = hmm.B[perm, t]
        for i = 1:size(A, 1), j = 1:size(A, 2)
            A[i, j, t] = hmm.A[perm[i], perm[j], t]
        end
    end
    HMM(a, A, B)
end
#TODO add possibility to permute differently w.r.t. t, quid of hmm.a ?
# """
#     permute(hmm, perm) -> HMM

# Permute the states of `hmm` according to `perm`.

# Arguments

# - `perm::Matrix{<:Integer}`: permutation of the states can be different for each time `t`.

# """
# function permute(hmm::AbstractHMM, perm::Matrix{<:Integer})
#     @argcheck size(perm, 1) == length(hmm.a) == size(hmm.A, 1) == size(hmm.B, 1)
#     @argcheck size(perm, 1)
#     a = hmm.a[perm]
#     B = copy(hmm.B)
#     T = size(B, 3)
#     A = zeros(size(hmm.A))
#     for t in 1:T
#         B[:, t] = hmm.B[perm, t]
#         for i = 1:size(A, 1), j = 1:size(A, 2)
#             A[i, j, t] = hmm.A[perm[i], perm[j], t]
#         end
#     end
#     HMM(a, A, B)
# end
# function nparams(hmm::PeriodicHMM)
#     (length(hmm.a) - 1) + (size(hmm.A, 1) * size(hmm.A, 2) - size(hmm.A, 1)) * size(hmm.A, 3)
# end