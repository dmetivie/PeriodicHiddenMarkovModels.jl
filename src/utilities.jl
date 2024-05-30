"""
    n_to_t(N::Int, T::Int)

    This function transforms all index of the chain `n` into their periodic counterpart `t`.
"""
function n_to_t(N::Int, T::Int)
    return [repeat(1:T, N ÷ T); remaining(N - T * (N ÷ T))]
end

remaining(N::Int) = N > 0 ? range(1, length=N) : Int64[]

#TODO: change these following function with HiddenMarkovModels equivalents

# Taken from https://github.com/dmetivie/HMMBase.jl/blob/e65510335354be7637a1598a79f02788b718dd1f/src/hmm.jl#L81-L82
"""
    issquare(A) -> Bool

Return true if `A` is a square matrix.
"""
issquare(A::AbstractMatrix) = size(A, 1) == size(A, 2)

"""
    istransmat(A) -> Bool

Return true if `A` is square and its rows sums to 1.
"""
istransmat(A::AbstractMatrix) =
    issquare(A) && all([isprobvec(A[i, :]) for i = 1:size(A, 1)])

# https://github.com/dmetivie/HMMBase.jl/blob/e65510335354be7637a1598a79f02788b718dd1f/src/utilities.jl#L128-L136
function vec_maximum(v::AbstractVector)
    m = v[1]
    @inbounds for i in OneTo(length(v))
        if v[i] > m
            m = v[i]
        end
    end
    m
end

# https://github.com/dmetivie/HMMBase.jl/blob/e65510335354be7637a1598a79f02788b718dd1f/src/messages.jl#L105-L120
# In-place posterior computation, where γ is allocated beforehand.
function posteriors!(γ::AbstractMatrix, α::AbstractMatrix, β::AbstractMatrix)
    @argcheck size(γ) == size(α) == size(β)
    T, K = size(α)
    for t in OneTo(T)
        c = 0.0
        for i in OneTo(K)
            γ[t, i] = α[t, i] * β[t, i]
            c += γ[t, i]
        end

        for i in OneTo(K)
            γ[t, i] /= c
        end
    end
end

# https://github.com/dmetivie/HMMBase.jl/blob/e65510335354be7637a1598a79f02788b718dd1f/src/mle.jl#L1-L17
# In-place update of the initial state distribution.
function update_a!(a::AbstractVector, α::AbstractMatrix, β::AbstractMatrix)
    @argcheck size(α, 1) == size(β, 1)
    @argcheck size(α, 2) == size(β, 2) == size(a, 1)

    K = length(a)
    c = 0.0

    for i in OneTo(K)
        a[i] = α[1, i] * β[1, i]
        c += a[i]
    end

    for i in OneTo(K)
        a[i] /= c
    end
end