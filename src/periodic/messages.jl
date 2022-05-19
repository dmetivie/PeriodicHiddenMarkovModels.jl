# In-place forward pass, where α and c are allocated beforehand.
function forwardlog!(
    α::AbstractMatrix,
    c::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractMatrix; 
    n2t = n_to_t(size(LL, 1), size(A, 3))::AbstractArray{<:Integer}
)
    @argcheck size(α, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(α, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    N, K = size(LL)

    fill!(α, 0)
    fill!(c, 0)

    m = vec_maximum(view(LL, 1, :))

    for j in OneTo(K)
        α[1, j] = a[j] * exp(LL[1, j] - m)
        c[1] += α[1, j]
    end

    for j in OneTo(K)
        α[1, j] /= c[1]
    end

    c[1] = log(c[1]) + m

    @inbounds for n = 2:N
        tₙ₋₁ = n2t[n-1] # periodic t-1
        m = vec_maximum(view(LL, n, :))

        for j in OneTo(K)
            for i in OneTo(K)
                α[n, j] += α[n-1, i] * A[i, j, tₙ₋₁]
            end
            α[n, j] *= exp(LL[n, j] - m)
            c[n] += α[n, j]
        end

        for j in OneTo(K)
            α[n, j] /= c[n]
        end

        c[n] = log(c[n]) + m
    end
end

# In-place backward pass, where β and c are allocated beforehand.
function backwardlog!(
    β::AbstractMatrix,
    c::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractMatrix; 
    n2t = n_to_t(size(LL, 1), size(A, 3))::AbstractArray{<:Integer}
)
    @argcheck size(β, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(β, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    N, K = size(LL)
    T = size(A, 3)
    L = zeros(K)
    (T == 0) && return

    fill!(β, 0)
    fill!(c, 0)

    for j in OneTo(K)
        β[end, j] = 1
    end

    @inbounds for n = N-1:-1:1
        t = n2t[n] # periodic t
        m = vec_maximum(view(LL, n + 1, :))

        for i in OneTo(K)
            L[i] = exp(LL[n+1, i] - m)
        end

        for j in OneTo(K)
            for i in OneTo(K)
                β[n, j] += β[n+1, i] * A[j, i, t] * L[i]
            end
            c[n+1] += β[n, j]
        end

        for j in OneTo(K)
            β[n, j] /= c[n+1]
        end

        c[n+1] = log(c[n+1]) + m
    end

    m = vec_maximum(view(LL, 1, :))

    for j in OneTo(K)
        c[1] += a[j] * exp(LL[1, j] - m) * β[1, j]
    end

    c[1] = log(c[1]) + m
end

function forward(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix; kwargs...)
    m = Matrix{Float64}(undef, size(LL))
    c = Vector{Float64}(undef, size(LL, 1))
    forwardlog!(m, c, a, A, LL; kwargs...)
    m, sum(c)
end

function backward(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix; kwargs...)
    m = Matrix{Float64}(undef, size(LL))
    c = Vector{Float64}(undef, size(LL, 1))
    backwardlog!(m, c, a, A, LL; kwargs...)
    m, sum(c)
end

function posteriors(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix; kwargs...)
    α, _ = forward(a, A, LL; kwargs...)
    β, _ = backward(a, A, LL; kwargs...)
    posteriors(α, β)
end
