function viterbi!(
    T1::AbstractMatrix,
    T2::AbstractMatrix,
    z::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    L::AbstractMatrix;
    n2t = CyclicArray(1:size(A, 3), "1D")::AbstractArray{<:Integer}
)
    N, K = size(L)
    (N == 0) && return

    fill!(T1, 0)
    fill!(T2, 0)

    c = zero(eltype(T1))

    for i in OneTo(K)
        T1[1, i] = a[i] * L[1, i]
        c += T1[1, i]
    end

    for i in OneTo(K)
        T1[1, i] /= c
    end

    @inbounds for n = 2:N
        tₙ₋₁ = n2t[n-1] # t-1
        c = zero(eltype(T1))
    
        for j in OneTo(K)
            # TODO: If there is NaNs in T1 this may
            # stay to 0 (NaN > -Inf == false).
            # Hence it will crash when computing z[t].
            # Maybe we should check for NaNs beforehand ?
            amax = 0
            vmax = -Inf
    
            for i in OneTo(K)
                v = T1[n-1, i] * A[i, j, tₙ₋₁]
                if v > vmax
                    amax = i
                    vmax = v
                end
            end
    
            T1[n, j] = vmax * L[n, j]
            T2[n, j] = amax
            c += T1[n, j]
        end
    
        for i in OneTo(K)
            T1[n, i] /= c
        end
    end

    z[N] = argmax(T1[N, :])
    for n = N-1:-1:1
        z[n] = T2[n+1, z[n+1]]
    end
end

function viterbilog!(
    T1::AbstractMatrix,
    T2::AbstractMatrix,
    z::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractMatrix;
    n2t = CyclicArray(1:size(A, 3), "1D")::AbstractArray{<:Integer}
)
    N, K = size(LL)
    (N == 0) && return

    fill!(T1, 0)
    fill!(T2, 0)

    al = log.(a)
    Al = log.(A)

    for i in OneTo(K)
        T1[1, i] = al[i] + LL[1, i]
    end

    @inbounds for n = 2:N
        tₙ₋₁ = n2t[n-1] # t-1
        for j in OneTo(K)
            amax = 0
            vmax = -Inf

            for i in OneTo(K)
                v = T1[n-1, i] + Al[i, j, tₙ₋₁]
                if v > vmax
                    amax = i
                    vmax = v
                end
            end

            T1[n, j] = vmax + LL[n, j]
            T2[n, j] = amax
        end
    end

    z[N] = argmax(T1[N, :])
    for n = N-1:-1:1
        z[n] = T2[n+1, z[n+1]]
    end
end