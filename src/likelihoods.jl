function likelihoods!(L::AbstractMatrix, hmm::PeriodicHMM{Univariate}, observations)
    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(LL) == (N, K)
    n2t = CyclicArray(1:T, "1D")
    @inbounds for i in OneTo(K), n in OneTo(T)
        t = n2t[n] # periodic t
        L[n, i] = pdf(hmm.B[i, t], observations[n])
    end
end

function likelihoods!(L::AbstractMatrix, hmm::PeriodicHMM{Multivariate}, observations)
    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(LL) == (N, K)
    n2t = CyclicArray(1:T, "1D")
    @inbounds for i in OneTo(K), n in OneTo(T)
        t = n2t[n] # periodic t
        L[n, i] = pdf(hmm.B[i, t], view(observations, n, :))
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::PeriodicHMM{Univariate}, observations)
    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(LL) == (N, K)
    n2t = CyclicArray(1:T, "1D")
    @inbounds for i in OneTo(K), n in OneTo(N)
        t = n2t[n] # periodic t
        LL[n, i] = logpdf(hmm.B[i, t], observations[n])
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::PeriodicHMM{Multivariate}, observations)
    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(LL) == (N, K)
    n2t = CyclicArray(1:T, "1D")
    @inbounds for i in OneTo(K), n in OneTo(N)
        t = n2t[n] # periodic t
        LL[n, i] = logpdf(hmm.B[i, t], view(observations, n, :))
    end
end