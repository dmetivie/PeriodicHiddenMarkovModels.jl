function update_A!(
    A::AbstractArray{F,3} where {F},
    ξ::AbstractArray,
    α::AbstractMatrix,
    β::AbstractMatrix,
    LL::AbstractMatrix,
)
    @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1)
    @argcheck size(α, 2) ==
              size(β, 2) ==
              size(LL, 2) ==
              size(A, 1) ==
              size(A, 2) ==
              size(ξ, 2) ==
              size(ξ, 3)

    N, K = size(LL)
    T = size(A, 3)
    n2t = CyclicArray(1:T, "1D")
    @inbounds for n in OneTo(N - 1)
        t = n2t[n] # periodic t
        m = vec_maximum(view(LL, n + 1, :))
        c = zero(eltype(ξ))

        for i in OneTo(K), j in OneTo(K)
            ξ[n, i, j] = α[n, i] * A[i, j, t] * exp(LL[n+1, j] - m) * β[n+1, j]
            c += ξ[n, i, j]
        end

        for i in OneTo(K), j in OneTo(K)
            ξ[n, i, j] /= c
        end
    end

    fill!(A, 0)
    ## For periodicHMM only the n observation corresponding to A(t) are used to update A(t)
    tₙ = n2t[1:N]
    n_in_t = [findall(tₙ .== t) for t = 1:T] # could probably be speeded up

    @inbounds for t in OneTo(T)
        for i in OneTo(K)
            c = zero(eltype(ξ))

            for j in OneTo(K)
                for n in setdiff(n_in_t[t], N)
                    A[i, j, t] += ξ[n, i, j]
                end
                c += A[i, j, t]
            end

            for j in OneTo(K)
                A[i, j, t] /= c
            end
        end
    end
end

# # In-place update of the observations distributions.
function update_B!(B::AbstractMatrix, γ::AbstractMatrix, observations, estimator)
    @argcheck size(γ, 1) == size(observations, 1)
    @argcheck size(γ, 2) == size(B, 1)
    N = size(γ, 1)
    K = size(B, 1)
    T = size(B, 2)
    ## For periodicHMM only the n observation corresponding to A(t) are used to update A(t)
    n2t = CyclicArray(1:T, "1D")
    tₙ = n2t[1:N]
    n_in_t = [findall(tₙ .== t) for t = 1:T] # could probably be speeded up. For exemple computed outside the function only once

    @inbounds for t in OneTo(T)
        n_t = n_in_t[t]
        for i in OneTo(K)
            if sum(γ[n_t, i]) > 0
                B[i, t] = estimator(typeof(B[i, t]), permutedims(observations[n_t, :]), γ[n_t, i])
            end
        end
    end
end

function fit_mle!(
    hmm::PeriodicHMM,
    observations;
    display=:none,
    maxiter=100,
    tol=1e-3,
    robust=false,
    estimator=fit_mle
)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck T == size(hmm.B, 2)
    history = EMHistory(false, 0, [])

    # Allocate memory for in-place updates
    c = zeros(N)
    α = zeros(N, K)
    β = zeros(N, K)
    γ = zeros(N, K)
    ξ = zeros(N, K, K)
    LL = zeros(N, K)

    loglikelihoods!(LL, hmm, observations)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL)
    backwardlog!(β, c, hmm.a, hmm.A, LL)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_a!(hmm.a, α, β)
        update_A!(hmm.A, ξ, α, β, LL)
        update_B!(hmm.B, γ, observations, estimator)

        # Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely observations.
        robust && (hmm.A .+= eps())

        @check isprobvec(hmm.a)
        @check istransmats(hmm.A)

        loglikelihoods!(LL, hmm, observations)
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

        forwardlog!(α, c, hmm.a, hmm.A, LL)
        backwardlog!(β, c, hmm.a, hmm.A, LL)
        posteriors!(γ, α, β)

        logtotp = sum(c)
        (display == :iter) && println("Iteration $it: logtot = $logtotp")

        push!(history.logtots, logtotp)
        history.iterations += 1

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history.converged = true
            break
        end

        logtot = logtotp
    end

    if !history.converged
        if display in [:iter, :final]
            println("EM has not converged after $(history.iterations) iterations, logtot = $logtot")
        end
    end

    history
end