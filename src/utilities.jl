"""
    n_to_t(N::Int, T::Int)

    This function transforms all index of the chain `n` into their periodic counterpart `t`.
"""
function n_to_t(N::Int, T::Int)
    return [repeat(1:T, N ÷ T); remaining(N - T * (N ÷ T))]
end

remaining(N::Int) = N > 0 ? range(1, length=N) : Int64[]