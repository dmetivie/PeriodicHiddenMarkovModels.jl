# Changelog

## v0.3.2

### Fixed

- **`fit!` re-estimated each transition matrix from the wrong transitions, so Baum–Welch could
  decrease the loglikelihood.** `HiddenMarkovModels.jl` builds the transition marginal as

  ```julia
  trans = transition_matrix(hmm, control_seq[t + 1])
  mul_rows_cols!(ξ[t], view(α, :, t), trans, view(Bβ, :, t + 1))
  ```

  so `A(l)` governs the step that *arrives at* a time of period `l`. The M-step was grouping
  `ξ[t]` by `n2t[t]` rather than `n2t[t + 1]`, which attributes each `ξ` to the period one step
  too early and makes `A(l)` an estimate of `A(l+1)`'s transitions. The symptom was
  `Loglikelihood decreased … in Baum-Welch`.

  This was invisible before `HiddenMarkovModels v0.6`, which moved the indexing from
  `control_seq[t]` to `control_seq[t + 1]` — hence the `0.6` lower bound in `[compat]`. It is
  also invisible whenever there is a single period, which is why the homogeneous `T = 1` testset
  never caught it.

### Tests

- The periodic testset now runs Baum–Welch to convergence (it needs 1684 iterations to reach
  `atol = 1e-5`, not the default 100) and asserts the property that actually characterises a
  maximum likelihood estimate: the fit reaches a *higher* loglikelihood than the parameters the
  data was drawn from. Here the excess is 29.7 nats against the `p/2 = 30` that Wilks predicts
  for the model's 60 free parameters.
- Parameter recovery in that testset is a loose smoke test, not a sharp one, and now says so.
  The two states' emission means differ by at most 2.0 while their standard deviations are 1–2,
  so they overlap at every period and the transition probabilities are weakly identified: at
  convergence the fitted first row at `t = 4` sits on the edge of the simplex, `[1.0, 3.2e-6]`,
  against a true `[0.784, 0.216]`. Separating the states would make a sharp test possible.
- The comparison is entrywise now. `≈` on matrices compares `norm(A - B)`, so four entries each
  within `0.216` still fail a matrix-level `atol = 0.25`.
