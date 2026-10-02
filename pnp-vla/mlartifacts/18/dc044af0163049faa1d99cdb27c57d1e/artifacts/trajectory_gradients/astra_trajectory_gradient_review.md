# Astra review of trajectory gradients

## Conclusion

No blocking computation bug found. Gradient magnitude does not establish a beneficial action direction.

## Checks

- Correct sigmoid conversion, action-standard-deviation factor, fixed-reference differentiation, trajectory outcome labels, branch-root context and known-deadline masks.
- Astra independently checked directional logit derivatives on CPU for one stored-source and one fresh-branch first chunk per model, at two epsilon sizes. Relative finite-difference discrepancy was at most 0.0202% for scalar MC/TD and 0.488% for CNN. CPU derivatives matched saved results within 1.5e-7 scalar and 1.07e-4 CNN; Q mismatch at most 1.2e-7.
- Chunk-weighted ratio-estimator SE clustered by originating episode is appropriate for the reported estimand.

## Interpretation

CNN uses a 100-fold displacement scale and learned steep residual slopes. When the recorded candidate equals the fixed reference, the residual value cancels exactly while its candidate partial derivative can be large. This applies to at least 95.6% of the held-out audited chunks (at most 640 fresh-branch first chunks differ among 14,553 chunks). Most CNN Q values here therefore reproduce its frozen baseline. Its CNN residual was trained on root states; continuation-state slopes also extrapolate outside that training distribution. Large slopes can saturate Q at finite correction radii, so linear predictions should not be extrapolated.

Outcome groups differ in state, stage, task and trajectory lengths: held-out success trajectories average about 10.2 chunks, failures about 23.9. Success/failure contributing episode sets overlap in 198 training and 41 held-out roots. Any difference uncertainty must retain their covariance through joint root influence or a joint episode bootstrap, not independent-SE quadrature. No difference SE was reported. This chunk-weighted estimand differs from the equal-trajectory/equal-episode Bellman summaries.

Native action units are policy outputs before critic standardization, not physical robot units. Gradients evaluate full recorded proposals, including potentially unexecuted terminal tails.

## Optional future hardening

Fingerprint cached trajectory inputs/reference convention in addition to checkpoint SHA before reusing gradient arrays. No evidence current cache reuse was stale.
