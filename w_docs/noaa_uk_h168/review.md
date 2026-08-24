### Potentially excellent result requiring immediate rerun

### One DDIM step appears sufficient
- The guidance/step sweep shows only a \(0.33\%\) total CRPS range, and changing 64 steps to one changes CRPS by only \(0.17\%\).
- If this survives a protocol-correct rerun, it changes the flagship’s cost completely:

CMD-D(25), 1 step = 25 network evaluations,
That could become the strongest paper result: The number of mixture components matters, but reverse-trajectory refinement does not; a one-step noise-to-GP-component mapping retains the mixture quality.

However, the existing sweep cannot support that claim because:
- It used dynamic thresholding \(0.995\), while the frozen protocol uses \(0\).
- It has only two seeds.
- It reports only CRPS.
- Absolute values are inconsistent with the main protocol.
- Clip rate, ES, VS, PIT-ECE, and coverage are missing.

Highest-priority rerun:
\(M\in\{1,2,4,8,16,25\}
\times
\text{steps}\in\{1,4,8,64\}\)
under the frozen configuration, measuring ES, VS, CRPS, mixture NLL, coverage, PIT-ECE, and wall-clock.


## All methods remain undercovered
For CMD-D(25):
- nominal 80% → empirical 73.1%;
- nominal 95% → empirical 86.5%;
- nominal 99% → empirical 89.5%.

The mixture is better than the sampled ensemble but remains materially miscalibrated.
Investigate:
- whether \(q_k,p_k^0,\sigma^2\) actually received NLL gradients;
- whether mixture NLL or componentwise NLL was used;
- whether coverage uses correct mixture quantiles;
- decoder-induced distortion;
- whether variance is transported correctly after normalization;
- whether validation-only variance scaling improves test calibration.
Post-hoc scaling is acceptable only if fitted on validation, frozen, and reported both before and after.

## No nugget advantage 

2K = 256; d_z = 16 

A proper nugget test requires a deliberately rank-deficient configuration such as \(2K<d_z\)