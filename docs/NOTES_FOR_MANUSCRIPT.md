# Notes for manuscript alignment

1. The main paper should cite `src/vgib_real_benchmarks.py` as the primary code path.
2. Avoid claiming PCam as a primary benchmark unless complete matched runs are reproduced.
3. If CovType outputs are only training-loss summaries, describe CovType as an optimization-side supporting check rather than primary generalization evidence.
4. The diagnostic quantities are proxies: curvature is estimated by Jacobian/Hutchinson-style norms and intrinsic dimension by participation ratio.
5. Keep the claim narrow: V-GIB is a geometry-aware alternative, not a universal dominance claim over ERM/VIB.
