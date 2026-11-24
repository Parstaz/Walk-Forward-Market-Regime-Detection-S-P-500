"""Walk-forward HMM market regime detection (v2).

Thread counts are pinned to 1 before numpy is imported anywhere in the
package. Multithreaded BLAS reductions are not bit-for-bit reproducible,
which made v1 results drift between runs even with a fixed random_state.
Import this package before importing numpy for the setting to take effect.
VECLIB_MAXIMUM_THREADS covers Apple Accelerate, which macOS numpy wheels use.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"
