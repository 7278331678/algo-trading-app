import os

# Match math-library threads to the CPUs we actually have (avoids massive oversubscription
# when the library sees the host's cores but the sandbox/pod only gets a few).
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, str(min(4, os.cpu_count() or 1)))
