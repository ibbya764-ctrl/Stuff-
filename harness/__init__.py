"""Experiment harness for the CP2 AI program (v3 record, Part 5.2 ladder).

Every experiment in this package shares one skeleton:
  matched-budget baseline + intervention, fixed eval suite, spectral
  telemetry per checkpoint, logged config/seed, and a PRE-REGISTERED
  prediction + kill condition written before the run (the periodogram
  discipline). The runner refuses to start without the registration.
"""

import os
import sys

# Make the repo-root spectral_telemetry module importable from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
