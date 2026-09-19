"""SOC dashboard backend.

Serves the real-LANL detection pipeline to the analyst dashboard in
``frontend/``. It imports the research pipeline in ``ml/`` and never modifies
it: every score, feature, label and explanation it serves is produced by the
existing, tested research code.
"""
