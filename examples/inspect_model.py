"""Inspect an untrained PCG-NDT architecture."""

import json
from pcg_ndt import build_model, architecture_summary

if __name__ == "__main__":
    print(json.dumps(architecture_summary(build_model()), indent=2))
