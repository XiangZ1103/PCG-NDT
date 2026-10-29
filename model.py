"""Convenient PCG-NDT entry point."""

from pcg_ndt import (
    PCGNDT,
    build_model,
    architecture_summary,
    parameter_breakdown,
    save_bundle,
    load_bundle,
)


if __name__ == "__main__":
    import json

    print(json.dumps(architecture_summary(build_model()), indent=2))
