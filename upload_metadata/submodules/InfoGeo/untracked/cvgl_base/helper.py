"""Minimal public-backbone helper required by the released InfoGeo modules.

The upstream release imports this module but does not include it.  The Route-A
reproduction runner supplies its own DINOv2 backbone, therefore this fallback
only keeps the public module importable and deliberately fails if an omitted
upstream backbone factory is requested.
"""


def get_backbone(*args, **kwargs):
    raise RuntimeError(
        "The public InfoGeo release omitted cvgl_base.helper and its local "
        "DINOv2 paths. Use v36-exp/repro_papers/run_infogeo_routea.py, which "
        "constructs the released slot modules with a public DINOv2 backbone."
    )
