"""Unified imputer adapters for the battery_pack_wltp experiment."""

__all__ = [
    "AECSImputer",
    "DeepAEImputer",
    "MentorAEFamilyImputer",
    "SDAIImputer",
    "SMDAEImputer",
    "TRDAEImputer",
    "GAINImputer",
]


def __getattr__(name):
    if name == "AECSImputer":
        from .aecs_imputer import AECSImputer

        return AECSImputer
    if name in {"DeepAEImputer", "MentorAEFamilyImputer", "SDAIImputer", "SMDAEImputer", "TRDAEImputer"}:
        from .mentor_ae_family_imputer import (
            DeepAEImputer,
            MentorAEFamilyImputer,
            SDAIImputer,
            SMDAEImputer,
            TRDAEImputer,
        )

        return {
            "DeepAEImputer": DeepAEImputer,
            "MentorAEFamilyImputer": MentorAEFamilyImputer,
            "SDAIImputer": SDAIImputer,
            "SMDAEImputer": SMDAEImputer,
            "TRDAEImputer": TRDAEImputer,
        }[name]
    if name == "GAINImputer":
        from .gain_imputer import GAINImputer

        return GAINImputer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
