from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    phase: str
    adapter_module: str | None
    adapter_class: str | None
    enabled: bool = False


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "aecs": MethodSpec("aecs", "1B", "experiments.battery_pack_wltp.imputers.aecs_imputer", "AECSImputer", enabled=True),
    "deep_ae": MethodSpec("deep_ae", "1B", "experiments.battery_pack_wltp.imputers.mentor_ae_family_imputer", "DeepAEImputer", enabled=True),
    "sm_dae": MethodSpec("sm_dae", "1B", "experiments.battery_pack_wltp.imputers.mentor_ae_family_imputer", "SMDAEImputer", enabled=True),
    "sdai": MethodSpec("sdai", "1B", "experiments.battery_pack_wltp.imputers.mentor_ae_family_imputer", "SDAIImputer", enabled=True),
    "trdae": MethodSpec("trdae", "1B", "experiments.battery_pack_wltp.imputers.mentor_ae_family_imputer", "TRDAEImputer", enabled=True),
    "gain": MethodSpec("gain", "1C", "experiments.battery_pack_wltp.imputers.gain_imputer", "GAINImputer", enabled=True),
}


def list_method_specs() -> tuple[MethodSpec, ...]:
    return tuple(METHOD_REGISTRY.values())


def list_method_names() -> tuple[str, ...]:
    return tuple(METHOD_REGISTRY.keys())


def get_method_spec(method_name: str) -> MethodSpec:
    try:
        return METHOD_REGISTRY[method_name]
    except KeyError as exc:
        raise KeyError(f"Unknown method placeholder: {method_name}") from exc
