"""Decimation strategies for the UV-decimate node.

``TextureQEMStrategy`` keeps the input atlas and rebakes only the normal map; the router picks
it whenever the face budget clears the mesh's UV-seam lock floor.
``RetopoBakeStrategy`` throws the atlas away, simplifies in 3D, builds a fresh ``xatlas`` atlas
and bakes every PBR channel onto it; it is the only way to reach very low budgets.
``AutoRouterStrategy`` chooses between them from the mesh's own measurements and falls back when
the 5D collapse stalls.

``create_strategy``/``decimate`` are the entry points the node layer uses.
"""

from __future__ import annotations

try:  # normal case: imported as part of the comfyui-uv-decimate package
    from .auto_router import AutoRouterStrategy
    from .base import DecimationStrategy
    from .retopo_bake import RetopoBakeStrategy, repack_uv_xatlas
    from .tex_qem import TextureQEMStrategy
except ImportError:  # loaded as a plain directory rather than a package
    from auto_router import AutoRouterStrategy
    from base import DecimationStrategy
    from retopo_bake import RetopoBakeStrategy, repack_uv_xatlas
    from tex_qem import TextureQEMStrategy

try:
    from ..contracts import DecimateConfig, MeshPayload
except ImportError:  # loaded as a plain directory rather than a package
    from contracts import DecimateConfig, MeshPayload

#: Every ``DecimateConfig.strategy`` value, including the ``keep_uv`` compatibility alias.
STRATEGY_REGISTRY = {
    "auto": AutoRouterStrategy,
    "tex_qem": TextureQEMStrategy,
    "keep_uv": TextureQEMStrategy,
    "retopo_bake": RetopoBakeStrategy,
}


def create_strategy(name: str = "auto") -> DecimationStrategy:
    """Build the strategy registered under ``name`` (case-insensitive)."""
    try:
        factory = STRATEGY_REGISTRY[str(name).lower()]
    except KeyError:
        raise ValueError("unknown strategy %r; expected one of %s"
                         % (name, sorted(STRATEGY_REGISTRY))) from None
    return factory()


def decimate(mesh_data: MeshPayload, config: DecimateConfig) -> MeshPayload:
    """Decimate one payload with the strategy named by ``config.strategy``."""
    return create_strategy(config.strategy).execute(mesh_data, config)


__all__ = [
    "AutoRouterStrategy",
    "DecimationStrategy",
    "RetopoBakeStrategy",
    "STRATEGY_REGISTRY",
    "TextureQEMStrategy",
    "create_strategy",
    "decimate",
    "repack_uv_xatlas",
]
