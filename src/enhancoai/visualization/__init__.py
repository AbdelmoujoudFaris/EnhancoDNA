"""3D and 2D scientific visualisation.

3D rendering prefers PyVista (interactive) when installed, and always falls
back to a static matplotlib 3D projection otherwise, so the GUI and report
generator work without the optional PyVista dependency.
"""

try:
    import pyvista  # noqa: F401

    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
