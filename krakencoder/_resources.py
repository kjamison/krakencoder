from pathlib import Path

try:
    from importlib.resources import files as resource_files  # Py ≥ 3.9
except ImportError:
    from importlib_resources import files as resource_files  # backport for 3.8
    
def _pkg_name():
    """
    Return the *tests* package name even from subpackages (e.g., tests.unit).
    Falls back to plain filesystem path if tests aren't executed as a package.
    """
    pkg = __package__ or __name__  # e.g., "tests" or "tests._resources"
    root = pkg.split(".", 1)[0]    # "tests"
    return root

def resource_path(*parts):
    try:
        return str(resource_files(_pkg_name()).joinpath("resources", *parts))
    except Exception:
        # Fallback for non-package execution (e.g., someone runs a test file directly)
        here = Path(__file__).resolve().parent
        return str(here / "resources" / Path(*parts))
