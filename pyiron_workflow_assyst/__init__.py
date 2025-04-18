"""
pyiron_workflow_assyst - A workflow package for ASSYST (Automated Symmetry and Stability Testing)
"""

try:
    from .workflow import (
        run_ASSYST_on_structure,
        get_ASSYST_deformed_structures,
        collect_structures,
    )
    from .structure_filter_utils import RCORE, is_valid_structure
except ImportError as e:
    raise ImportError(
        f"Error importing pyiron_workflow_assyst: {str(e)}. "
        "Make sure all dependencies are installed correctly."
    )

__version__ = "0.1.0" 