
# --- PyArrow Monkeypatch for stlite/Pyodide ---
# Fixes AttributeError: module 'pyarrow' has no attribute 'RecordBatch'/'ChunkedArray'
# This is due to a partial pyarrow installation in the browser environment.
import sys
try:
    import pyarrow
    # Define dummy classes for missing attributes to satisfy sklearn checks
    if not hasattr(pyarrow, 'RecordBatch'):
        pyarrow.RecordBatch = type("RecordBatch", (), {})
    if not hasattr(pyarrow, 'ChunkedArray'):
        pyarrow.ChunkedArray = type("ChunkedArray", (), {})
    if not hasattr(pyarrow, 'Table'):
        pyarrow.Table = type("Table", (), {})
    if not hasattr(pyarrow, 'Array'):
        pyarrow.Array = type("Array", (), {})
except ImportError:
    pass
# ----------------------------------------------
