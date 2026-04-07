from __future__ import annotations

import sys
import os
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

class ActionType(Enum):
    ZOOM = "zoom"
    CALL_SVM = "call_svm"
    UNKNOWN = "unknown"

@dataclass
class ActionResult:
    action_type: ActionType
    action_params: dict
    feedback: str
    is_terminal: bool = False

PARENT_DIR = "/home/aisuan"
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    import ISOGraph_C.isograph_env_c as _mod
    IsoGraphEnvironment = _mod.IsoGraphEnvironment
    
    print(f"\n=======================================================")
    print(f"[SUCCESS] Member C's IsoGraphEnvironment forcefully loaded as a package!")
    print(f"=======================================================\n")
except Exception as e:
    raise RuntimeError(f"FATAL: Could not load Member C from {PARENT_DIR}. Error: {e}")

def get_environment_class():
    return IsoGraphEnvironment

def create_environment(
    oracle_graph_path: Optional[str] = None,
    image_path: Optional[str] = None,
    device: str = "cuda",
    svm_backend: str = "dummy",
    svm_model_path: Optional[str] = None,
    fgw_alpha: float = 0.5,
    **kwargs,
):
    return IsoGraphEnvironment(
        oracle_graph_path=oracle_graph_path,
        image_path=image_path,
        svm_backend=svm_backend,
        svm_model_path=svm_model_path,
        fgw_alpha=fgw_alpha,
        device=device,
    )

class DummyEnvironment:
    """
    An empty stub to satisfy the import statements in train_isograph_sdpo.py 
    and __init__.py. The real IsoGraphEnvironment is now forcefully used.
    """
    pass

# 更新对外的导出列表，把DummyEnvironment加上
__all__ = ["ActionType", "ActionResult", "get_environment_class", "create_environment", "DummyEnvironment"]