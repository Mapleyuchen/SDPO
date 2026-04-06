"""
IsoGraph VE-MDP Environment — Role C Integration Layer.

This module provides the environment interface used by A's ActionInterceptor
and IsoGraphRollout. It re-exports the correct classes based on availability:

- Member C's IsoGraphEnvironment (production VE-MDP with FGW + DGR + SVM)
  when ISOGraph-C package is importable and isograph.use_dummy_env=false.
- DummyEnvironment (pure Python, no external deps) when isograph.use_dummy_env=true
  or when ISOGraph-C is not available.

ActionInterceptor and IsoGraphRollout import from here, so they always get the
correct types (ActionType / ActionResult / Environment) without knowing which
backend is configured.

For Member B data (data-B/page_*.json), Member C's graph_schema validates the
extended node type taxonomy (MAIN_TEXT, SIDE_MARGINALIA, ANNOTATION, SEAL,
COLUMN_SEPARATOR, OUTER_PAGE_FRAME, CONFINED_WITHIN, etc.) and edge types
(READS_AFTER, ANNOTATES, OVERLAPS, INVERTS, REPLACES, DESCRIBES, CONFINED_WITHIN,
etc.).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


# ==============================================================================
# A-compatible types (used by both backends)
# ==============================================================================

class ActionType(Enum):
    """Types of actions that can be intercepted."""
    ZOOM = "zoom"
    CALL_SVM = "call_svm"
    UNKNOWN = "unknown"


@dataclass
class ActionResult:
    """Result from environment step."""
    action_type: ActionType
    action_params: dict
    feedback: str
    is_terminal: bool = False


# ==============================================================================
# Member C environment finder
# ==============================================================================

import os
import sys

ENVIRONMENT_BACKEND = os.environ.get("ISOGRAPH_ENV_BACKEND", "auto")

_MEMBER_C_ENV: Optional[type] = None
_MEMBER_C_REASON: str = ""
_DUMMY_REASON: str = ""


def _find_member_c_environment():
    """Locate and import Member C's IsoGraphEnvironment."""
    _SDPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _MEMBER_C_ROOT = os.environ.get("ISOGRAPH_C_ROOT", "")

    # 1. Try ISOGRAPH_C_ROOT environment variable
    if _MEMBER_C_ROOT:
        if _MEMBER_C_ROOT not in sys.path:
            sys.path.insert(0, _MEMBER_C_ROOT)
        try:
            import ISOGraph_C.isograph_env_c as _mod
            return _mod.IsoGraphEnvironment, f"ISOGRAPH_C_ROOT={_MEMBER_C_ROOT}"
        except ImportError:
            try:
                sys.path.remove(_MEMBER_C_ROOT)
            except ValueError:
                pass
            try:
                sys.path.insert(0, _SDPO_ROOT)
            except ValueError:
                pass

    # 2. Try ISOGraph-C as sibling directory
    _iso_c_default = os.path.join(os.path.dirname(_SDPO_ROOT), "ISOGraph-C")
    if os.path.isdir(_iso_c_default) and _iso_c_default not in sys.path:
        sys.path.insert(0, _iso_c_default)
        try:
            import ISOGraph_C.isograph_env_c as _mod
            return _mod.IsoGraphEnvironment, f"ISOGraph-C sibling={_iso_c_default}"
        except ImportError:
            try:
                sys.path.remove(_iso_c_default)
            except ValueError:
                pass

    # 3. Try pip-installed ISOGraph_C
    try:
        import ISOGraph_C.isograph_env_c as _mod
        return _mod.IsoGraphEnvironment, "pip-installed ISOGraph_C"
    except ImportError:
        pass

    # 4. Try pip-installed C
    try:
        import C.isograph_env_c as _mod
        return _mod.IsoGraphEnvironment, "pip-installed C"
    except ImportError:
        pass

    return None, "Member C not available"


_MEMBER_C_ENV, _MEMBER_C_REASON = _find_member_c_environment()


# ==============================================================================
# DummyEnvironment (inlined — no external dependencies)
# This is a pure-Python fallback used when Member C is unavailable or
# when isograph.use_dummy_env=true.
# ==============================================================================

class DummyEnvironment:
    """
    Dummy environment for testing the Action Interceptor.

    Returns predefined diagnostic feedback when the MLLM triggers special
    action tokens like <zoom> or <call_svm>. It also implements the
    Member C interface stubs wired to the oracle graph.
    """

    def __init__(
        self,
        device: str = "cuda",
        oracle_graph_path: Optional[str] = None,
    ):
        self.device = device
        self.oracle_graph: Optional[Dict[str, Any]] = None
        self.image_id = "unknown"
        self.scenario = "unknown"
        self.img_width = 1000
        self.img_height = 1000

        if oracle_graph_path is None:
            _sdpo_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
            oracle_graph_path = os.path.join(
                _sdpo_root, "global_oracle_graph_demo.json")

        self._load_oracle_graph(oracle_graph_path)

    def _load_oracle_graph(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.oracle_graph = data.get("oracle_graph", {})
            self.image_id = data.get("image_id", "unknown")
            self.scenario = data.get("adversarial_scenario", "unknown")
        except FileNotFoundError:
            self.oracle_graph = {"nodes": [], "edges": []}

    def step_zoom(self, polygon: List[float]) -> str:
        if len(polygon) != 8:
            return ("[VE-MDP step_zoom] ERROR: polygon must have 8 coordinates "
                    "[x1,y1,x2,y2,x3,y3,x4,y4].")
        pts = [f"{polygon[i]:.1f},{polygon[i+1]:.1f}" for i in range(0, 8, 2)]
        return (
            f"[VE-MDP step_zoom] Cropped patch with polygon "
            f"(normalized coords): [{' | '.join(pts)}]. "
            f"Local patch ready for SVM input."
        )

    def step_svm(self, image_patch: Any = None) -> str:
        if self.oracle_graph is None or not self.oracle_graph.get("nodes"):
            return "[VE-MDP step_svm] No oracle graph available for simulation."

        nodes = self.oracle_graph.get("nodes", [])
        edges = self.oracle_graph.get("edges", [])
        simulated_nodes = [
            {
                "node_id": n["node_id"],
                "type": n.get("type", "UNKNOWN"),
                "text": n.get("text", ""),
                "polygon": n.get("polygon", []),
            }
            for n in nodes[:3]
        ]
        simulated_edges = [
            e for e in edges
            if e.get("source") in [n["node_id"] for n in simulated_nodes]
            and e.get("target") in [n["node_id"] for n in simulated_nodes]
        ]
        graph_str = json.dumps(
            {"nodes": simulated_nodes, "edges": simulated_edges},
            ensure_ascii=False,
            indent=2,
        )
        return f"[VE-MDP step_svm] SVM extracted local graph Gl:\n{graph_str}"

    def get_dgr_feedback(self, trajectory: Any = None) -> str:
        if trajectory is not None and hasattr(trajectory, "local_graph"):
            local_graph: Dict[str, Any] = trajectory.local_graph
        elif trajectory is not None and hasattr(trajectory, "nodes"):
            local_graph = {
                "nodes": trajectory.nodes,
                "edges": getattr(trajectory, "edges", []),
            }
        else:
            local_graph = {
                "nodes": [
                    {
                        "node_id": "n_main1",
                        "type": "MAIN_TEXT",
                        "text": "庚子事变后，清廷推行新政...",
                        "polygon": [850, 100, 950, 100, 950, 500, 850, 500],
                    },
                    {
                        "node_id": "n_main2",
                        "type": "MAIN_TEXT",
                        "text": "便上来回王夫人话",
                        "polygon": [850, 500, 950, 500, 950, 900, 850, 900],
                    },
                    {
                        "node_id": "n_side1",
                        "type": "SIDE_MARGINALIA",
                        "text": "不回凤姐",
                        "polygon": [800, 500, 840, 500, 845, 600, 805, 600],
                    },
                ],
                "edges": [
                    {"source": "n_main1", "target": "n_main2", "type": "READS_AFTER"},
                ],
            }

        oracle_graph = self.oracle_graph
        if oracle_graph is None or not oracle_graph.get("nodes"):
            return "[System Diagnostic Report]: Oracle graph not loaded."

        S_node, S_edge, S_order, summary = self._compute_fgw_verification(
            local_graph, oracle_graph
        )
        tau_node, tau_edge, tau_order = 0.8, 0.7, 0.6

        lines = [
            "[System Diagnostic Report]:",
            f"Image: {self.image_id} | Scenario: {self.scenario}",
            "",
        ]
        if S_node < tau_node:
            lines.append(
                f"[!] Semantic Entity Issue (S_node={S_node:.3f} < τ_n={tau_node}): "
                "Hallucinated or missing visual entities."
            )
        else:
            lines.append(f"[+] Semantic Entity: Excellent alignment (S_node={S_node:.3f}).")

        if S_edge < tau_edge:
            lines.append(
                f"[!] Spatial Topology Issue (S_edge={S_edge:.3f} < τ_e={tau_edge}): "
                "Spatial relationships are incorrect."
            )
        else:
            lines.append(f"[+] Spatial Topology: Correct structure (S_edge={S_edge:.3f}).")

        if S_order < tau_order:
            lines.append(
                f"[!] Sequential Logic Issue (S_order={S_order:.3f} < τ_o={tau_order}): "
                "Reading order is incorrect."
            )
        else:
            lines.append(f"[+] Sequential Logic: Correct order (S_order={S_order:.3f}).")

        lines.append("")
        failed = []
        if S_node < tau_node:
            failed.append("entity recognition")
        if S_edge < tau_edge:
            failed.append("spatial topology")
        if S_order < tau_order:
            failed.append("reading order")

        if not failed:
            lines.append("All structural checks passed.")
        else:
            lines.append(f"Please revise: {', '.join(failed)}.")

        lines.append("")
        lines.append(f"[FGW Transport Plan Summary]: {summary}")
        return "\n".join(lines)

    def _compute_fgw_verification(
        self, local_graph: Dict[str, Any], oracle_graph: Dict[str, Any],
    ) -> tuple:
        lambda_n, lambda_e = 1.0, 1.0
        local_nodes = local_graph.get("nodes", [])
        oracle_nodes = oracle_graph.get("nodes", [])
        local_edges = local_graph.get("edges", [])
        oracle_edges = oracle_graph.get("edges", [])

        n_l = max(len(local_nodes), 1)
        n_g = max(len(oracle_nodes), 1)

        C = [[0.0] * n_g for _ in range(n_l)]
        for i, ln in enumerate(local_nodes):
            for j, gn in enumerate(oracle_nodes):
                cost = 0.0
                lpoly = ln.get("polygon", [0, 0, 1, 1])
                gpoly = gn.get("polygon", [0, 0, 1, 1])
                lx1, ly1, lx2, ly2 = (lpoly[0], lpoly[1],
                                       lpoly[4] if len(lpoly) >= 5 else lpoly[2],
                                       lpoly[5] if len(lpoly) >= 6 else lpoly[3])
                gx1, gy1, gx2, gy2 = (gpoly[0], gpoly[1],
                                       gpoly[4] if len(gpoly) >= 5 else gpoly[2],
                                       gpoly[5] if len(gpoly) >= 6 else gpoly[3])
                ix1, iy1 = max(lx1, gx1), max(ly1, gy1)
                ix2, iy2 = min(lx2, gx2), min(ly2, gy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                l_area = max(abs(lx2 - lx1) * abs(ly2 - ly1), 1)
                g_area = max(abs(gx2 - gx1) * abs(gy2 - gy1), 1)
                iou = inter / (l_area + g_area - inter + 1e-6)
                cost += (1.0 - iou) * 0.7
                if ln.get("type") != gn.get("type"):
                    cost += 0.3
                C[i][j] = cost

        T = [[1.0 / (n_l * n_g)] * n_g for _ in range(n_l)]
        for _ in range(20):
            for r in range(n_l):
                row_sum = sum(T[r])
                if row_sum > 0:
                    T[r] = [v / row_sum for v in T[r]]
            for c in range(n_g):
                col_sum = sum(T[r][c] for r in range(n_l))
                if col_sum > 0:
                    for r in range(n_l):
                        T[r][c] /= col_sum

        node_cost = sum(C[i][j] * T[i][j]
                        for i in range(n_l) for j in range(n_g))
        S_node = math.exp(-lambda_n * node_cost)

        local_edge_set = {(e.get("source", ""), e.get("target", ""),
                           e.get("type", "")) for e in local_edges}
        oracle_edge_set = {(e.get("source", ""), e.get("target", ""),
                             e.get("type", "")) for e in oracle_edges}
        matched = sum(1 for le in local_edge_set if le in oracle_edge_set)
        S_edge = math.exp(-lambda_e *
                           (1.0 - matched / max(len(local_edge_set), 1)))

        S_order = 0.85  # Simplified default
        summary = (f"FGW | nodes={n_l}/{n_g}, "
                   f"S_node={S_node:.3f}, S_edge={S_edge:.3f}, S_order={S_order:.3f}")
        return S_node, S_edge, S_order, summary

    def _parse_zoom_action(self, action_str: str) -> Optional[ActionResult]:
        pattern = r"<zoom>\s*\[([\d.\s,]+)\]\s*</zoom>"
        m = re.search(pattern, action_str, re.IGNORECASE)
        if not m:
            return None
        try:
            coords = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
        except ValueError:
            return None
        if len(coords) == 8:
            feedback = self.step_zoom(coords)
        elif len(coords) >= 4:
            feedback = self.step_zoom(coords[:4] + [0, 0, 0, 0])
        else:
            return None
        return ActionResult(
            action_type=ActionType.ZOOM,
            action_params={"polygon": coords[:8]},
            feedback=feedback,
        )

    def _parse_call_svm_action(self, action_str: str) -> Optional[ActionResult]:
        if "<call_svm>" in action_str:
            return ActionResult(
                action_type=ActionType.CALL_SVM,
                action_params={},
                feedback=self.step_svm(),
            )
        return None

    def step(self, action: str) -> ActionResult:
        result = self._parse_zoom_action(action)
        if result is not None:
            return result
        result = self._parse_call_svm_action(action)
        if result is not None:
            return result
        return ActionResult(
            action_type=ActionType.UNKNOWN,
            action_params={},
            feedback="",
        )

    def reset(self) -> None:
        pass


# ==============================================================================
# Factory
# ==============================================================================

# Determine active backend
if ENVIRONMENT_BACKEND == "member_c":
    _ACTIVE_ENV = _MEMBER_C_ENV
    _ACTIVE_REASON = _MEMBER_C_REASON
elif ENVIRONMENT_BACKEND == "dummy":
    _ACTIVE_ENV = DummyEnvironment
    _ACTIVE_REASON = "ISOGRAPH_ENV_BACKEND=dummy"
else:
    # "auto": prefer Member C, fall back to DummyEnvironment
    _ACTIVE_ENV = _MEMBER_C_ENV or DummyEnvironment
    _ACTIVE_REASON = _MEMBER_C_REASON if _MEMBER_C_ENV else "DummyEnvironment (fallback)"


def get_environment_class():
    """Return the active environment class (Member C or DummyEnvironment)."""
    if _ACTIVE_ENV is None:
        raise RuntimeError(
            f"[isograph_env] No environment available. "
            f"Member C: {_MEMBER_C_REASON}. "
            "Set ISOGRAPH_C_ROOT or pip install the ISOGraph-C package."
        )
    return _ACTIVE_ENV


def create_environment(
    oracle_graph_path: Optional[str] = None,
    image_path: Optional[str] = None,
    device: str = "cuda",
    svm_backend: str = "dummy",
    svm_model_path: Optional[str] = None,
    fgw_alpha: float = 0.5,
    **kwargs,
):
    """
    Factory: create the active VE-MDP environment.

    Args:
        oracle_graph_path: Path to oracle graph JSON.
        image_path: Path to source image (Member C only).
        device: Device string.
        svm_backend: "dummy" or "onnx" (Member C only).
        svm_model_path: Path to ONNX SVM model (Member C only).
        fgw_alpha: FGW alpha parameter (Member C only).
        kwargs: Passed to DGRConfig (tau_node, tau_edge, tau_order).

    Returns:
        An environment instance compatible with ActionInterceptor.
    """
    env_cls = get_environment_class()
    try:
        if env_cls is _MEMBER_C_ENV and _MEMBER_C_ENV is not None:
            return env_cls(
                oracle_graph_path=oracle_graph_path,
                image_path=image_path,
                svm_backend=svm_backend,
                svm_model_path=svm_model_path,
                fgw_alpha=fgw_alpha,
                device=device,
            )
        else:
            return env_cls(
                oracle_graph_path=oracle_graph_path,
                device=device,
            )
    except TypeError as e:
        raise TypeError(
            f"[isograph_env] create_environment() failed for "
            f"{_ACTIVE_REASON}: {e}"
        ) from e


__all__ = [
    "ActionType",
    "ActionResult",
    "DummyEnvironment",
    "get_environment_class",
    "create_environment",
]
