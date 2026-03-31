# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dummy Environment for IsoGraph (Active-Symbolic SDPO) Framework.

This is a placeholder environment that returns predefined diagnostic feedback
when the MLLM triggers special action tokens like <zoom> or <call_svm>.

In production, this would be replaced by:
- VE-MDP (Visual-Evidence Markov Decision Process) for image/text interaction
- DGR (Diagnostic Graph Report) 富文本诊断反馈

Member C interface stubs:
- step_zoom(polygon): Receives 8-point polygon, returns local crop / placeholder text.
- step_svm(image_patch): Simulates calling a frozen Small Visual Model (e.g. LayoutLM),
  returns a serialized local symbolic graph Gl = (Vl, El).
- get_dgr_feedback(trajectory): Performs FGW optimal-transport matching between the
  accumulated local evidence and the oracle ground-truth graph Gg, then returns
  a natural-language Diagnostic Graph Report (f_DGR).
"""

import re
import json
import math
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


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


class DummyEnvironment:
    """
    Dummy environment for testing the Action Interceptor.

    In the full IsoGraph framework, this would be replaced by:
    - VE-MDP: Visual-Evidence Markov Decision Process
    - DGR: Diagnostic Graph Report generator

    This dummy returns fixed strings simulating SVM extraction results.
    It also implements the Member C interface stubs wired to the oracle graph
    defined in global_oracle_graph_demo.json.
    """

    def __init__(
        self,
        device: str = "cuda",
        oracle_graph_path: str = None,
    ):
        self.device = device
        # Predefined responses simulating SVM/OCR extraction
        self.svm_responses = [
            "[System: SVM extracted text from region x1=100,y1=200,x2=400,y2=350: '庚子事变后，清廷推行新政...']",
            "[System: OCR detected vertical text in column 3: '光绪三十一年乙巳...']",
            "[System: SVM classified layout region as '序跋类' with confidence 0.92]",
            "[System: Visual evidence extracted: 书籍尺寸 23.5cm × 16.8cm, 板框...]",
            "[System: Symbol detection found 12个古籍专有字符 in region...]",
        ]
        self.response_idx = 0

        # ---- Member C: Load oracle graph ----
        self.oracle_graph: Optional[Dict[str, Any]] = None
        if oracle_graph_path is None:
            oracle_graph_path = "/home/mail-robo/IsoGraph/SDPO/global_oracle_graph_demo.json"
        self._load_oracle_graph(oracle_graph_path)

    def _load_oracle_graph(self, path: str) -> None:
        """Load the ground-truth oracle graph Gg from JSON."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.oracle_graph = data.get("oracle_graph", {})
            self.image_id = data.get("image_id", "unknown")
            self.scenario = data.get("adversarial_scenario", "unknown")
        except FileNotFoundError:
            self.oracle_graph = {"nodes": [], "edges": []}
            self.image_id = "unknown"
            self.scenario = "unknown"

    # ========================================================================
    # Member C Interface: step_zoom(polygon)
    # ========================================================================
    def step_zoom(self, polygon: List[float]) -> str:
        """
        Member C stub: receive an 8-point normalized polygon [x1,y1,...,x4,y4],
        crop the corresponding region from the global image, and return a
        placeholder text or the extracted patch metadata.

        In production this would:
          1. Map normalized coordinates to pixel space.
          2. Crop I_patch from I_global.
          3. Return either the image patch or a serialized description.

        Here we return a deterministic placeholder so the rollout stays reproducible.

        Args:
            polygon: List of 8 floats (x1,y1,x2,y2,x3,y3,x4,y4), normalized to [0,1].

        Returns:
            A string describing the cropped region.
        """
        if len(polygon) != 8:
            return "[VE-MDP step_zoom] ERROR: polygon must have 8 coordinates [x1,y1,x2,y2,x3,y3,x4,y4]."

        # Normalize coordinates assuming image size from oracle graph metadata
        img_w = 1000  # fallback width
        img_h = 1000  # fallback height
        try:
            # Try to read image dimensions from oracle graph metadata if available
            meta = {}
            # Heuristic: use the max extent of oracle polygons
            for node in self.oracle_graph.get("nodes", []):
                poly = node.get("polygon", [])
                # Not strictly necessary here; just use defaults
                pass
        except Exception:
            pass

        pts = [f"{polygon[i]:.1f},{polygon[i+1]:.1f}" for i in range(0, 8, 2)]
        feedback = (
            f"[VE-MDP step_zoom] Cropped patch with polygon (normalized coords): "
            f"[{' | '.join(pts)}]. "
            f"Image region extracted. "
            f"Returning local patch representation for SVM input."
        )
        return feedback

    # ========================================================================
    # Member C Interface: step_svm(image_patch)
    # ========================================================================
    def step_svm(self, image_patch: Any = None) -> str:
        """
        Member C stub: invoke a frozen Small Visual Model (SVM) on the given
        image patch and return a serialized local symbolic graph Gl = (Vl, El).

        In production this would call a LayoutLM/Donut model to parse the patch
        into structured graph nodes and edges.

        Here we return a deterministic placeholder derived from the oracle graph
        so that the Self-Teacher can be evaluated against a known structure.

        Args:
            image_patch: Either a PIL Image, an image tensor, or any patch
                         identifier. Currently unused (placeholder).

        Returns:
            A string serialization of the local graph Gl = (nodes, edges) that
            the SVM would extract from the patch.
        """
        if self.oracle_graph is None or not self.oracle_graph.get("nodes"):
            return "[VE-MDP step_svm] No oracle graph available for simulation."

        # Simulate SVM extracting a subset of the oracle nodes relevant to a patch.
        # In a real setting, the SVM would output actual detected entities.
        # Here we return a representative subgraph to make the DGR feedback meaningful.
        nodes = self.oracle_graph.get("nodes", [])
        edges = self.oracle_graph.get("edges", [])

        # Simulate: SVM detects the first two nodes (main text + first marginalia)
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

    # ========================================================================
    # Member C Interface: get_dgr_feedback(trajectory)
    # ========================================================================
    def get_dgr_feedback(self, trajectory: Any = None) -> str:
        """
        Member C stub: evaluate the accumulated local graph evidence against
        the oracle ground-truth graph Gg using FGW optimal-transport matching,
        then return a natural-language Diagnostic Graph Report (f_DGR).

        This method ties together the three verification channels:
          - Node alignment score S_node
          - Edge topology score S_edge
          - Reading-order consistency score S_order

        The DGR is passed to the Self-Teacher as the rich feedback f_DGR.

        Args:
            trajectory: Either an InterceptedTrajectory or any object exposing
                        a `local_graph` attribute (Dict). If None, a default
                        simulated local graph is used.

        Returns:
            A string DGR report that can be appended to the context window.
        """
        # Resolve the local graph from the trajectory
        if trajectory is not None and hasattr(trajectory, "local_graph"):
            local_graph: Dict[str, Any] = trajectory.local_graph
        elif trajectory is not None and hasattr(trajectory, "nodes"):
            # Assume it is already a graph dict
            local_graph = {"nodes": trajectory.nodes, "edges": getattr(trajectory, "edges", [])}
        else:
            # Fallback: simulate a local graph with a deliberate topology error
            local_graph = {
                "nodes": [
                    {
                        "node_id": "n_main1",
                        "type": "MAIN_TEXT",
                        "text": "話說周瑞家的送了刘姥姥去後",
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
                    # Intentionally wrong: direct main1 -> main2, missing side chain
                    {"source": "n_main1", "target": "n_main2", "type": "READS_AFTER"},
                    # Missing ANNOTATES edge from n_side1 -> n_main2
                ],
            }

        oracle_graph = self.oracle_graph
        if oracle_graph is None or not oracle_graph.get("nodes"):
            return "[System Diagnostic Report]: Oracle graph not loaded. Cannot generate DGR."

        # ---- FGW Verification (simplified, CPU-based) ----
        S_node, S_edge, S_order, T_str = self._compute_fgw_verification(
            local_graph, oracle_graph
        )

        # ---- Thresholds from the paper ----
        tau_node = 0.8
        tau_edge = 0.7
        tau_order = 0.6

        # ---- Build DGR text following Φ(...) from the paper ----
        lines = []
        lines.append("[System Diagnostic Report]:")
        lines.append(f"Image: {self.image_id} | Scenario: {self.scenario}")
        lines.append("")

        # Node alignment
        if S_node < tau_node:
            lines.append(
                f"[!] Semantic Entity Issue (S_node={S_node:.3f} < τ_n={tau_node}): "
                "Detected hallucinated or missing visual entities. "
                "The local graph does not fully cover the oracle nodes."
            )
        else:
            lines.append(f"[+] Semantic Entity: Excellent alignment (S_node={S_node:.3f}).")

        # Edge topology
        if S_edge < tau_edge:
            lines.append(
                f"[!] Spatial Topology Issue (S_edge={S_edge:.3f} < τ_e={tau_edge}): "
                "The spatial relationships between entities are incorrect. "
                "The layout edge structure diverges from the oracle. "
                "For example, the ANNOTATES edge that should connect marginalia to the "
                "main text column may be missing or misdirected."
            )
        else:
            lines.append(f"[+] Spatial Topology: Correct structure (S_edge={S_edge:.3f}).")

        # Reading order
        if S_order < tau_order:
            lines.append(
                f"[!] Sequential Logic Issue (S_order={S_order:.3f} < τ_o={tau_order}): "
                "Reading order is incorrect. The Kendall rank correlation between the "
                "local traversal sequence and the oracle is low. "
                "Marginalia notes should follow a parallel branching reading order, "
                "not a linear flattening."
            )
        else:
            lines.append(f"[+] Sequential Logic: Correct order (S_order={S_order:.3f}).")

        lines.append("")

        # Summary
        failed_checks = []
        if S_node < tau_node:
            failed_checks.append("entity recognition")
        if S_edge < tau_edge:
            failed_checks.append("spatial topology")
        if S_order < tau_order:
            failed_checks.append("reading order")

        if not failed_checks:
            lines.append(
                "All structural checks passed. The trajectory demonstrates correct "
                "spatial reasoning over the ancient-book layout."
            )
        else:
            lines.append(
                f"Please revise the following structural aspects: {', '.join(failed_checks)}. "
                "Use the <zoom> action to isolate dense regions and <call_svm> to extract "
                "precise local symbolic graphs before reasoning about reading order."
            )

        lines.append("")
        lines.append(f"[FGW Transport Plan Summary]: {T_str}")

        return "\n".join(lines)

    def _compute_fgw_verification(
        self,
        local_graph: Dict[str, Any],
        oracle_graph: Dict[str, Any],
    ) -> tuple:
        """
        Simplified FGW-style verification between local and oracle graphs.

        Returns:
            (S_node, S_edge, S_order, summary_str)
        """
        lambda_n = 1.0
        lambda_e = 1.0

        local_nodes = local_graph.get("nodes", [])
        oracle_nodes = oracle_graph.get("nodes", [])
        local_edges = local_graph.get("edges", [])
        oracle_edges = oracle_graph.get("edges", [])

        n_l = max(len(local_nodes), 1)
        n_g = max(len(oracle_nodes), 1)

        # ---- Semantic cost matrix C (bounding-box IoU + type cost) ----
        C = [[0.0] * n_g for _ in range(n_l)]
        for i, ln in enumerate(local_nodes):
            for j, gn in enumerate(oracle_nodes):
                cost = 0.0
                lpoly = ln.get("polygon", [0, 0, 1, 1])
                gpoly = gn.get("polygon", [0, 0, 1, 1])

                # Simplified axis-aligned IoU
                lx1, ly1, lx2, ly2 = lpoly[0], lpoly[1], lpoly[4] if len(lpoly) >= 5 else lpoly[2], lpoly[5] if len(lpoly) >= 6 else lpoly[3]
                gx1, gy1, gx2, gy2 = gpoly[0], gpoly[1], gpoly[4] if len(gpoly) >= 5 else gpoly[2], gpoly[5] if len(gpoly) >= 6 else gpoly[3]

                inter_x1 = max(lx1, gx1)
                inter_y1 = max(ly1, gy1)
                inter_x2 = min(lx2, gx2)
                inter_y2 = min(ly2, gy2)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                l_area = max(abs(lx2 - lx1) * abs(ly2 - ly1), 1)
                g_area = max(abs(gx2 - gx1) * abs(gy2 - gy1), 1)
                union_area = l_area + g_area - inter_area
                iou = inter_area / (union_area + 1e-6)
                cost += (1.0 - iou) * 0.7

                # Type mismatch cost
                if ln.get("type") != gn.get("type"):
                    cost += 0.3

                C[i][j] = cost

        # ---- Simplified Sinkhorn-like OT ----
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

        # ---- S_node ----
        node_cost = sum(C[i][j] * T[i][j] for i in range(n_l) for j in range(n_g))
        S_node = math.exp(-lambda_n * node_cost)

        # ---- S_edge ----
        local_edge_set = set(
            (e.get("source", ""), e.get("target", ""), e.get("type", ""))
            for e in local_edges
        )
        oracle_edge_set = set(
            (e.get("source", ""), e.get("target", ""), e.get("type", ""))
            for e in oracle_edges
        )
        matched = sum(1 for le in local_edge_set if le in oracle_edge_set)
        total_local_edges = max(len(local_edge_set), 1)
        S_edge = math.exp(-lambda_e * (1.0 - matched / total_local_edges))

        # ---- S_order (Kendall tau on READS_AFTER sequences) ----
        def extract_reads_order(nodes, edges):
            adj = {}
            in_deg = {}
            for n in nodes:
                nid = n.get("node_id", "")
                in_deg[nid] = 0
            for e in edges:
                if e.get("type") == "READS_AFTER":
                    tgt = e.get("target", "")
                    if tgt in in_deg:
                        in_deg[tgt] += 1
            queue = [k for k, v in in_deg.items() if v == 0]
            order = []
            while queue:
                nid = queue.pop(0)
                order.append(nid)
                for e in edges:
                    if e.get("type") == "READS_AFTER" and e.get("source") == nid:
                        t = e.get("target", "")
                        if t in in_deg:
                            in_deg[t] -= 1
                            if in_deg[t] == 0:
                                queue.append(t)
            for n in nodes:
                nid = n.get("node_id", "")
                if nid not in order:
                    order.append(nid)
            return order

        l_order = extract_reads_order(local_nodes, local_edges)
        g_order = extract_reads_order(oracle_nodes, oracle_edges)
        l_rank = {n: i for i, n in enumerate(l_order)}
        g_rank = {n: i for i, n in enumerate(g_order)}

        matched_nodes_l = [n.get("node_id", "") for n in local_nodes]
        matched_nodes_g = [n.get("node_id", "") for n in oracle_nodes]

        concordant = discordant = 0
        n_pairs = len(matched_nodes_l)
        for i in range(n_pairs):
            for j in range(i + 1, n_pairs):
                nid_i_l, nid_j_l = matched_nodes_l[i], matched_nodes_l[j]
                nid_i_g, nid_j_g = matched_nodes_g[i], matched_nodes_g[j]
                ri_l, ri_g = l_rank.get(nid_i_l, i), g_rank.get(nid_i_g, i)
                rj_l, rj_g = l_rank.get(nid_j_l, j), g_rank.get(nid_j_g, j)
                if (ri_l - rj_l) * (ri_g - rj_g) > 0:
                    concordant += 1
                elif (ri_l - rj_l) * (ri_g - rj_g) < 0:
                    discordant += 1

        total_pairs = max(n_pairs * (n_pairs - 1) // 2, 1)
        S_order = max(0.0, 2 * concordant / (n_pairs * (n_pairs - 1) + 1))

        summary = (
            f"FGW verification | nodes={n_l}/{n_g}, "
            f"S_node={S_node:.3f}, S_edge={S_edge:.3f}, S_order={S_order:.3f}"
        )
        return S_node, S_edge, S_order, summary

    # ========================================================================
    # Original action parsing (kept for backward compatibility)
    # ========================================================================

    def _parse_zoom_action(self, action_str: str) -> Optional[ActionResult]:
        """
        Parse <zoom> action with support for both 4-point and 8-point polygon formats.

        Format A (legacy): <zoom> [x1, y1, x2, y2] </zoom>
        Format B (Member C): <zoom> [x1, y1, x2, y2, x3, y3, x4, y4] </zoom>
        """
        # Try 8-point polygon first (Member C format)
        pattern_8 = r'<zoom>\s*\[([\d.\s,]+)\]\s*</zoom>'
        match_8 = re.search(pattern_8, action_str)
        if match_8:
            coords_str = match_8.group(1)
            try:
                coords = [float(x.strip()) for x in coords_str.split(',') if x.strip()]
            except ValueError:
                coords = []

            if len(coords) == 8:
                # Use Member C step_zoom interface
                feedback = self.step_zoom(coords)
                return ActionResult(
                    action_type=ActionType.ZOOM,
                    action_params={"polygon": coords},
                    feedback=feedback,
                    is_terminal=False,
                )
            elif len(coords) >= 4:
                # Fallback: use first 4 values (x1,y1,x2,y2)
                coords_4 = coords[:4]
                feedback = self.step_zoom(coords_4 + [0, 0, 0, 0])  # pad to 8
                return ActionResult(
                    action_type=ActionType.ZOOM,
                    action_params={"polygon": coords_4},
                    feedback=feedback,
                    is_terminal=False,
                )

        # Fallback: legacy 4-integer format
        pattern_4 = r'<zoom>\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*</zoom>'
        match_4 = re.search(pattern_4, action_str)
        if match_4:
            x1, y1, x2, y2 = map(int, match_4.groups())
            coords = [float(x1), float(y1), float(x2), float(y2),
                      float(x2), float(y2), float(x1), float(y1)]
            feedback = self.step_zoom(coords)
            return ActionResult(
                action_type=ActionType.ZOOM,
                action_params={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                feedback=feedback,
                is_terminal=False,
            )
        return None

    def _parse_call_svm_action(self, action_str: str) -> Optional[ActionResult]:
        """Parse <call_svm> action and invoke the Member C step_svm interface."""
        if '<call_svm>' in action_str:
            # Use Member C step_svm interface
            feedback = self.step_svm()
            self.response_idx += 1
            return ActionResult(
                action_type=ActionType.CALL_SVM,
                action_params={},
                feedback=feedback,
                is_terminal=False,
            )
        return None

    def step(self, action: str) -> ActionResult:
        """
        Execute environment step with the given action string.

        This method now routes to Member C interfaces:
          - <zoom> [...] </zoom>  →  step_zoom(polygon)
          - <call_svm>           →  step_svm(image_patch=None)

        Args:
            action: The action string from MLLM.

        Returns:
            ActionResult containing feedback string to append to context.
        """
        # Try to parse as zoom action (now delegates to step_zoom)
        result = self._parse_zoom_action(action)
        if result is not None:
            return result

        # Try to parse as call_svm action (now delegates to step_svm)
        result = self._parse_call_svm_action(action)
        if result is not None:
            return result

        # Unknown action - return empty feedback (continue generation)
        return ActionResult(
            action_type=ActionType.UNKNOWN,
            action_params={},
            feedback="",
            is_terminal=False,
        )
    
    def reset(self):
        """Reset environment state."""
        self.response_idx = 0


class VE_MDP_Environment:
    """
    Placeholder for the full VE-MDP (Visual-Evidence Markov Decision Process) environment.
    
    This would handle:
    - Image region extraction and analysis
    - Spatial topology reasoning for complex ancient book layouts
    - Interactive visual feedback
    """
    
    def __init__(self, image_path: str, device: str = "cuda"):
        self.image_path = image_path
        self.device = device
        self.state = {}
    
    def step(self, action: str) -> ActionResult:
        """
        Execute VE-MDP step.
        
        In full implementation, this would:
        1. Parse the action (zoom coordinates, region classification, etc.)
        2. Extract visual evidence from the image
        3. Generate diagnostic graph report (DGR)
        4. Return formatted feedback
        """
        # Placeholder - would integrate with actual image processing pipeline
        return ActionResult(
            action_type=ActionType.UNKNOWN,
            action_params={},
            feedback="[VE-MDP: Full environment not implemented yet]",
            is_terminal=False
        )
    
    def reset(self):
        """Reset VE-MDP state."""
        self.state = {}