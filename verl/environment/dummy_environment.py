"""Dummy environment for IsoGraph Action Interceptor testing.

Provides a simple `DummyEnvironment.step(action)` that returns a fixed textual
response for testing the action interception flow.
"""
from typing import Any


class DummyEnvironment:
    """A minimal placeholder environment used by the Action Interceptor.

    step(action) accepts either a dict (parsed JSON) or a string and returns
    a short string representing the environment feedback. This allows the
    agent loop to concatenate environment feedback back into the prompt.
    """

    def __init__(self) -> None:
        pass

    def step(self, action: Any) -> str:
        """Perform an environment step.

        Args:
            action: a dict or string describing the requested operation. Example:
                {"action": "zoom", "bbox": [x1, y1, x2, y2]} or
                {"action": "call_svm"}

        Returns:
            A short textual observation from the environment.
        """
        # simple behavior for testing
        if isinstance(action, dict):
            act = action.get("action")
            if act == "zoom":
                bbox = action.get("bbox", [])
                return f"[System: Zoomed to bbox {bbox}]"
            if act == "call_svm":
                return "[System: SVM extracted text X]"
            # fallback
            return f"[System: Executed action {action}]"

        # If given a plain string, echo it in system style
        return f"[System: {str(action)}]"
