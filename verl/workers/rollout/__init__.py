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

from .base import BaseRollout, get_rollout_class
from .hf_rollout import HFRollout
from .naive import NaiveRollout

# IsoGraph environment layer: auto-selects Member C's IsoGraphEnvironment
# (production VE-MDP with FGW + DGR) or DummyEnvironment (pure Python fallback).
# Exports DummyEnvironment as the stable type for type-checking.
from .isograph_env import (
    DummyEnvironment,
    ActionResult,
    ActionType,
    get_environment_class,
    create_environment,
)
from .action_interceptor import (
    ActionInterceptor,
    InterceptedTrajectory,
    TrajectoryStep,
    InterceptState,
)
from .isograph_rollout import IsoGraphRollout

__all__ = [
    "BaseRollout",
    "NaiveRollout",
    "HFRollout",
    "get_rollout_class",
    # IsoGraph components
    "DummyEnvironment",
    "ActionResult",
    "ActionType",
    "get_environment_class",
    "create_environment",
    "ActionInterceptor",
    "InterceptedTrajectory",
    "TrajectoryStep",
    "InterceptState",
    "IsoGraphRollout",
]
