"""A small PPO policy, trained by `tools/train_ppo_sap2.py` and loaded here.

Two hidden layers of 128 units, tanh, a policy head and a value head. The
value head is dead weight at play time and stays only so the weight file
matches the trained module.

The policy takes the highest-scoring legal action, which makes it
deterministic. Illegal logits go to a large negative number rather than
negative infinity: a fully masked row cannot occur under the current rules
core, and it must not be able to produce a NaN if that ever changes.

Weights come from `weights.safetensors` in this directory. Training ran 40
iterations of 64 episodes against a pool of the current policy, `greedy`, and
`random`, on seeds below 500000 - disjoint from the arena's evaluation range.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from policyclash_envs.sap2 import NUM_ACTIONS, OBS_FLOATS

# Derived from the env, not pinned: sap2's observation and action layouts
# changed when its rules were corrected against the shipped game (see
# policy-clash-re-tools). A checkpoint trained before that no longer fits, and
# silently reshaping it would produce a policy that plays noise.
OBS = OBS_FLOATS
ACTIONS = NUM_ACTIONS
HIDDEN = 128

WEIGHTS = Path(__file__).with_name("weights.safetensors")


class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(OBS, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, HIDDEN)
        self.pi = nn.Linear(HIDDEN, ACTIONS)
        self.v = nn.Linear(HIDDEN, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = mx.tanh(self.l2(mx.tanh(self.l1(x))))
        return self.pi(h)


class Bot:
    def __init__(self, seed: int) -> None:
        del seed  # the policy is deterministic
        self.model = ActorCritic()
        try:
            self.model.load_weights(str(WEIGHTS))
        except ValueError as exc:
            raise RuntimeError(
                f"{WEIGHTS.name} does not fit the current sap2 layout "
                f"(obs {OBS}, actions {ACTIONS}); retrain with "
                "tools/train_ppo_sap2.py"
            ) from exc
        mx.eval(self.model.parameters())

    def act(self, obs) -> int:
        x = mx.array(np.asarray(obs.features, dtype=np.float32)[None, :])
        mask = mx.array(np.asarray(obs.legal_actions)[None, :])
        logits = self.model(x)
        masked = mx.where(mask, logits, mx.array(-1e9, dtype=logits.dtype))
        return int(mx.argmax(masked, axis=-1).item())


def make_bot(seed: int) -> Bot:
    return Bot(seed)
