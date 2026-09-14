"""Deterministic scripted agents for `sap2-v1`, and a tournament that rates them.

Three agents, in increasing order of how much work they do per decision:

- `random` - uniform over the legal mask. The floor.
- `greedy` - a fixed rule over the decoded observation. No simulation, no
  lookahead: combine when possible, otherwise buy the best-statted pet that
  fits, otherwise end the turn. This is the "baby deterministic script".
- `mc` - one-ply search over the legal mask, scoring each candidate action by
  rolling the match forward with `env.clone()`. Needs the `Forkable` feature
  that `base.py` declares; without it every node expansion would cost a
  replay from the seed.

Every agent is deterministic: given the same seed and the same opponent, each
one replays the same match. `random` and `mc` own a `random.Random` seeded from
the match seed, so "deterministic" here means reproducible, not action-free.

IMPORTANT - what `mc` is and is not. It searches on a clone, and a clone
carries the whole match state: the opponent's team, and the random number
generator words that decide the coming shop rolls and battles. An observation
shows neither. So `mc` is a tooling-grade agent that measures how much room
for improvement the game leaves above a fixed rule. It is NOT a legal ladder
submission, and its rating below is not comparable to a submission's. See
`base.Forkable`'s warning and `docs/adding-an-env.md`.

Usage:

    envs/.venv/bin/python tools/scripted_sap2.py --matches 60
    envs/.venv/bin/python tools/scripted_sap2.py --matches 200 --rollouts 8
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from itertools import combinations

import numpy as np

try:
    from policyclash_envs import make
    from policyclash_envs.base import Forkable, Outcome
except ImportError:
    sys.exit(
        "Couldn't import policyclash_envs. Run this with the venv it's installed\n"
        "into, e.g.:\n\n"
        "  envs/.venv/bin/python tools/scripted_sap2.py\n"
    )

ENV_ID = "sap2-v1"

# Action layout, from sap2.h. Same constants the tests and the visualizer use.
END_TURN = 0
BUY_PET_BASE = 1
SELL_BASE = 6
COMBINE_BASE = 11
REROLL = 21
REPOSITION_BASE = 22
BUY_FOOD_BASE = 32
FREEZE_PET_BASE = 42
FREEZE_FOOD_BASE = 47

IGNORED = 0  # a seat that is not acting has its argument ignored, not validated

# Observation layout, from sap2.h's SAP2_OBS_FLOATS. Mirrors the visualizer.
TEAM_BASE = 4  # after gold(1) lives(1) trophies(1) turn(1)
TEAM_SLOT_WIDTH = 19
SHOP_PET_BASE = TEAM_BASE + 5 * TEAM_SLOT_WIDTH
SHOP_PET_SLOT_WIDTH = 13

# Base stats, from sap2.h's SAP2_BASE_ATK / SAP2_BASE_HP, indexed by species.
BASE_ATK = [0, 2, 3, 1, 2, 2, 2, 2, 1, 4, 3, 0, 1]
BASE_HP = [0, 2, 2, 3, 2, 3, 1, 2, 4, 1, 2, 0, 1]

PAIRS = [(i, j) for i, j in combinations(range(5), 2)]


# ---- decoding the observation ------------------------------------------


def team_species(f: np.ndarray, slot: int) -> int:
    block = f[TEAM_BASE + slot * TEAM_SLOT_WIDTH : TEAM_BASE + slot * TEAM_SLOT_WIDTH + 13]
    return int(np.argmax(block))


def team_stats(f: np.ndarray, slot: int) -> tuple[float, float]:
    base = TEAM_BASE + slot * TEAM_SLOT_WIDTH
    return float(f[base + 13]), float(f[base + 14])


def shop_pet_species(f: np.ndarray, slot: int) -> int:
    base = SHOP_PET_BASE + slot * SHOP_PET_SLOT_WIDTH
    return int(np.argmax(f[base : base + 11]))


def meta(f: np.ndarray) -> dict:
    return {
        "gold": float(f[0]),
        "lives": float(f[1]),
        "trophies": float(f[2]),
        "turn": float(f[3]),
    }


# ---- agents ------------------------------------------------------------


class RandomAgent:
    """Uniform over the legal mask. The rating floor."""

    name = "random"
    needs_fork = False

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def act(self, env, seat: int, obs) -> int:
        legal = np.flatnonzero(obs.legal_actions)
        return int(self.rng.choice(legal.tolist()))


class GreedyAgent:
    """A fixed rule, no simulation. The baby deterministic script.

    The rule, in priority order:

    1. Combine a duplicate pair. A level-up is `max(a, b) + 1` on both stats,
       so it strictly beats holding two copies, and it frees a team slot.
    2. Buy the shop pet with the best attack plus health, if a team slot is
       empty. Board presence dominates at Tier 1: an empty slot contributes
       nothing to a battle.
    3. Buy a pet that duplicates a species already on the team, which sets up
       rule 1 next tick.
    4. End the turn. The rule never rerolls, never sells, never buys food, and
       never freezes - each of those needs a judgement about a future shop
       that a fixed rule cannot make well, and spending the gold on a body is
       the reliable alternative at this roster size.
    """

    name = "greedy"
    needs_fork = False

    def __init__(self, seed: int) -> None:
        del seed  # no randomness: the rule is total over the legal mask

    def act(self, env, seat: int, obs) -> int:
        f = obs.features
        legal = obs.legal_actions

        for idx, (i, j) in enumerate(PAIRS):
            if legal[COMBINE_BASE + idx]:
                return COMBINE_BASE + idx

        team = [team_species(f, s) for s in range(5)]
        buys = [s for s in range(5) if legal[BUY_PET_BASE + s]]
        if buys:
            if any(sp == 0 for sp in team):
                best = max(buys, key=lambda s: self._value(f, s))
                return BUY_PET_BASE + best
            dupes = [s for s in buys if shop_pet_species(f, s) in team]
            if dupes:
                return BUY_PET_BASE + max(dupes, key=lambda s: self._value(f, s))

        return END_TURN

    @staticmethod
    def _value(f: np.ndarray, shop_slot: int) -> float:
        species = shop_pet_species(f, shop_slot)
        hp_bonus = float(f[SHOP_PET_BASE + shop_slot * SHOP_PET_SLOT_WIDTH + 11])
        return BASE_ATK[species] + BASE_HP[species] + hp_bonus


class McAgent:
    """One-ply search, each candidate scored by rollouts on a clone.

    For every legal action: fork the match, apply the action, then play both
    seats to the end of a bounded number of rounds and score the result from
    this seat's view. Pick the best total. Ties break toward the lower action
    index, which keeps the agent deterministic.

    A note on what the rollouts do and do not sample. `clone()` copies the
    random number generator words rather than re-seeding them, so every
    rollout of a single candidate resolves the CURRENT round's battle
    identically - the first round of a rollout is exact lookahead, not a
    sample. Variance appears only past that round, where the random
    continuation actions change both teams and consume the streams
    differently. So `rollouts` above 1 buys nothing at `horizon=1`, and the
    default horizon is therefore 2.
    """

    name = "mc"
    needs_fork = True

    def __init__(self, seed: int, rollouts: int = 4, horizon: int = 2) -> None:
        self.rng = random.Random(seed)
        self.rollouts = rollouts
        self.horizon = horizon
        self.ticks = 0  # simulation ticks burned, for the cost report

    def act(self, env, seat: int, obs) -> int:
        legal = np.flatnonzero(obs.legal_actions).tolist()
        if len(legal) == 1:
            return int(legal[0])

        best_action = int(legal[0])
        best_score = -math.inf
        for action in legal:
            total = 0.0
            for _ in range(self.rollouts):
                total += self._rollout(env, seat, int(action))
            if total > best_score:
                best_score = total
                best_action = int(action)
        return best_action

    def _rollout(self, env, seat: int, action: int) -> float:
        sim = env.clone()
        start_turn = sim.turn
        actions = [IGNORED, IGNORED]
        actions[seat] = action
        actions[1 - seat] = END_TURN
        result = sim.step(*actions)
        self.ticks += 1

        # Stop at the horizon, but never while this seat has no observation:
        # a seat that already ended its shop turn has no features to read, so
        # the rollout keeps going until the round resolves and the next one
        # deals it back in. The per-round action budget bounds that wait.
        while not result.done and (
            sim.turn < start_turn + self.horizon or result.observations[seat] is None
        ):
            step = [IGNORED, IGNORED]
            for s in (0, 1):
                if result.observations[s] is not None:
                    choices = np.flatnonzero(result.observations[s].legal_actions).tolist()
                    step[s] = int(self.rng.choice(choices))
            result = sim.step(*step)
            self.ticks += 1

        return self._score(sim, seat, result)

    @staticmethod
    def _score(sim, seat: int, result) -> float:
        """Trophies gained minus lives lost, from this seat's view. A decided
        match saturates to a win or a loss, which keeps the agent from trading
        a won match for one more trophy."""
        if result.done:
            if result.outcome == Outcome.DRAW:
                return 0.0
            won = result.outcome == (Outcome.PLAYER_0 if seat == 0 else Outcome.PLAYER_1)
            return 100.0 if won else -100.0

        own = result.observations[seat]
        m = meta(own.features)
        return m["trophies"] - (5.0 - m["lives"])


AGENTS = {"random": RandomAgent, "greedy": GreedyAgent, "mc": McAgent}


# ---- match play --------------------------------------------------------


def play(agent_0, agent_1, seed: int) -> tuple[Outcome, int, int]:
    """One match. Returns the outcome, the tick count, and the final turn."""
    env = make(ENV_ID)
    for agent in (agent_0, agent_1):
        if getattr(agent, "needs_fork", False) and not isinstance(env, Forkable):
            sys.exit(f"{agent.name} needs the Forkable feature, which {ENV_ID} does not expose")

    result = env.reset(seed=seed)
    agents = (agent_0, agent_1)
    ticks = 0
    while not result.done:
        actions = [IGNORED, IGNORED]
        for seat in (0, 1):
            obs = result.observations[seat]
            if obs is not None:
                actions[seat] = agents[seat].act(env, seat, obs)
        result = env.step(*actions)
        ticks += 1
    return result.outcome, ticks, env.turn


# ---- rating ------------------------------------------------------------


def bradley_terry(
    wins: dict[tuple[str, str], float], names: list[str], prior: float = 1.0
) -> dict[str, float]:
    """Fit a Bradley-Terry strength for each agent and return it on the Elo
    scale, with the field's mean anchored at 1500.

    Elo updates would also work, but they depend on the order the matches
    arrive in and on a K factor. A round robin has no meaningful order, so a
    maximum likelihood fit over the whole result table is the honest summary.
    The iteration is minorization-maximization, the standard one for this
    model.

    `prior` adds that many drawn games between every pair, split evenly. An
    agent that wins every game has no finite maximum likelihood strength,
    because a larger number always fits better, so an unregularized fit
    diverges and prints a rating that only reports how long the iteration
    ran. A clean sweep is the normal result for this field, so the prior is
    not an edge case here. It makes each rating a conservative bound: the
    true separation is at least this large.
    """
    strength = {n: 1.0 for n in names}
    played = {
        (a, b): wins.get((a, b), 0.0) + wins.get((b, a), 0.0) + prior
        for a in names
        for b in names
        if a != b
    }
    scored = {n: sum(wins.get((n, b), 0.0) + prior / 2 for b in names if b != n) for n in names}

    for _ in range(5000):
        updated = {}
        for a in names:
            denom = sum(played[(a, b)] / (strength[a] + strength[b]) for b in names if a != b)
            updated[a] = scored[a] / denom if denom > 0 else strength[a]
        total = sum(updated.values())
        strength = {n: v * len(names) / total for n, v in updated.items()}

    scale = 400.0 / math.log(10.0)
    logs = {n: math.log(max(s, 1e-12)) for n, s in strength.items()}
    mean = sum(logs.values()) / len(logs)
    return {n: 1500.0 + scale * (v - mean) for n, v in logs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", type=int, default=60, help="matches per ordered pair")
    parser.add_argument("--rollouts", type=int, default=4, help="mc rollouts per candidate")
    parser.add_argument("--horizon", type=int, default=2, help="mc rollout depth, in rounds")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    args = parser.parse_args()

    names = list(AGENTS)

    def build(name: str, seed: int):
        if name == "mc":
            return McAgent(seed, rollouts=args.rollouts, horizon=args.horizon)
        return AGENTS[name](seed)

    wins: dict[tuple[str, str], float] = {}
    records: dict[tuple[str, str], list[int]] = {}
    ticks_total = 0
    mc_sim_ticks = 0
    t0 = time.perf_counter()

    # Every unordered pair plays both seatings for each seed. Seat 0 is not a
    # neutral position, so a one-sided sample would rate the seat, not the
    # agent.
    for a, b in combinations(names, 2):
        record = [0, 0, 0]  # a wins, b wins, draws
        for i in range(args.matches):
            seed = args.seed + i
            for first, second in ((a, b), (b, a)):
                agent_first = build(first, seed * 7919 + 1)
                agent_second = build(second, seed * 7919 + 2)
                outcome, ticks, _ = play(agent_first, agent_second, seed)
                ticks_total += ticks
                for agent in (agent_first, agent_second):
                    mc_sim_ticks += getattr(agent, "ticks", 0)

                if outcome == Outcome.DRAW:
                    record[2] += 1
                    wins[(a, b)] = wins.get((a, b), 0.0) + 0.5
                    wins[(b, a)] = wins.get((b, a), 0.0) + 0.5
                    continue

                winner = first if outcome == Outcome.PLAYER_0 else second
                loser = second if outcome == Outcome.PLAYER_0 else first
                wins[(winner, loser)] = wins.get((winner, loser), 0.0) + 1.0
                record[0 if winner == a else 1] += 1
        records[(a, b)] = record

    elapsed = time.perf_counter() - t0
    ratings = bradley_terry(wins, names)

    print(f"\nsap2-v1 round robin - {args.matches} seeds x 2 seatings per pair")
    print(f"mc: {args.rollouts} rollouts/candidate, horizon {args.horizon} rounds\n")

    print("head to head")
    for (a, b), (wa, wb, dr) in records.items():
        n = wa + wb + dr
        print(f"  {a:>7} vs {b:<7}  {wa:>4}-{wb:<4} ({dr} draws)   {a} winrate {wa / n:.1%}")

    print("\nrating (Bradley-Terry, Elo scale, field mean 1500)")
    for name, r in sorted(ratings.items(), key=lambda kv: -kv[1]):
        print(f"  {name:>7}  {r:7.0f}")

    print(f"\nmatch ticks {ticks_total}, mc simulation ticks {mc_sim_ticks}")
    if ticks_total:
        print(f"search overhead {mc_sim_ticks / ticks_total:.1f}x simulated ticks per real tick")
    print(f"wall clock {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
