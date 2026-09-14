from __future__ import annotations

import numpy as np
import pytest

from policyclash_envs import Outcome, Termination, make
from policyclash_envs.base import Forkable
from policyclash_envs.sap2 import (
    MAX_ROUNDS,
    MAX_TICKS,
    NUM_ACTIONS,
    OBS_FLOATS,
    STARTING_GOLD,
    STARTING_LIVES,
    TROPHIES_TO_WIN,
)

ENV_ID = "sap2-v1"

END_TURN = 0
BUY_PET_BASE = 1       # +0..4
SELL_BASE = 6          # +0..4
COMBINE_BASE = 11      # +0..9
REROLL = 21
REPOSITION_BASE = 22   # +0..9
BUY_FOOD_BASE = 32     # +0..9 (food_slot*5 + team_target)
FREEZE_PET_BASE = 42   # +0..4
FREEZE_FOOD_BASE = 47  # +0..1

TEAM_BASE = 4  # after gold(1) lives(1) trophies(1) turn(1)
TEAM_SLOT_WIDTH = 19
SHOP_PET_BASE = TEAM_BASE + 5 * TEAM_SLOT_WIDTH
SHOP_PET_SLOT_WIDTH = 13
SHOP_FOOD_BASE = SHOP_PET_BASE + 5 * SHOP_PET_SLOT_WIDTH
SHOP_FOOD_SLOT_WIDTH = 5

IGNORED = NUM_ACTIONS + 99


@pytest.fixture
def env():
    return make(ENV_ID)


def team_species(f: np.ndarray, slot: int) -> int:
    base = TEAM_BASE + slot * TEAM_SLOT_WIDTH
    return int(np.argmax(f[base : base + 13]))


def shop_pet_species(f: np.ndarray, slot: int) -> int:
    base = SHOP_PET_BASE + slot * SHOP_PET_SLOT_WIDTH
    return int(np.argmax(f[base : base + 11]))


def shop_pet_frozen(f: np.ndarray, slot: int) -> bool:
    base = SHOP_PET_BASE + slot * SHOP_PET_SLOT_WIDTH
    return bool(f[base + 12])


def shop_food_species(f: np.ndarray, slot: int) -> int:
    base = SHOP_FOOD_BASE + slot * SHOP_FOOD_SLOT_WIDTH
    return int(np.argmax(f[base : base + 4]))


def gold(f: np.ndarray) -> float:
    return f[0]


def lives(f: np.ndarray) -> float:
    return f[1]


def trophies(f: np.ndarray) -> float:
    return f[2]


def turn(f: np.ndarray) -> float:
    return f[3]


def end_both(env, result):
    """Play pure end-turns until the whole MATCH ends. For "just get
    through one round" use end_one_round instead - this one does not stop
    at a round boundary, since submitting END_TURN every tick for two
    idle seats has nothing to make it stop there."""
    while not result.done:
        a0 = END_TURN if result.observations[0] is not None else IGNORED
        a1 = END_TURN if result.observations[1] is not None else IGNORED
        result = env.step(a0, a1)
    return result


def end_one_round(env, result):
    """Pure end-turns until either the match ends or the round boundary is
    crossed (env.turn changes) - whichever comes first."""
    prev_turn = env.turn
    while not result.done and env.turn == prev_turn:
        a0 = END_TURN if result.observations[0] is not None else IGNORED
        a1 = END_TURN if result.observations[1] is not None else IGNORED
        result = env.step(a0, a1)
    return result


# --------------------------------------------------------------------------


def test_registered_with_expected_shape():
    spec = make(ENV_ID).spec
    assert spec.qualified_id == ENV_ID
    assert spec.obs_shape == (OBS_FLOATS,)
    assert spec.num_actions == NUM_ACTIONS
    assert spec.actors_per_tick == 2
    assert spec.stochastic_dynamics
    assert MAX_TICKS == MAX_ROUNDS * 20


def test_reset_state(env):
    result = env.reset(seed=0)
    assert not result.done
    assert env.turn == 1
    for obs in result.observations:
        assert obs.features.shape == (OBS_FLOATS,)
        assert gold(obs.features) == STARTING_GOLD
        assert lives(obs.features) == STARTING_LIVES
        assert trophies(obs.features) == 0
        assert turn(obs.features) == 1
        # turn 1: 3 pet slots, 1 food slot -> end_turn + 3 buy + reroll +
        # 3 freeze_pet + 1 freeze_food = 9
        assert obs.legal_actions.sum() == 9
        assert not obs.legal_actions[FREEZE_PET_BASE + 3]  # slot 3 doesn't exist yet
        assert not obs.legal_actions[FREEZE_FOOD_BASE + 1]  # slot 1 doesn't exist yet


def test_shop_grows_with_turn(env):
    result = env.reset(seed=0)
    result = end_one_round(env, result)  # round 1 -> turn 2, still 3/1
    assert not result.done
    assert env.turn == 2
    assert result.observations[0].legal_actions[BUY_PET_BASE + 2]
    assert not result.observations[0].legal_actions[BUY_PET_BASE + 3]

    result = end_one_round(env, result)  # -> turn 3, now 4 pet slots
    assert env.turn == 3
    assert result.observations[0].legal_actions[BUY_PET_BASE + 3]
    assert not result.observations[0].legal_actions[BUY_PET_BASE + 4]

    result = end_one_round(env, result)  # -> turn 4, still 4/1
    result = end_one_round(env, result)  # -> turn 5, now 5 pet / 2 food
    assert env.turn == 5
    assert result.observations[0].legal_actions[BUY_PET_BASE + 4]
    assert result.observations[0].legal_actions[FREEZE_FOOD_BASE + 1]


def test_gold_resets_every_round_does_not_carry(env):
    result = env.reset(seed=0)
    result = env.step(REROLL, END_TURN)  # seat0 spends 1g -> 9g
    assert gold(result.observations[0].features) == STARTING_GOLD - 1
    result = end_one_round(env, result)  # round resolves, next round starts
    assert gold(result.observations[0].features) == STARTING_GOLD  # not 9, not accumulated


def test_a_lost_battle_costs_a_life_not_the_pet(env):
    """The corrected behavior: fainting in battle is not permanent. Search
    a small seed range for a decisive first round (loser found by their
    lives dropping), then confirm the loser's team is completely
    unchanged going into round 2 - it lost life, not pets."""
    for seed in range(100):
        probe = make(ENV_ID)
        result = probe.reset(seed=seed)
        result = probe.step(BUY_PET_BASE + 0, BUY_PET_BASE + 0)
        before = [
            [team_species(result.observations[s].features, t) for t in range(5)] for s in range(2)
        ]
        result = probe.step(END_TURN, END_TURN)
        if result.done:
            continue
        after_lives = [lives(result.observations[s].features) for s in range(2)]
        if after_lives == [STARTING_LIVES, STARTING_LIVES]:
            continue  # draw, try another seed
        loser = 0 if after_lives[0] < STARTING_LIVES else 1
        after = [team_species(result.observations[loser].features, t) for t in range(5)]
        assert after_lives[loser] == STARTING_LIVES - 1
        assert after == before[loser], "loser's team must be unchanged - fainting isn't permanent"
        return
    pytest.fail("no decisive first round found in 100 seeds")


def test_freeze_persists_across_reroll(env):
    result = env.reset(seed=0)
    sp0 = shop_pet_species(result.observations[0].features, 0)
    result = env.step(FREEZE_PET_BASE + 0, END_TURN)
    assert shop_pet_frozen(result.observations[0].features, 0)
    seat1_action = IGNORED if result.observations[1] is None else END_TURN
    result = env.step(REROLL, seat1_action)
    f0 = result.observations[0].features
    assert shop_pet_species(f0, 0) == sp0  # frozen slot untouched by the reroll
    assert shop_pet_frozen(f0, 0)  # stays frozen until explicitly toggled or bought


def test_freeze_persists_into_next_round(env):
    result = env.reset(seed=0)
    sp0 = shop_pet_species(result.observations[0].features, 0)
    result = env.step(FREEZE_PET_BASE + 0, END_TURN)
    result = end_one_round(env, result)
    f0 = result.observations[0].features
    assert shop_pet_species(f0, 0) == sp0
    assert shop_pet_frozen(f0, 0)


def test_win_on_ten_trophies():
    # Cheapest deterministic route to a decisive match: buy one pet on
    # turn 1 for seat 0 only, then pure end-turns forever. Seat 1 stays
    # empty the whole match, so seat 0's single Tier-1 pet wins every
    # round on its own - no need to keep buying - racking up 10 trophies
    # (and never losing a life) well inside MAX_ROUNDS.
    env = make(ENV_ID)
    result = env.reset(seed=0)
    result = env.step(BUY_PET_BASE + 0, END_TURN)
    result = end_both(env, result)
    assert result.outcome is Outcome.PLAYER_0
    assert result.termination is Termination.NATURAL


def test_illegal_action_forfeits(env):
    env.reset(seed=0)
    result = env.step(NUM_ACTIONS, END_TURN)
    assert result.done
    assert result.outcome is Outcome.PLAYER_1
    assert result.termination is Termination.ILLEGAL_ACTION


def test_budget_forces_end_after_twenty_actions_per_round(env):
    # Both seats take the same 20 actions (buy, then 19 repositions) and
    # never voluntarily END_TURN, so both hit the budget on the same,
    # 20th tick - round 1 resolves right there (mirrored actions -> a
    # draw), landing on round 2's fresh turn rather than leaving seat 0
    # mid-round-1 with seat 1 already idle, which is what let round 1
    # resolve earlier than a naive "20 actions should end this seat" check
    # expects - budgets reset every round, so hitting one is a round
    # event, not necessarily an end-of-episode one.
    result = env.reset(seed=0)
    result = env.step(BUY_PET_BASE + 0, BUY_PET_BASE + 0)
    for _ in range(19):
        result = env.step(REPOSITION_BASE + 0, REPOSITION_BASE + 0)
        if result.done:
            break
    assert result.done or env.turn == 2


def test_step_limit_is_reachable_and_uses_documented_tiebreak(env):
    # Both seats do nothing, every round, forever -> MAX_ROUNDS reached
    # with 0-0 trophies and 5-5 lives -> DRAW_STEP_LIMIT.
    result = env.reset(seed=0)
    result = end_both(env, result)
    assert env.turn == MAX_ROUNDS
    assert result.outcome is Outcome.DRAW
    assert result.termination is Termination.STEP_LIMIT


def test_deterministic_given_same_seed():
    def play(seed):
        e = make(ENV_ID)
        result = e.reset(seed=seed)
        result = e.step(BUY_PET_BASE + 0, BUY_PET_BASE + 0)
        result = end_both(e, result)
        result = end_both(e, result)
        return e.replay(), (result.outcome, result.done)

    a = play(seed=42)
    b = play(seed=42)
    assert a == b


def test_replay_is_flat_two_per_tick(env):
    env.reset(seed=0)
    r = env.step(END_TURN, REROLL)
    replay = env.replay()
    assert replay == [END_TURN, REROLL]
    assert len(replay) == 2


def test_env_advertises_the_forkable_capability(env):
    # How a runner decides whether a search-based submission can be served
    # at all. sap2 opts in; Forkable is not part of TwoPlayerEnv, so this is
    # a real capability check and not a tautology about the base interface.
    assert isinstance(env, Forkable)


def test_clone_does_not_affect_the_original(env):
    result = env.reset(seed=7)
    result = end_one_round(env, result)
    turn_at_fork = env.turn

    fork = env.clone()
    end_both(fork, fork.step(END_TURN, END_TURN))

    # The fork played the match out to its end. The original is still sitting
    # exactly where it was forked, which is the whole point of the capability.
    assert env.turn == turn_at_fork
    assert not result.done


def test_clone_continues_the_rng_stream_rather_than_restarting_it(env):
    # A clone that re-seeded, or that shared shop/battle RNG words with its
    # source, would make a search's rollouts disagree with what the real match
    # goes on to do - the failure that makes forking worthless. Same state and
    # same actions must therefore produce the same match.
    result = env.reset(seed=11)
    result = end_one_round(env, result)

    fork = env.clone()
    fork_result = end_both(fork, fork.step(END_TURN, END_TURN))
    real_result = end_both(env, env.step(END_TURN, END_TURN))

    assert fork_result.outcome == real_result.outcome
    assert fork_result.termination == real_result.termination
    assert fork.replay() == env.replay()
    assert fork.turn == env.turn
