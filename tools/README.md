# tools

## `visualize_sap2.py`

A terminal visualizer for `sap2-v1` matches, driven by the real C engine
(`policyclash_envs.make("sap2-v1")`) — not a reimplementation.

```
uv pip install --python envs/.venv/bin/python rich   # one-time, not an envs/ dependency - see below
envs/.venv/bin/python tools/visualize_sap2.py
envs/.venv/bin/python tools/visualize_sap2.py --seed 42 --pause 0.6
envs/.venv/bin/python tools/visualize_sap2.py --style random --max-rounds 5
```

Run with `envs/.venv`'s Python — `policyclash_envs` is installed editable
there. `rich` isn't declared in `envs/pyproject.toml`: it's this one
terminal tool's dependency, not the rules/adapter package's, and adding it
there would be exactly the kind of adapter bloat `adding-an-env.md` argues
against — install it into that same venv once, as above.

**What it can and can't show, and why.** `sap2_battle` in the C core
resolves a round's fight and returns only the aggregate outcome
(`P0_WIN`/`P1_WIN`/`DRAW`) — it doesn't record a per-exchange trace, by
design (`adding-an-env.md`'s "keep the adapter thin" rule). So this tool
shows exactly what the real engine exposes: each seat's team/shop/gold/
lives/trophies before a round, the shop actions taken that round (decoded
from the actions the env actually received), and the round's result — not
a fabricated blow-by-blow battle animation. A companion project has a
separate JS reimplementation of the battle algorithm, in a browser game,
that does track a trace for exactly the reason a human player wants to
see it; this tool deliberately doesn't duplicate that here, so nothing it
shows can silently drift from what the real engine did.

Caught one real thing worth knowing about while building this: the
turn-3 life-back rule can fire in the same transition that costs a seat
its first life, immediately restoring it — so "Seat X loses a life" from
naively inferring off the trophy count can flatly contradict the very
next panel's displayed life total. The tool reads the actual before/after
lives delta rather than inferring, and calls out the catch-up rule
explicitly when it's what happened.

## `scripted_sap2.py`

Three deterministic scripted agents for `sap2-v1`, and a round robin that
rates them. Answers how much room a fixed rule leaves on the table, which
is the same question as how strong a learned policy has to be before it is
interesting.

```
envs/.venv/bin/python tools/scripted_sap2.py --matches 60
envs/.venv/bin/python tools/scripted_sap2.py --matches 20 --rollouts 1 --horizon 1
```

No `rich` dependency — plain stdout, same venv rule as the visualizer.

- `random` — uniform over the legal mask. The floor.
- `greedy` — a fixed rule, no simulation: combine a duplicate pair, else buy
  the best-statted pet that fits an empty slot, else buy a pet that sets up a
  future combine, else end the turn. Never rerolls, sells, buys food, or
  freezes: each needs a judgement about a future shop that a fixed rule can't
  make well, and at Tier 1 spending the gold on a body is the reliable
  alternative.
- `mc` — one-ply search over the legal mask, each candidate scored by
  rollouts on `env.clone()`. Needs the optional `Forkable` feature.

Measured, 60 seeds × both seatings per pair (360 matches), `mc` at 4
rollouts and horizon 2:

| pair | record | winrate |
|---|---|---|
| `greedy` vs `random` | 120–0 | 100% |
| `mc` vs `random` | 120–0 | 100% |
| `mc` vs `greedy` | 110–10 | 91.7% |

Bradley-Terry on the Elo scale, field mean 1500: `mc` 2050, `greedy` 1649,
`random` 801. A clean sweep has no finite maximum-likelihood strength, so the
fit adds one drawn game per pair as a prior and every number is a
conservative bound on the real separation, not a point estimate.

**The depth is doing the work, not the peeking.** At `--rollouts 1 --horizon
1` — exact lookahead over the current round only — `mc` is a coin flip
against `greedy` (52.5% over 40 matches). Going to horizon 2 takes it to
91.7%. Both configurations see the same cloned state, so the gap is search
depth rather than the information a clone exposes. What a one-round lookahead
cannot see is that this round's best board is often the wrong purchase for
next round's economy.

**`mc` is tooling, not a submission.** A clone carries the opponent's team and
the RNG words behind the coming shop rolls and battles; an observation shows
neither. So this rating is not comparable to a ladder submission's, and a
legal search submission needs the seat-masked fork that no env implements
yet. See `base.Forkable` and `docs/adding-an-env.md`.

Cost: search burns ~784 simulated ticks per real match tick at horizon 2, ~50
at horizon 1. `clone()` is what makes that affordable — replaying from the
seed instead costs 289x more per node at a depth of 30 ticks.
