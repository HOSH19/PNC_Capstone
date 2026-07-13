# dashboard — Streamlit app (Phase 5, not yet implemented)

Owns: the Streamlit dashboard visualizing the stability index, sentiment trends,
and recent items per bank.

First-screen concept (mentor discussion, 2026-07-12): search a bank → show
① its sentiment state (3-class, with trend), ② the **standout keywords**
driving that sentiment (explainability), ③ a quarter-over-quarter fundamentals
risk profile with the GP score bands. Build **after** the EDA phase — the EDA
decides what is actually worth putting on screen.

Concept renderings (illustrative data — design artifacts, not model output):

**Watch state** — fundamentals neutral, sentiment negative (Wells Fargo demo):

![First-screen concept, Watch state](concept/first-screen-watch.png)

**Elevated Risk state** — both axes negative at once, the strongest
configuration of the warning signal (Western Alliance demo):

![First-screen concept, Elevated Risk state](concept/first-screen-elevated.png)

- Phase: 5
- Reads: index and score tables (read-only; schema in `db/migrations/` is the only shared contract)
- Writes: nothing
- Owner: TBD

Nothing in this directory is executable yet; teammates own all design decisions here.
