# Working agreements (mandatory)

## Step-gate workflow
- Work in SMALL units: one migration, one module, or one workflow file per step.
- After EVERY unit of work, STOP and post a review block before touching
  anything else:
  1. WHAT: files created/changed (paths + one line each)
  2. WHY: which requirement/design decision this implements
  3. HOW TO VERIFY: the exact command or SQL I can run to check it
  4. RISKS/QUESTIONS: anything you're unsure about, tradeoffs you made
- Then WAIT for my explicit "go" before the next unit. Never chain
  multiple units in one turn, even if the next step is obvious.
- If a step turns out bigger than expected, stop mid-way and report
  rather than pushing through.

## Hard rules
- Never commit without showing the diff summary first.
- Never add dependencies, tables, or source enum values not in the
  approved plan without flagging it as a question first.